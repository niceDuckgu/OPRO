"""Few-sample optimization smoke test for plain FluxFill + LoRA + OPRO.

This script picks ONE DreamBooth subject (e.g. ``cat``) and runs a short
flow-matching optimization on a 3-panel canvas to verify three things:

  1. The pipeline runs end-to-end on GPU (load FluxFill → LoRA + OPRO → step).
  2. Loss decreases monotonically over a handful of steps.
  3. Adding OPRO does not break training: the LoRA+OPRO loss trajectory
     stays in the same ballpark as LoRA-only.

This is a *smoke test* — not full benchmarking. We do not generate images
or compute DINO/CLIP-I; the unit tests under ``tests/`` cover correctness
properties (isometry, same-panel invariance, etc.) on synthetic data.

Usage::

    CUDA_VISIBLE_DEVICES=7 python scripts/smoke_dreambooth_fluxfill.py \\
        --subject cat --num_steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opro import compute_panel_ids                        # noqa: E402
from dreambooth_fluxfill.data import (                    # noqa: E402
    build_three_panel_canvas, make_flux_img_ids,
)
from dreambooth_fluxfill.inject import inject_opro        # noqa: E402
from dreambooth_fluxfill.processor import install_opro_processors  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--flux_path", type=str, required=True,
                   help="Local FLUX.1-Fill-dev path (HF cache snapshot dir).")
    p.add_argument("--data_root", type=str, required=True,
                   help="DreamBooth dataset root with one directory per subject.")
    p.add_argument("--subject", type=str, default="cat")
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--opro_rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, choices=["lora_only", "lora_opro"], default="lora_opro")
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def _build_batch(subject_dir: Path, panel_size: int):
    """Pick 3 random shots from the subject and tile them into a 3-panel canvas."""
    shots = sorted(p for p in subject_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    chosen = random.sample(shots, k=3)
    imgs = [Image.open(p) for p in chosen]
    canvas, mask = build_three_panel_canvas(imgs[0], imgs[1], imgs[2], panel_size=panel_size)

    # Convert to tensors in [-1, 1] for VAE encode.
    import numpy as np
    canvas_t = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0 * 2 - 1
    mask_t = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
    return canvas_t.unsqueeze(0), mask_t.unsqueeze(0)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    print(f"[setup] mode={args.mode}, steps={args.num_steps}, subject={args.subject}")
    print(f"[setup] device={torch.cuda.get_device_name(0)} | free={torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # -----------------------------------------------------------------
    # Load FluxFill (bf16). The Pipeline pulls VAE + T5 + CLIP + Transformer.
    # -----------------------------------------------------------------
    from diffusers import FluxFillPipeline
    from peft import LoraConfig, inject_adapter_in_model

    t0 = time.time()
    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16).to(device)
    print(f"[setup] FluxFill loaded in {time.time()-t0:.1f}s")

    transformer = pipe.transformer
    transformer.requires_grad_(False)
    transformer.train()

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    inject_adapter_in_model(lora_cfg, transformer)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    print(f"[setup] LoRA params: {sum(p.numel() for p in lora_params)/1e6:.2f}M")

    if args.mode == "lora_opro":
        opro_modules = inject_opro(
            transformer, num_panels=3, panel_rows=1, panel_cols=3,
            rank=args.opro_rank, variant="opro",
            dtype=torch.float32, device=str(device),
        )
        install_opro_processors(transformer, opro_modules)
        opro_params = list(opro_modules.parameters())
        print(f"[setup] OPRO params: {sum(p.numel() for p in opro_params)/1e6:.2f}M")
    else:
        opro_params = []

    optim = torch.optim.AdamW(lora_params + opro_params, lr=args.lr, weight_decay=0.01)

    # -----------------------------------------------------------------
    # Pre-compute panel ids (geometry is constant per training run).
    # -----------------------------------------------------------------
    img_ids = make_flux_img_ids(args.panel_size, num_panels=3).to(device)
    panel_ids = compute_panel_ids(img_ids.cpu(), panel_rows=1, panel_cols=3)
    if args.mode == "lora_opro":
        transformer._opro_ctx = {
            "panel_ids": panel_ids.unsqueeze(0).long(),
            "txt_seq_len": 512,  # FluxFill uses 512 T5 tokens; updated each step
        }

    subject_dir = Path(args.data_root) / args.subject
    if not subject_dir.is_dir():
        raise SystemExit(f"Subject not found: {subject_dir}")

    # -----------------------------------------------------------------
    # Training loop: a few flow-matching steps on this subject.
    # -----------------------------------------------------------------
    vae = pipe.vae
    losses: list[float] = []

    for step in range(args.num_steps):
        canvas, mask = _build_batch(subject_dir, args.panel_size)
        canvas = canvas.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            # Encode the full canvas (we want the model to predict it).
            latents = (vae.encode(canvas).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
            # Encode the prompt
            prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
                prompt=f"A triptych of three side-by-side photos of the same {args.subject}.",
                prompt_2=None, device=device, num_images_per_prompt=1, max_sequence_length=512,
            )

        if args.mode == "lora_opro":
            transformer._opro_ctx["txt_seq_len"] = prompt_embeds.shape[1]

        # Pack latents into Flux's patchified shape [B, S, C*4]
        bsz, c, lat_h, lat_w = latents.shape
        latents_packed = pipe._pack_latents(latents, bsz, c, lat_h, lat_w)

        latent_image_ids = pipe._prepare_latent_image_ids(bsz, lat_h // 2, lat_w // 2, device, latents.dtype)

        # Flow matching: noise + timestep
        t = torch.rand(bsz, device=device, dtype=latents.dtype)
        noise = torch.randn_like(latents_packed)
        noisy = (1 - t.view(-1, 1, 1)) * latents_packed + t.view(-1, 1, 1) * noise

        # FluxFill conditioning: [packed_masked_latent (64c) | packed_mask (256c)]
        with torch.no_grad():
            masked_canvas = canvas * (1 - mask.to(device, dtype=torch.bfloat16))
            masked_packed, mask_packed = pipe.prepare_mask_latents(
                mask=mask.to(device, dtype=torch.bfloat16),
                masked_image=masked_canvas,
                batch_size=bsz, num_channels_latents=c,
                num_images_per_prompt=1,
                height=canvas.shape[-2], width=canvas.shape[-1],
                dtype=latents.dtype, device=device, generator=None,
            )

        model_input = torch.cat([noisy, masked_packed, mask_packed], dim=-1)

        # FluxFill transformer forward: predicts velocity (flow target)
        velocity = transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            timestep=(t * 1000).to(latents.dtype),
            img_ids=latent_image_ids,
            txt_ids=text_ids,
            guidance=torch.full((bsz,), 30.0, device=device, dtype=latents.dtype),
            return_dict=False,
        )[0]

        target = noise - latents_packed
        loss = F.mse_loss(velocity.float(), target.float())

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + opro_params, 1.0)
        optim.step()

        losses.append(loss.item())
        print(f"  step {step:3d} | loss {loss.item():.4f} | mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    avg_first = sum(losses[:5]) / 5
    avg_last = sum(losses[-5:]) / 5
    summary = {
        "mode": args.mode, "subject": args.subject,
        "num_steps": args.num_steps,
        "losses": losses,
        "avg_first5": avg_first, "avg_last5": avg_last,
        "delta": avg_last - avg_first,
        "lora_params_M": sum(p.numel() for p in lora_params) / 1e6,
        "opro_params_M": sum(p.numel() for p in opro_params) / 1e6,
    }
    print("\n=== summary ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "losses"}, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {args.output_json}")


if __name__ == "__main__":
    main()
