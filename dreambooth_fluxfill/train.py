"""Reference training loop: plain FluxFill + LoRA + OPRO on DreamBooth (3-panel).

This script is the *Track A reference implementation* from the OPRO release —
it deliberately uses the vanilla FluxFill checkpoint (not ICEdit / ACE++ /
InsertAnything) so the integration story stays minimal and any user can copy
the same pattern into their own backbone.

The training loop intentionally mirrors the supplementary protocol
(Sec. A): 2 reference panels + 1 fully masked target panel, AdamW with
lr=1e-4, ~2k steps. We do NOT include FluxFill weights — the script
expects a local FluxFill checkpoint path.

Usage::

    python -m dreambooth_fluxfill.train \\
        --flux_path /path/to/FLUX.1-Fill-dev \\
        --data_root /path/to/dreambooth_dataset \\
        --output_dir ./output/db_opro \\
        --max_steps 2000

The script is intentionally short and forks gracefully if ``diffusers`` /
``peft`` are not installed; in that case we print install instructions
instead of crashing the import (so the smoke tests can still load this
module).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opro import OPROConfig, compute_panel_ids
from dreambooth_fluxfill.data import (
    DreamBoothThreePanelDataset,
    discover_subjects,
    make_flux_img_ids,
)
from dreambooth_fluxfill.inject import (
    inject_opro,
    apply_opro_to_attention,
    save_opro,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plain FluxFill + LoRA + OPRO on DreamBooth (3-panel)")
    p.add_argument("--config", type=str, default=None, help="Optional yaml; CLI overrides win.")
    p.add_argument("--flux_path", type=str, required=True, help="Local FLUX.1-Fill-dev path.")
    p.add_argument("--data_root", type=str, required=True, help="DreamBooth subjects directory.")
    p.add_argument("--output_dir", type=str, default="./output/db_opro")
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--opro_rank", type=int, default=32)
    p.add_argument("--opro_variant", type=str, default="opro",
                   choices=["opro", "opro_bd", "abl1", "abl2"])
    p.add_argument("--save_interval", type=int, default=500)
    args = p.parse_args()

    if args.config:
        with open(args.config) as f:
            ycfg = yaml.safe_load(f) or {}
        for k, v in ycfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
    return args


def _load_flux(flux_path: str):
    """Lazy import + load FluxFill so the module is importable without diffusers."""
    try:
        from diffusers import FluxFillPipeline  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "diffusers is required for actual training. Install with:\n"
            "  pip install -r requirements.txt"
        ) from e
    pipe = FluxFillPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
    return pipe


def _attach_lora(transformer, rank: int):
    try:
        from peft import LoraConfig, inject_adapter_in_model  # type: ignore
    except ImportError as e:
        raise SystemExit("peft is required. pip install peft.") from e

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    inject_adapter_in_model(lora_cfg, transformer)
    return [p for p in transformer.parameters() if p.requires_grad]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    subjects = discover_subjects(args.data_root)
    print(f"[data] {len(subjects)} subjects discovered")
    ds = DreamBoothThreePanelDataset(subjects, panel_size=args.panel_size,
                                     epoch_length=args.max_steps * args.batch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # Model: FluxFill + LoRA + OPRO
    # ------------------------------------------------------------------
    pipe = _load_flux(args.flux_path)
    transformer = pipe.transformer.to(device).train()

    # Freeze everything first; LoRA + OPRO add the only trainable params.
    for p in transformer.parameters():
        p.requires_grad_(False)

    lora_params = _attach_lora(transformer, args.lora_rank)
    opro_modules = inject_opro(
        transformer,
        num_panels=3, panel_rows=1, panel_cols=3,
        rank=args.opro_rank, variant=args.opro_variant,
        dtype=torch.float32, device=str(device),
    )
    opro_params = list(opro_modules.parameters())
    optim = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": opro_params, "lr": args.lr}],
        weight_decay=0.01, betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda")

    # ------------------------------------------------------------------
    # Pre-build static img_ids and panel_ids (canvas geometry is constant).
    # ------------------------------------------------------------------
    img_ids = make_flux_img_ids(args.panel_size, num_panels=3)
    panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=3)
    panel_ids = panel_ids.unsqueeze(0).to(device).long()

    # ------------------------------------------------------------------
    # OPRO call site: attach panel context + swap attention processor.
    #
    # The diffusers FluxAttnProcessor2_0 applies RoPE inside its forward().
    # To inject OPRO post-RoPE, you need a thin wrapper processor that calls
    # apply_opro_to_attention() on the image-token slice of (q, k) and then
    # delegates to the original scaled_dot_product_attention.
    #
    # See integrations/ for ready-made wrappers per baseline:
    #   - integrations/ace_plus/   (single-stream Flux variant)
    #   - integrations/insert_anything/   (FluxPriorRedux + dual context)
    #   - integrations/uno/         (UNO double-stream blocks)
    #
    # The minimal pattern is mirrored in dreambooth_fluxfill/inject.py
    # under apply_opro_to_attention() — that helper is dtype-/device-clean
    # and can be dropped into your attention processor as 3 lines.
    # ------------------------------------------------------------------
    transformer._opro_panel_ids = panel_ids   # picked up by the user-supplied processor

    # ------------------------------------------------------------------
    # Standard FluxFill flow-matching loss (rectified flow / Euler).
    # Replace this with your project's own loss if you need a custom schedule.
    # ------------------------------------------------------------------
    vae = pipe.vae.to(device).eval()
    text_encode = pipe.encode_prompt   # diffusers helper
    scheduler = pipe.scheduler

    step = 0
    optim.zero_grad()
    for batch in loader:
        if step >= args.max_steps:
            break
        canvas = batch["canvas"].to(device, dtype=torch.bfloat16)
        mask = batch["mask"].to(device, dtype=torch.bfloat16)
        target = batch["target"].to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            latents = vae.encode(canvas * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            target_latents = vae.encode(target * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            prompt_embeds, pooled_embeds, text_ids = text_encode(
                prompt=batch["prompt"], device=device,
            )

        # Rectified-flow timestep + noisy latent
        bsz = latents.shape[0]
        t = torch.rand(bsz, device=device, dtype=latents.dtype)
        noise = torch.randn_like(latents)
        noisy = (1 - t.view(-1, 1, 1, 1)) * target_latents + t.view(-1, 1, 1, 1) * noise

        velocity = transformer(
            hidden_states=noisy, encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds, timestep=t * 1000,
            img_ids=img_ids.to(device), txt_ids=text_ids,
            return_dict=False,
        )[0]
        loss = F.mse_loss(velocity.float(), (noise - target_latents).float())

        scaler.scale(loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        if step % 50 == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f}")

        if step and step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_opro(opro_modules, transformer.opro_config, ckpt_dir)
            transformer.save_pretrained(os.path.join(ckpt_dir, "transformer"))

        step += 1

    save_opro(opro_modules, transformer.opro_config, args.output_dir)
    print(f"[done] step={step}, ckpt → {args.output_dir}")


if __name__ == "__main__":
    main()
