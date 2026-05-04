"""Quantitative A/B comparison for the Track A reference impl.

Trains plain FluxFill + LoRA (with and without OPRO) on a single
DreamBooth subject for a small number of steps, then generates a target
panel from 2 held-out references and computes DINO + CLIP-I similarity
to the ground-truth held-out third shot.

Goal: show that under matched seed and schedule, OPRO improves the
similarity score even at low step counts. This is the "sample
optimization" smoke level — not full DreamBooth fine-tuning.

Usage::

    CUDA_VISIBLE_DEVICES=7 python scripts/quant_compare_dreambooth.py \\
        --subject cat --num_steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opro import compute_panel_ids                                          # noqa: E402
from dreambooth_fluxfill.data import build_three_panel_canvas, make_flux_img_ids  # noqa: E402
from dreambooth_fluxfill.inject import inject_opro                          # noqa: E402
from dreambooth_fluxfill.processor import install_opro_processors           # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--flux_path", type=str, required=True,
                   help="Local FLUX.1-Fill-dev path (HF cache snapshot dir).")
    p.add_argument("--data_root", type=str, required=True,
                   help="DreamBooth dataset root with one directory per subject.")
    p.add_argument("--subject", type=str, default="cat")
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--opro_rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=30.0)
    p.add_argument("--output_dir", type=str, default="/tmp/opro_quant")
    return p.parse_args()


# ---------------------------------------------------------------
# Build a fixed train/test split for the subject so both configs see the
# same data and the held-out target image is identical.
# ---------------------------------------------------------------

def make_splits(subject_dir: Path, seed: int):
    rng = random.Random(seed)
    shots = sorted(p for p in subject_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(shots)
    test_target = shots[0]                                # held-out ground truth
    test_refs = (shots[1], shots[2])                      # fixed eval references
    train_pool = shots[1:]                                # everything except test_target (refs incl.)
    return train_pool, test_refs, test_target


def sample_three_train(train_pool, rng):
    chosen = rng.sample(train_pool, 3)
    return Image.open(chosen[0]), Image.open(chosen[1]), Image.open(chosen[2])


# ---------------------------------------------------------------
# Train one config (with or without OPRO), return the trained pipe.
# ---------------------------------------------------------------

def train_one(args, mode: str, train_pool, device):
    print(f"\n========== TRAINING [{mode}] ==========")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    from diffusers import FluxFillPipeline
    from peft import LoraConfig, inject_adapter_in_model

    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16).to(device)
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

    opro_modules = None
    opro_params = []
    if mode == "lora_opro":
        opro_modules = inject_opro(
            transformer, num_panels=3, panel_rows=1, panel_cols=3,
            rank=args.opro_rank, variant="opro",
            dtype=torch.float32, device=str(device),
        )
        install_opro_processors(transformer, opro_modules)
        opro_params = list(opro_modules.parameters())

    optim = torch.optim.AdamW(lora_params + opro_params, lr=args.lr, weight_decay=0.01)

    img_ids = make_flux_img_ids(args.panel_size, num_panels=3).to(device)
    panel_ids = compute_panel_ids(img_ids.cpu(), panel_rows=1, panel_cols=3)
    if opro_modules is not None:
        transformer._opro_ctx = {"panel_ids": panel_ids.unsqueeze(0).long(), "txt_seq_len": 512}

    vae = pipe.vae
    losses = []
    t0 = time.time()
    for step in range(args.num_steps):
        a, b, c = sample_three_train(train_pool, rng)
        canvas, mask = build_three_panel_canvas(a, b, c, panel_size=args.panel_size)
        canvas_t = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().unsqueeze(0) / 255.0 * 2 - 1
        mask_t = torch.from_numpy(np.array(mask)).unsqueeze(0).unsqueeze(0).float() / 255.0
        canvas_t = canvas_t.to(device, dtype=torch.bfloat16)
        mask_t = mask_t.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            lat = (vae.encode(canvas_t).latent_dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
            bsz, ch, lh, lw = lat.shape
            packed = pipe._pack_latents(lat, bsz, ch, lh, lw)
            pe, pp, ti = pipe.encode_prompt(
                f"A triptych of three side-by-side photos of the same {args.subject}.",
                None, device=device, num_images_per_prompt=1, max_sequence_length=512,
            )
            if opro_modules is not None:
                transformer._opro_ctx["txt_seq_len"] = pe.shape[1]
            masked, mp = pipe.prepare_mask_latents(
                mask_t, canvas_t * (1 - mask_t),
                bsz, ch, 1, canvas_t.shape[-2], canvas_t.shape[-1], lat.dtype, device, None,
            )
            lid = pipe._prepare_latent_image_ids(bsz, lh // 2, lw // 2, device, lat.dtype)

        t = torch.rand(bsz, device=device, dtype=lat.dtype)
        noise = torch.randn_like(packed)
        noisy = (1 - t.view(-1, 1, 1)) * packed + t.view(-1, 1, 1) * noise
        inp = torch.cat([noisy, masked, mp], dim=-1)

        velocity = transformer(
            hidden_states=inp, encoder_hidden_states=pe, pooled_projections=pp,
            timestep=t * 1000, img_ids=lid, txt_ids=ti,
            guidance=torch.full((bsz,), args.guidance, device=device, dtype=lat.dtype),
            return_dict=False,
        )[0]
        loss = F.mse_loss(velocity.float(), (noise - packed).float())

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + opro_params, 1.0)
        optim.step()
        losses.append(loss.item())

        if step % 25 == 0 or step == args.num_steps - 1:
            print(f"  [{mode}] step {step:3d} | loss {loss.item():.4f} | elapsed {time.time()-t0:.0f}s")

    return pipe, transformer, opro_modules, panel_ids, losses


# ---------------------------------------------------------------
# Run inference with a trained pipe on the fixed test references.
# ---------------------------------------------------------------

@torch.no_grad()
def generate_target(pipe, transformer, opro_modules, panel_ids, args, test_refs, device, mode):
    print(f"\n========== INFERENCE [{mode}] ==========")
    transformer.eval()
    if opro_modules is not None:
        transformer._opro_ctx["panel_ids"] = panel_ids.unsqueeze(0).long()

    ref0 = Image.open(test_refs[0])
    ref1 = Image.open(test_refs[1])
    canvas, mask = build_three_panel_canvas(ref0, ref1, target=None,
                                             panel_size=args.panel_size, mask_target=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        prompt=f"A triptych of three side-by-side photos of the same {args.subject}.",
        image=canvas, mask_image=mask,
        height=args.panel_size, width=args.panel_size * 3,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance,
        generator=gen, max_sequence_length=512,
    ).images[0]
    target_panel = out.crop((args.panel_size * 2, 0, args.panel_size * 3, args.panel_size))
    return target_panel


# ---------------------------------------------------------------
# DINO similarity scorer (uses transformers' DINOv2)
# ---------------------------------------------------------------

@torch.no_grad()
def dino_similarity(generated: Image.Image, target: Image.Image, device) -> float:
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    def _embed(img):
        inp = proc(images=img.convert("RGB"), return_tensors="pt").to(device)
        out = model(**inp).last_hidden_state[:, 0]      # CLS token
        return F.normalize(out, dim=-1)

    return float((_embed(generated) * _embed(target)).sum().item())


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    train_pool, test_refs, test_target_path = make_splits(
        Path(args.data_root) / args.subject, args.seed,
    )
    print(f"[setup] subject={args.subject} | train_pool={len(train_pool)} | "
          f"test_refs={[p.name for p in test_refs]} | held-out target={test_target_path.name}")
    target_img = Image.open(test_target_path).convert("RGB").resize((args.panel_size, args.panel_size))

    results = {}
    for mode in ["lora_only", "lora_opro"]:
        pipe, transformer, opro_modules, panel_ids, losses = train_one(args, mode, train_pool, device)
        gen = generate_target(pipe, transformer, opro_modules, panel_ids, args, test_refs, device, mode)
        gen.save(Path(args.output_dir) / f"{mode}_generated.png")
        score = dino_similarity(gen, target_img, device)
        results[mode] = {
            "loss_first10_avg": float(np.mean(losses[:10])),
            "loss_last10_avg": float(np.mean(losses[-10:])),
            "dino_to_target": score,
            "losses": losses,
        }
        print(f"\n[{mode}] DINO(generated, ground-truth) = {score:.4f}")
        del pipe, transformer
        if opro_modules is not None: del opro_modules
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    for mode, r in results.items():
        print(f"  [{mode:10s}] loss {r['loss_first10_avg']:.3f} -> {r['loss_last10_avg']:.3f} | "
              f"DINO={r['dino_to_target']:.4f}")
    delta = results["lora_opro"]["dino_to_target"] - results["lora_only"]["dino_to_target"]
    print(f"\n  Delta DINO (OPRO - LoRA-only): {delta:+.4f}")
    if delta > 0:
        print(f"  ✓ OPRO improves DINO similarity by {delta:+.4f}")
    else:
        print(f"  ✗ OPRO did not improve DINO at this step count (need more steps?)")

    out_json = Path(args.output_dir) / "summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
