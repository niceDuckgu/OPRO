"""Inference helper: plain FluxFill + LoRA + OPRO on a 3-panel canvas.

Given two reference images of a subject and a prompt, generate a third
panel that depicts the same subject. The OPRO module modulates the
inter-panel attention so identity is transferred from the references
rather than reconstructed from local inpainting cues.

This script is intentionally framework-thin: it loads a FluxFill pipeline
from a local path, attaches LoRA + OPRO weights from a release checkpoint,
and runs generation. Customize the AttnProcessor wrap point as needed.

Usage::

    python -m dreambooth_fluxfill.inference \\
        --flux_path /path/to/FLUX.1-Fill-dev \\
        --ckpt_dir ./output/db_opro \\
        --ref0 path/to/ref0.png --ref1 path/to/ref1.png \\
        --prompt "A wooden cat figurine on a marble counter" \\
        --out generated.png
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opro import compute_panel_ids
from dreambooth_fluxfill.data import build_three_panel_canvas, make_flux_img_ids
from dreambooth_fluxfill.inject import load_opro


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--flux_path", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Directory with opro.pt + opro_config.json (and optionally a LoRA adapter).")
    p.add_argument("--lora_dir", type=str, default=None,
                   help="Optional separate LoRA adapter directory (peft format).")
    p.add_argument("--ref0", type=str, required=True)
    p.add_argument("--ref1", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="generated.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from diffusers import FluxFillPipeline  # type: ignore
    except ImportError as e:
        raise SystemExit("diffusers is required: pip install diffusers") from e

    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)

    if args.lora_dir is not None:
        pipe.load_lora_weights(args.lora_dir)

    # Attach OPRO and panel ids; user processor reads transformer._opro_panel_ids.
    opro_modules = load_opro(pipe.transformer, args.ckpt_dir, dtype=torch.float32, device=str(device))
    img_ids = make_flux_img_ids(args.panel_size, num_panels=3)
    panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=3)
    pipe.transformer._opro_panel_ids = panel_ids.unsqueeze(0).to(device).long()
    pipe.transformer._opro_modules = opro_modules

    # Build masked canvas (right panel blank, left+middle = references)
    ref0 = Image.open(args.ref0)
    ref1 = Image.open(args.ref1)
    canvas, mask = build_three_panel_canvas(
        ref0, ref1, target=None, panel_size=args.panel_size, mask_target=True,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        prompt=args.prompt, image=canvas, mask_image=mask,
        height=args.panel_size, width=args.panel_size * 3,
        num_inference_steps=args.num_steps, guidance_scale=args.guidance,
        generator=generator,
    ).images[0]

    # Crop the rightmost panel as the final output
    target_panel = out.crop((args.panel_size * 2, 0, args.panel_size * 3, args.panel_size))
    target_panel.save(args.out)
    print(f"[done] saved → {args.out}")


if __name__ == "__main__":
    main()
