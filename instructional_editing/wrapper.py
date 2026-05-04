"""Inference wrapper around a *user-supplied* ICEdit clone.

We deliberately do NOT vendor the ICEdit codebase. Instead, you point this
wrapper at a local ICEdit clone (``--icedit_path``) and a joint LoRA+OPRO
checkpoint we publish on Hugging Face Hub. The wrapper:

  1. Inserts ``--icedit_path`` at the head of ``sys.path`` so ``from ic_edit
     import ...`` resolves to the user's clone.
  2. Loads the FluxFill backbone via ICEdit's pipeline factory.
  3. Loads our joint LoRA+OPRO weights — both are part of the same release
     ckpt because they were trained jointly. We do *not* reuse ICEdit's
     released LoRA; the LoRA in our ckpt was learned alongside OPRO.
  4. Attaches the panel context the OPRO-aware AttnProcessor expects and
     runs editing.

Setup (one-time)::

    git clone https://github.com/River-Zhang/ICEdit /your/icedit
    huggingface-cli download <org>/opro-icedit-magicbrush --local-dir ./ckpts/opro_icedit
    pip install -r requirements.txt   # in this repo

Run::

    python -m instructional_editing.wrapper \\
        --icedit_path /your/icedit \\
        --flux_path /path/to/FLUX.1-Fill-dev \\
        --ckpt_dir ./ckpts/opro_icedit \\
        --image source.png \\
        --instruction "Make the cat wear sunglasses" \\
        --out edited.png
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


# ---------------------------------------------------------------
# Diptych canvas builder (reused from dreambooth helper, 2-panel form)
# ---------------------------------------------------------------

def build_diptych(source: Image.Image, panel_size: int = 512) -> tuple[Image.Image, Image.Image]:
    """ICEdit-style diptych: source on the left, masked target on the right."""
    canvas = Image.new("RGB", (panel_size * 2, panel_size), "white")
    canvas.paste(source.convert("RGB").resize((panel_size, panel_size), Image.BICUBIC), (0, 0))
    mask = Image.new("L", (panel_size * 2, panel_size), 0)
    right = Image.new("L", (panel_size, panel_size), 255)
    mask.paste(right, (panel_size, 0))
    return canvas, mask


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint LoRA+OPRO inference on ICEdit (no repo vendoring)")
    p.add_argument("--icedit_path", type=str, required=True,
                   help="Local clone of https://github.com/River-Zhang/ICEdit")
    p.add_argument("--flux_path", type=str, required=True,
                   help="Local FLUX.1-Fill-dev path (loaded via ICEdit's pipeline factory).")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Directory with joint LoRA+OPRO weights (HF Hub snapshot).")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="edited.png")
    return p.parse_args()


# ---------------------------------------------------------------
# Pipeline factory: imports from the user's ICEdit clone
# ---------------------------------------------------------------

def load_icedit_pipeline(icedit_path: str, flux_path: str, ckpt_dir: str, device: str):
    """Resolve `icedit_path` on `sys.path`, then import + build their pipeline.

    We import inside the function so this module is importable in test envs
    that don't have ICEdit installed.
    """
    if not os.path.isdir(icedit_path):
        raise FileNotFoundError(f"ICEdit clone not found at: {icedit_path}")
    sys.path.insert(0, icedit_path)
    try:
        # ICEdit exposes its pipeline factory under different names depending on the
        # release; we try the common ones in order.
        try:
            from ic_edit.pipeline import build_pipeline  # type: ignore
        except ImportError:
            from train.src.train.pipeline import build_pipeline  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Could not import ICEdit's pipeline factory. Make sure --icedit_path "
            "points to a working clone with the official entry points."
        ) from e

    pipe = build_pipeline(flux_path=flux_path, lora_dir=ckpt_dir).to(device)
    return pipe


# ---------------------------------------------------------------
# Prompt template (matches the paper Sec. 4.2 wording)
# ---------------------------------------------------------------

DIPTYCH_TEMPLATE = (
    "A diptych with two side-by-side images of the same scene. "
    "On the right, the scene is identical to the left but {instruction}."
)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load ICEdit pipeline (their FluxFill + LoRA stack).
    pipe = load_icedit_pipeline(args.icedit_path, args.flux_path, args.ckpt_dir, str(device))

    # 2) Load OPRO from our checkpoint dir; same dir holds the LoRA the
    #    user-supplied ICEdit code already loaded.
    opro_modules = load_opro(pipe.transformer, args.ckpt_dir, dtype=torch.float32, device=str(device))

    # 3) Attach panel context the OPRO-aware AttnProcessor expects.
    img_ids = make_flux_img_ids(args.panel_size, num_panels=2)
    panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=2)
    pipe.transformer._opro_panel_ids = panel_ids.unsqueeze(0).to(device).long()
    pipe.transformer._opro_modules = opro_modules

    # 4) Build diptych and run.
    source = Image.open(args.image)
    canvas, mask = build_diptych(source, panel_size=args.panel_size)
    prompt = DIPTYCH_TEMPLATE.format(instruction=args.instruction)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = pipe(
        prompt=prompt, image=canvas, mask_image=mask,
        height=args.panel_size, width=args.panel_size * 2,
        num_inference_steps=args.num_steps, guidance_scale=args.guidance,
        generator=generator,
    ).images[0]

    edited = out.crop((args.panel_size, 0, args.panel_size * 2, args.panel_size))
    edited.save(args.out)
    print(f"[done] saved → {args.out}")


if __name__ == "__main__":
    main()
