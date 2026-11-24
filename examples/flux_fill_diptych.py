"""OPRO on FLUX.1 Fill for diptych-style instructional editing.

The source image is placed on the left half of a 512 x 1024 canvas, the right
half is masked, and the model inpaints the edited image (the layout used for
the inpainting-based baselines in Sec. 4.2). OPRO adds one operator per panel
and per attention layer; with freshly initialized operators the output is
identical to plain FLUX.1 Fill.

    python examples/flux_fill_diptych.py --image input.jpg \
        --instruction "make the sky pink" [--opro_weights path/to/opro_dir]

Requires `pip install "diffusers==0.32.*" transformers accelerate sentencepiece`
and access to black-forest-labs/FLUX.1-Fill-dev. No pretrained OPRO weights are
distributed with this repository; see the training notes at the bottom.
"""

import argparse

import torch
from PIL import Image

from opro import add_opro_to_flux, grid_panel_ids, load_opro_for_flux

PANEL = 512          # pixels per panel
TOKENS = PANEL // 16  # FLUX: 8x VAE downsampling, then 2x2 patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--opro_weights", default=None, help="directory written by FluxOPRO.save")
    parser.add_argument("--output", default="edited.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    # One operator per panel (left = context, right = target) in every attention layer.
    if args.opro_weights:
        opro = load_opro_for_flux(pipe.transformer, args.opro_weights)
    else:
        opro = add_opro_to_flux(pipe.transformer, num_panels=2, rank=32)
    opro.set_panel_ids(grid_panel_ids(TOKENS, 2 * TOKENS, panel_rows=1, panel_cols=2))
    print(f"OPRO parameters: {sum(p.numel() for p in opro.parameters()) / 1e6:.2f}M")

    source = Image.open(args.image).convert("RGB").resize((PANEL, PANEL))
    canvas = Image.new("RGB", (2 * PANEL, PANEL))
    canvas.paste(source, (0, 0))
    mask = Image.new("L", (2 * PANEL, PANEL), 0)
    mask.paste(255, (PANEL, 0, 2 * PANEL, PANEL))

    prompt = (
        "A diptych with two side-by-side images of the same scene. "
        f"On the right, the scene is identical to the left but {args.instruction}"
    )
    result = pipe(
        prompt=prompt, image=canvas, mask_image=mask, height=PANEL, width=2 * PANEL,
        guidance_scale=30.0, num_inference_steps=28,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    result.crop((PANEL, 0, 2 * PANEL, PANEL)).save(args.output)


# Training notes
# --------------
# OPRO is trained jointly with the pipeline's usual adapter (e.g. LoRA). The
# operators live outside the transformer, so add them to the optimizer yourself
# and keep them in float32:
#
#     opro = add_opro_to_flux(transformer, num_panels=2, rank=32)
#     opro.set_panel_ids(grid_panel_ids(32, 64, 1, 2))
#     optimizer = torch.optim.AdamW(
#         [{"params": lora_parameters, "lr": 1e-4},
#          {"params": opro.parameters(), "lr": 1e-4}],
#         weight_decay=0.01,
#     )
#     ...                      # standard flow-matching loss of the pipeline
#     opro.save("output/opro")  # writes opro.pt + opro_config.json

if __name__ == "__main__":
    main()
