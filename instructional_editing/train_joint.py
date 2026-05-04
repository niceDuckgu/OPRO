"""Joint LoRA + OPRO training script for the instructional-editing checkpoint
we publish on Hugging Face Hub.

This script reproduces the recipe behind the joint LoRA+OPRO weights that
``wrapper.py`` loads from HF. The file is included so reviewers can audit
the training procedure end-to-end without us vendoring ICEdit.

Differences vs. ``dreambooth_fluxfill/train.py`` (Track A):
  * Diptych canvas (2 panels) instead of triptych.
  * Prompt template matches ICEdit's MagicBrush evaluation:
        "A diptych with two side-by-side images of the same scene.
         On the right, the scene is identical to the left but {instruction}."
  * 5,000 steps vs. 2,000 (matches Sec. 4.2 of the paper).

Both LoRA and OPRO are optimized jointly under the same schedule. Only the
fused weights are published on HF (see instructional_editing/README.md).

Usage::

    python -m instructional_editing.train_joint \\
        --flux_path /path/to/FLUX.1-Fill-dev \\
        --data_root /path/to/magicbrush_train \\
        --output_dir ./output/icedit_opro \\
        --max_steps 5000
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opro import compute_panel_ids
from dreambooth_fluxfill.data import make_flux_img_ids
from dreambooth_fluxfill.inject import inject_opro, save_opro


DIPTYCH_TEMPLATE = (
    "A diptych with two side-by-side images of the same scene. "
    "On the right, the scene is identical to the left but {instruction}."
)


# ---------------------------------------------------------------
# Minimal MagicBrush-style dataset (replace with your own loader)
# ---------------------------------------------------------------

class DiptychInstructionDataset(Dataset):
    """Expects ``data_root/{source.png, target.png, instruction.txt}`` per sample."""

    def __init__(self, data_root: str, panel_size: int = 512):
        from pathlib import Path
        self.root = Path(data_root)
        self.samples = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not self.samples:
            raise FileNotFoundError(f"No samples under {data_root}")
        self.panel_size = panel_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        d = self.samples[idx]
        src = Image.open(d / "source.png").convert("RGB").resize(
            (self.panel_size, self.panel_size), Image.BICUBIC)
        tgt = Image.open(d / "target.png").convert("RGB").resize(
            (self.panel_size, self.panel_size), Image.BICUBIC)
        instruction = (d / "instruction.txt").read_text().strip()

        canvas = Image.new("RGB", (self.panel_size * 2, self.panel_size), "white")
        canvas.paste(src, (0, 0))
        canvas.paste(tgt, (self.panel_size, 0))
        mask = Image.new("L", (self.panel_size * 2, self.panel_size), 0)
        right = Image.new("L", (self.panel_size, self.panel_size), 255)
        mask.paste(right, (self.panel_size, 0))

        import numpy as np
        canvas_t = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(np.array(tgt)).permute(2, 0, 1).float() / 255.0

        return {
            "canvas": canvas_t, "mask": mask_t, "target": target_t,
            "prompt": DIPTYCH_TEMPLATE.format(instruction=instruction),
        }


# ---------------------------------------------------------------
# CLI + training
# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--flux_path", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output/icedit_opro")
    p.add_argument("--panel_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--opro_rank", type=int, default=32)
    p.add_argument("--opro_variant", type=str, default="opro",
                   choices=["opro", "opro_bd", "abl1", "abl2"])
    p.add_argument("--save_interval", type=int, default=1000)
    args = p.parse_args()
    if args.config:
        with open(args.config) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                if not hasattr(args, k) or getattr(args, k) is None:
                    setattr(args, k, v)
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from diffusers import FluxFillPipeline  # type: ignore
        from peft import LoraConfig, inject_adapter_in_model  # type: ignore
    except ImportError as e:
        raise SystemExit("pip install diffusers peft") from e

    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16).to(device)
    transformer = pipe.transformer.train()
    for p_ in transformer.parameters():
        p_.requires_grad_(False)

    # Joint adapters
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian",
    )
    inject_adapter_in_model(lora_cfg, transformer)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]

    opro_modules = inject_opro(
        transformer, num_panels=2, panel_rows=1, panel_cols=2,
        rank=args.opro_rank, variant=args.opro_variant,
        dtype=torch.float32, device=str(device),
    )

    optim = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr},
         {"params": list(opro_modules.parameters()), "lr": args.lr}],
        weight_decay=0.01, betas=(0.9, 0.999),
    )
    scaler = torch.amp.GradScaler("cuda")

    img_ids = make_flux_img_ids(args.panel_size, num_panels=2)
    panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=2)
    transformer._opro_panel_ids = panel_ids.unsqueeze(0).to(device).long()

    ds = DiptychInstructionDataset(args.data_root, panel_size=args.panel_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    vae = pipe.vae.to(device).eval()
    text_encode = pipe.encode_prompt

    step = 0
    optim.zero_grad()
    for batch in loader:
        if step >= args.max_steps:
            break
        canvas = batch["canvas"].to(device, dtype=torch.bfloat16)
        target = batch["target"].to(device, dtype=torch.bfloat16)
        prompt = batch["prompt"]

        with torch.no_grad():
            target_latents = vae.encode(target * 2 - 1).latent_dist.sample() * vae.config.scaling_factor
            prompt_embeds, pooled_embeds, text_ids = text_encode(prompt=prompt, device=device)

        bsz = target_latents.shape[0]
        t = torch.rand(bsz, device=device, dtype=target_latents.dtype)
        noise = torch.randn_like(target_latents)
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

        if step % 100 == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f}")

        if step and step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_opro(opro_modules, transformer.opro_config, ckpt_dir)
            transformer.save_pretrained(os.path.join(ckpt_dir, "transformer"))

        step += 1

    save_opro(opro_modules, transformer.opro_config, args.output_dir)
    transformer.save_pretrained(os.path.join(args.output_dir, "transformer"))
    print(f"[done] step={step}, ckpt → {args.output_dir}")


if __name__ == "__main__":
    main()
