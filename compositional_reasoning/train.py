"""Training for the two-stage compositional reasoning benchmark (Sec. 4.1).

Stage 1  trains a ViT from scratch on the single-panel pretext task.
Stage 2  loads the Stage-1 backbone, freezes it, and fine-tunes on n x n grid
         reasoning with a linear probe, LoRA, or LoRA + OPRO.

Run from the repository root:

    # Stage 1 (one backbone per positional encoding)
    python -m compositional_reasoning.train --stage 1 --pos_type rope \
        --output_dir output/stage1_rope

    # Stage 2: linear probe / LoRA / LoRA + OPRO on a 3x3 grid
    python -m compositional_reasoning.train --stage 2 --pos_type rope --grid_size 3 \
        --pretrain_ckpt output/stage1_rope/last.pt --output_dir output/rope_3x3_probe
    python -m compositional_reasoning.train --stage 2 --pos_type rope --grid_size 3 \
        --pretrain_ckpt output/stage1_rope/last.pt --use_lora --output_dir output/rope_3x3_lora
    python -m compositional_reasoning.train --stage 2 --pos_type rope --grid_size 3 \
        --pretrain_ckpt output/stage1_rope/last.pt --use_lora --use_opro \
        --output_dir output/rope_3x3_lora_opro

Defaults follow the supplementary material: AdamW, batch size 256, weight decay
0.05; Stage 1 runs 50k steps at 1e-3 with warmup and cosine decay, Stage 2 runs
2k steps at a constant 5e-4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import NUM_CLASSES, Stage1Dataset, Stage2Dataset
from .model import POS_TYPES, VIT_CONFIGS, ViTClassifier

STAGE_DEFAULTS = {
    1: {"max_steps": 50_000, "lr": 1e-3, "warmup_steps": 1_000, "schedule": "cosine"},
    2: {"max_steps": 2_000, "lr": 5e-4, "warmup_steps": 0, "schedule": "constant"},
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage compositional reasoning benchmark")
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--output_dir", type=str, required=True)

    # Model
    p.add_argument("--size", type=str, default="base", choices=sorted(VIT_CONFIGS))
    p.add_argument("--pos_type", type=str, default="rope", choices=POS_TYPES)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)

    # Stage 2
    p.add_argument("--pretrain_ckpt", type=str, default=None, help="Stage-1 checkpoint (required for stage 2)")
    p.add_argument("--grid_size", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--use_lora", action="store_true", help="LoRA on the Q/K projections")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--use_opro", action="store_true")
    p.add_argument("--opro_rank", type=int, default=8)
    p.add_argument("--opro_variant", type=str, default="opro", choices=["opro", "abl1", "abl2"],
                   help="opro | abl1 (additive panel bias) | abl2 (asymmetric Q/K operators)")
    p.add_argument("--opro_symmetric_init", action="store_true", help="'w/o Zero Init' ablation")

    # Optimization (None = stage default)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--schedule", type=str, default=None, choices=["cosine", "constant"])
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA only)")

    # Data
    p.add_argument("--train_samples", type=int, default=50_000)
    p.add_argument("--val_samples", type=int, default=1_000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--val_interval", type=int, default=None, help="default: every 1000 (stage 1) / 200 (stage 2) steps")

    args = p.parse_args(argv)
    for key, value in STAGE_DEFAULTS[args.stage].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.val_interval is None:
        args.val_interval = 1000 if args.stage == 1 else 200
    if args.stage == 2 and args.pretrain_ckpt is None:
        p.error("--pretrain_ckpt is required for stage 2")
    return args


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * step / max(1, args.warmup_steps)
    if args.schedule == "constant":
        return args.lr
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def build_model(args: argparse.Namespace) -> ViTClassifier:
    model = ViTClassifier(NUM_CLASSES, args.size, args.pos_type, args.image_size, args.patch_size)
    if args.stage == 1:
        return model

    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True)
    model.load_backbone(ckpt["model"])
    if args.use_lora:
        model.add_lora(args.lora_rank, args.lora_alpha)
    if args.use_opro:
        model.add_opro(args.grid_size ** 2, args.opro_rank, args.opro_variant, args.opro_symmetric_init)
    model.freeze_backbone()
    return model


def build_loaders(args: argparse.Namespace):
    if args.stage == 1:
        make = lambda split, n: Stage1Dataset(n, split, args.seed, args.image_size)
    else:
        make = lambda split, n: Stage2Dataset(args.grid_size, n, split, args.seed,
                                              image_size=args.image_size, patch_size=args.patch_size)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                  persistent_workers=args.num_workers > 0)
    train = DataLoader(make("train", args.train_samples), shuffle=True, drop_last=True, **common)
    val = DataLoader(make("val", args.val_samples), shuffle=False, **common)
    return train, val


@torch.no_grad()
def evaluate(model: ViTClassifier, loader: DataLoader, device: torch.device, amp: bool) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        panel_labels = batch["panel_labels"].to(device) if "panel_labels" in batch else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(pixels, panel_labels)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.numel()
    model.train()
    return 100.0 * correct / max(1, total)


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.bf16 and device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model = build_model(args).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in trainable),
        "lora": sum(p.numel() for n, p in model.named_parameters() if "lora_" in n),
        "opro": sum(p.numel() for n, p in model.named_parameters() if n.startswith("operator.")),
    }
    print(f"stage {args.stage} | ViT-{args.size} | pos={args.pos_type} | params {counts}")

    train_loader, val_loader = build_loaders(args)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_acc, step = 0.0, 0
    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            lr = learning_rate(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            panel_labels = batch["panel_labels"].to(device) if "panel_labels" in batch else None
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(model(pixels, panel_labels), labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.log_interval == 0:
                print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e}", flush=True)
            if step % args.val_interval == 0 or step == args.max_steps:
                acc = evaluate(model, val_loader, device, amp)
                best_acc = max(best_acc, acc)
                history.append({"step": step, "loss": loss.item(), "val_acc": acc})
                print(f"step {step:6d} | val acc {acc:.2f}", flush=True)

    result = {"final_val_acc": history[-1]["val_acc"], "best_val_acc": best_acc,
              "params": counts, "history": history}
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step},
               os.path.join(args.output_dir, "last.pt"))
    print(f"done | final val acc {result['final_val_acc']:.2f} | best {best_acc:.2f}")
    return result


if __name__ == "__main__":
    train(parse_args())
