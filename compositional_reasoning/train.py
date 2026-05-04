"""
Training Script for 2-Stage Compositional Reasoning (Section 4.1)
==================================================================

Stage 1: Single-Panel Pretext
  - Train ViT from scratch on arrow orientation sum classification (8-way).
  - 50k training / 1k validation images at 224x224.

Stage 2: Grid Reasoning Fine-tuning
  - Load Stage 1 checkpoint and freeze backbone.
  - Fine-tune with LoRA (r=8) + OPRO (rho=8).

Usage:
  # Stage 1
  python train.py --stage 1 --pos_type rope --size base

  # Stage 2
  python train.py --stage 2 --pretrain_ckpt output/stage1/best.pt \\
      --grid_size 3 --use_opro --use_lora --freeze_backbone
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import ArrowGridDataset
from model import ViTWithOPRO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2-Stage Compositional Reasoning")

    p.add_argument("--stage", type=int, default=2, choices=[1, 2])

    # Model
    p.add_argument("--size", type=str, default="base",
                   choices=["tiny", "small", "base", "large"])
    p.add_argument("--pos_type", type=str, default="rope", choices=["ape", "rope"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)

    # OPRO / LoRA (Stage 2)
    p.add_argument("--use_opro", action="store_true")
    p.add_argument("--opro_rank", type=int, default=8)
    p.add_argument("--opro_variant", type=str, default="opro",
                   choices=["opro", "abl1", "abl2"])
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--freeze_backbone", action="store_true")

    # Grid (Stage 2)
    p.add_argument("--grid_size", type=int, default=3)

    # Training
    p.add_argument("--max_steps", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--val_interval", type=int, default=1000)

    # Data
    p.add_argument("--train_samples", type=int, default=50_000)
    p.add_argument("--val_samples", type=int, default=1_000)

    # Checkpointing
    p.add_argument("--pretrain_ckpt", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./output")

    return p.parse_args()


def build_model(args: argparse.Namespace) -> ViTWithOPRO:
    num_panels = args.grid_size ** 2 if args.stage == 2 else 1
    use_opro = args.use_opro and args.stage == 2
    use_lora = args.use_lora and args.stage == 2

    model = ViTWithOPRO(
        size=args.size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=8,
        pos_type=args.pos_type,
        num_panels=num_panels,
        opro_rank=args.opro_rank,
        opro_variant=args.opro_variant,
        use_opro=use_opro,
        use_lora=use_lora,
        lora_rank=args.lora_rank,
    )

    if args.stage == 2 and args.pretrain_ckpt is not None:
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"Loaded Stage 1 checkpoint: {args.pretrain_ckpt}")

    if args.freeze_backbone and args.stage == 2:
        model.freeze_backbone()

    return model


@torch.no_grad()
def evaluate(model: ViTWithOPRO, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        panel_labels = batch.get("panel_labels")
        if panel_labels is not None:
            panel_labels = panel_labels.to(device)
        logits = model(pixels, panel_labels)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    n = args.grid_size if args.stage == 2 else 1

    train_ds = ArrowGridDataset(stage=args.stage, n=n, num_samples=args.train_samples,
                                patch_size=args.patch_size)
    val_ds = ArrowGridDataset(stage=args.stage, n=n, num_samples=args.val_samples,
                              patch_size=args.patch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage {args.stage} | {args.size.upper()} | pos={args.pos_type}")
    print(f"  Total: {total:,}  Trainable: {trainable:,}")

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    step = 0
    history_path = Path(args.output_dir) / "history.csv"
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_acc", "lr"])

    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            pixels = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            panel_labels = batch.get("panel_labels")
            if panel_labels is not None:
                panel_labels = panel_labels.to(device)

            logits = model(pixels, panel_labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_interval == 0:
                print(f"  step {step:6d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

            if step % args.val_interval == 0:
                acc = evaluate(model, val_loader, device)
                print(f"  [VAL] step {step:6d} | acc {acc:.4f}")

                with open(history_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, f"{loss.item():.4f}", f"{acc:.4f}",
                                            f"{scheduler.get_last_lr()[0]:.2e}"])

                if acc > best_acc:
                    best_acc = acc
                    torch.save({
                        "step": step, "model": model.state_dict(),
                        "best_acc": best_acc, "args": vars(args),
                    }, Path(args.output_dir) / "best.pt")
                    print(f"  Saved best (acc={best_acc:.4f})")

    print(f"\nDone. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train(parse_args())
