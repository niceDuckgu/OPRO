"""
2-Stage Compositional Reasoning Dataset (Section 4.1)
======================================================

Stage 1 (Single-Panel Pretext):
  - Two arrows in a single image; classify their orientation sum mod 360 (8-way).
  - Distractors (letters) added for robustness.

Stage 2 (Grid Reasoning):
  - n x n grid where each row follows a latent rule.
  - One cell per row is held out; predict the missing arrow orientation (8-way).
  - Rules: Rotation (constant angular offset), Mirror Symmetry.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


NUM_CLASSES = 8
ANGLE_STEP = 45  # degrees
CANVAS_SIZE = 224
ARROW_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
                "#264653", "#F4A261", "#606C38", "#BC6C25"]
DISTRACTOR_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class Rule(Enum):
    ROTATION = "rotation"
    MIRROR = "mirror"


@dataclass
class GridSample:
    image: Image.Image
    label: int
    grid_size: int
    blank_row: int
    blank_col: int
    grid_angles: list[list[Optional[int]]]


# ---------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------

def draw_arrow(
    draw: ImageDraw.ImageDraw,
    cx: float, cy: float,
    angle_deg: float,
    length: float = 20.0,
    color: str = "#E63946",
    width: int = 3,
) -> None:
    rad = math.radians(angle_deg)
    dx = math.cos(rad) * length
    dy = -math.sin(rad) * length
    x1, y1 = cx - dx, cy - dy
    x2, y2 = cx + dx, cy + dy
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    head_len = length * 0.35
    for da in [math.radians(150), math.radians(-150)]:
        hx = x2 + math.cos(rad + da) * head_len
        hy = y2 - math.sin(rad + da) * head_len
        draw.line([(x2, y2), (hx, hy)], fill=color, width=width)


def draw_distractor(draw: ImageDraw.ImageDraw, cx: float, cy: float, cell_size: float) -> None:
    letter = random.choice(DISTRACTOR_LETTERS)
    offset = cell_size * 0.25
    x = cx + random.uniform(-offset, offset)
    y = cy + random.uniform(-offset, offset)
    draw.text((x - 4, y - 6), letter, fill="#BBBBBB")


# ---------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------

def render_grid(
    grid_angles: list[list[Optional[int]]],
    n: int,
    canvas_size: int = CANVAS_SIZE,
    add_distractors: bool = False,
) -> Image.Image:
    img = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(img)
    cell = canvas_size / n
    margin = cell * 0.1

    for i in range(1, n):
        pos = int(i * cell)
        draw.line([(pos, 0), (pos, canvas_size)], fill="#CCCCCC", width=1)
        draw.line([(0, pos), (canvas_size, pos)], fill="#CCCCCC", width=1)

    for row in range(n):
        for col in range(n):
            cx = col * cell + cell / 2
            cy = row * cell + cell / 2
            angle_idx = grid_angles[row][col]

            if angle_idx is None:
                draw.text((cx - 6, cy - 8), "?", fill="#FF0000")
            else:
                angle_deg = angle_idx * ANGLE_STEP
                color = ARROW_COLORS[angle_idx % len(ARROW_COLORS)]
                arrow_len = (cell - 2 * margin) * 0.4
                draw_arrow(draw, cx, cy, angle_deg, length=arrow_len,
                           color=color, width=max(2, int(cell / 30)))

            if add_distractors and random.random() < 0.3:
                draw_distractor(draw, cx, cy, cell)

    return img


# ---------------------------------------------------------------
# Stage 1: Single-panel pretext
# ---------------------------------------------------------------

def generate_stage1_sample(add_distractors: bool = True) -> tuple[Image.Image, int]:
    a1 = random.randint(0, NUM_CLASSES - 1)
    a2 = random.randint(0, NUM_CLASSES - 1)
    label = (a1 + a2) % NUM_CLASSES

    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw = ImageDraw.Draw(img)

    cx1 = CANVAS_SIZE * 0.35 + random.uniform(-15, 15)
    cy1 = CANVAS_SIZE * 0.5 + random.uniform(-15, 15)
    cx2 = CANVAS_SIZE * 0.65 + random.uniform(-15, 15)
    cy2 = CANVAS_SIZE * 0.5 + random.uniform(-15, 15)

    draw_arrow(draw, cx1, cy1, a1 * ANGLE_STEP, length=30, color=ARROW_COLORS[a1])
    draw_arrow(draw, cx2, cy2, a2 * ANGLE_STEP, length=30, color=ARROW_COLORS[a2])

    if add_distractors:
        for _ in range(random.randint(0, 3)):
            dx = random.uniform(20, CANVAS_SIZE - 20)
            dy = random.uniform(20, CANVAS_SIZE - 20)
            draw_distractor(draw, dx, dy, 40)

    return img, label


# ---------------------------------------------------------------
# Stage 2: Grid reasoning
# ---------------------------------------------------------------

def generate_stage2_sample(
    n: int = 3,
    allowed_rules: Optional[list[Rule]] = None,
    add_distractors: bool = False,
) -> GridSample:
    if allowed_rules is None:
        allowed_rules = [Rule.ROTATION] if n == 2 else list(Rule)

    grid_angles: list[list[int]] = []
    for row in range(n):
        rule = random.choice(allowed_rules)
        seed = random.randint(0, NUM_CLASSES - 1)
        delta = random.randint(1, NUM_CLASSES - 1)

        if rule == Rule.ROTATION:
            row_angles = [(seed + c * delta) % NUM_CLASSES for c in range(n)]
        else:  # MIRROR
            half = [random.randint(0, NUM_CLASSES - 1) for _ in range((n + 1) // 2)]
            row_angles = half + half[:n // 2][::-1]

        grid_angles.append(row_angles)

    blank_row = random.randint(0, n - 1)
    blank_col = random.randint(0, n - 1)
    label = grid_angles[blank_row][blank_col]

    display = [row[:] for row in grid_angles]
    display[blank_row][blank_col] = None

    image = render_grid(display, n, add_distractors=add_distractors)
    return GridSample(
        image=image, label=label, grid_size=n,
        blank_row=blank_row, blank_col=blank_col, grid_angles=display,
    )


# ---------------------------------------------------------------
# Panel label computation
# ---------------------------------------------------------------

def compute_panel_labels(
    n: int,
    patch_size: int = 16,
    canvas_size: int = CANVAS_SIZE,
) -> torch.Tensor:
    """Map each ViT patch to its grid cell index [0, n*n)."""
    patches_per_side = canvas_size // patch_size
    cell_patches = patches_per_side // n

    labels = torch.zeros(patches_per_side, patches_per_side, dtype=torch.long)
    for r in range(n):
        for c in range(n):
            panel_id = r * n + c
            labels[r * cell_patches:(r + 1) * cell_patches,
                   c * cell_patches:(c + 1) * cell_patches] = panel_id

    return labels.flatten()


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class ArrowGridDataset(Dataset):
    """PyTorch Dataset for the 2-stage compositional reasoning benchmark.

    Args:
        stage:       1 (single-panel pretext) or 2 (grid reasoning).
        n:           Grid size for stage 2 (2, 3, or 4).
        num_samples: Number of samples per epoch.
        patch_size:  ViT patch size for panel label computation.
        transform:   Optional torchvision transform for the image.
    """

    def __init__(
        self,
        stage: int = 2,
        n: int = 3,
        num_samples: int = 50_000,
        add_distractors: bool = False,
        patch_size: int = 16,
        transform=None,
    ) -> None:
        assert stage in (1, 2)
        self.stage = stage
        self.n = n
        self.num_samples = num_samples
        self.add_distractors = add_distractors
        self.patch_size = patch_size
        self.transform = transform
        self.panel_labels = compute_panel_labels(n, patch_size) if stage == 2 else None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        if self.stage == 1:
            img, label = generate_stage1_sample(self.add_distractors)
        else:
            sample = generate_stage2_sample(n=self.n, add_distractors=self.add_distractors)
            img, label = sample.image, sample.label

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        out = {"pixel_values": img, "label": label}
        if self.panel_labels is not None:
            out["panel_labels"] = self.panel_labels
        return out
