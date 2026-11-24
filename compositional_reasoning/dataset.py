"""Synthetic data for the two-stage compositional reasoning benchmark (Sec. 4.1).

Stage 1 (single-panel pretext)
    A 224x224 image with two arrows at random, non-overlapping positions. Each
    arrow points in one of eight directions {0, 45, ..., 315} degrees and the label
    is the sum of the two orientations modulo 360 (8 classes). Letters and
    arrow-like triangles of random size are scattered as distractors.

Stage 2 (grid reasoning)
    The image is an n x n grid (n in {2, 3, 4}) with one arrow per panel. Every
    row is an instance of the same hidden rule; the last panel of the last row
    is blank and the label is the orientation that belongs there (8 classes).

    rotation   each panel is the previous one rotated by a fixed step k * 45 deg
    mirror     the row is left-right symmetric (panel c equals panel n-1-c)

    2x2 grids use rotation only; 3x3 and 4x4 grids draw one of the two rules
    uniformly.

Every sample is generated from ``random.Random(base_seed + index)``. The train
and validation splits use disjoint seed ranges, so no validation image is seen
during training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

NUM_CLASSES = 8
ANGLE_STEP_DEG = 45.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Seed offset of the validation split. Train indices stay far below this value.
VAL_SEED_OFFSET = 1_000_000_007

RULES = ("rotation", "mirror")
DEFAULT_ROTATION_STEPS = (-2, -1, 1, 2)
_LETTERS = "CDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def draw_arrow(size: int, rotation_deg: float) -> Image.Image:
    """Black arrow on a transparent canvas, pointing right at 0 deg (CCW positive)."""
    canvas = max(size * 2, 80)
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = canvas // 2
    head = size // 3
    body_h = size // 4
    draw.polygon(
        [(c + size // 2, c), (c + size // 2 - head, c - head // 2), (c + size // 2 - head, c + head // 2)],
        fill=(0, 0, 0),
    )
    draw.rectangle([c - size // 2, c - body_h // 2, c + size // 2 - head, c + body_h // 2], fill=(0, 0, 0))
    if rotation_deg != 0:
        img = img.rotate(rotation_deg, expand=True)
    return img


def _draw_triangle(size: int, rotation_deg: float) -> Image.Image:
    """Arrow-like distractor: a plain triangle."""
    canvas = max(size * 2, 80)
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = canvas // 2
    t = size // 2
    draw.polygon([(c + t, c), (c - t, c - t), (c - t, c + t)], fill=(0, 0, 0))
    if rotation_deg != 0:
        img = img.rotate(rotation_deg, expand=True)
    return img


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size)  # Pillow >= 10.1
        except TypeError:
            return ImageFont.load_default()


def _draw_letter(letter: str, size: int) -> Image.Image:
    canvas = max(size, 40)
    img = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(size)
    left, top, right, bottom = draw.textbbox((0, 0), letter, font=font)
    draw.text((canvas // 2 - (right - left) // 2, canvas // 2 - (bottom - top) // 2), letter,
              fill=(0, 0, 0), font=font)
    return img


def _boxes_overlap(pos_a: Tuple[int, int], size_a: int, pos_b: Tuple[int, int], size_b: int) -> bool:
    ax, ay = pos_a
    bx, by = pos_b
    return not (
        ax + size_a // 2 < bx - size_b // 2
        or ax - size_a // 2 > bx + size_b // 2
        or ay + size_a // 2 < by - size_b // 2
        or ay - size_a // 2 > by + size_b // 2
    )


# ---------------------------------------------------------------------------
# Stage 1: two-arrow orientation sum
# ---------------------------------------------------------------------------

def generate_stage1_sample(
    rng: random.Random,
    label: Optional[int] = None,
    image_size: int = 224,
    angle_jitter_deg: float = 0.5,
    max_distractors: int = 8,
) -> Tuple[Image.Image, int]:
    """Render one pretext scene. Returns ``(image, label)`` with label in [0, 8)."""
    if label is None:
        label = rng.randrange(NUM_CLASSES)

    scale = image_size / 224.0
    gray = rng.randint(220, 255)
    scene = Image.new("RGB", (image_size, image_size), (gray, gray, gray))

    arrow_min = max(16, round(32 * scale))
    arrow_max = max(arrow_min + 1, round(64 * scale))
    margin = max(16, round(32 * scale))

    num_distractors = rng.randint(0, max_distractors)
    num_letters = rng.randint(1, min(num_distractors, 6)) if num_distractors > 0 else 0
    num_triangles = num_distractors - num_letters

    size_a = rng.randint(arrow_min, arrow_max)
    size_b = rng.randint(arrow_min, arrow_max)
    largest = max(size_a, size_b)
    lo = margin + largest // 2
    hi = image_size - margin - largest // 2
    if lo > hi:
        lo, hi = margin, image_size - margin

    def random_pos() -> Tuple[int, int]:
        return rng.randint(lo, hi), rng.randint(lo, hi)

    # Distractor candidates are drawn first; those that collide are dropped below.
    candidates: List[Tuple[Image.Image, Tuple[int, int]]] = []
    for _ in range(num_letters):
        letter = rng.choice(_LETTERS)
        pos = random_pos()
        letter_min = max(16, round(20 * scale))
        letter_size = rng.randint(letter_min, max(letter_min + 1, round(40 * scale)))
        candidates.append((_draw_letter(letter, letter_size), pos))
    for _ in range(num_triangles):
        pos = random_pos()
        tri_min = max(16, round(24 * scale))
        tri_size = rng.randint(tri_min, max(tri_min + 1, round(48 * scale)))
        candidates.append((_draw_triangle(tri_size, rng.randrange(NUM_CLASSES) * ANGLE_STEP_DEG), pos))

    # The two orientations sum to the label (mod 8).
    idx_a = rng.randrange(NUM_CLASSES)
    idx_b = (label - idx_a) % NUM_CLASSES
    jitter = lambda: rng.uniform(-angle_jitter_deg, angle_jitter_deg) if angle_jitter_deg > 0 else 0.0
    rot_a = idx_a * ANGLE_STEP_DEG + jitter()
    rot_b = idx_b * ANGLE_STEP_DEG + jitter()

    pos_a = random_pos()
    pos_b = pos_a
    for _ in range(100):
        pos_b = random_pos()
        if not _boxes_overlap(pos_b, size_b, pos_a, size_a):
            break
    else:
        shift = largest + 20
        pos_b = (min(max(pos_a[0] + shift, lo), hi), min(max(pos_a[1] + shift, lo), hi))

    placed_pos = [pos_a, pos_b]
    placed_size = [size_a, size_b]
    objects = [(draw_arrow(size_a, rot_a), pos_a), (draw_arrow(size_b, rot_b), pos_b)]

    def try_place(img: Image.Image, pos: Tuple[int, int], force: bool = False) -> bool:
        extent = max(img.width, img.height)
        if not force and any(_boxes_overlap(pos, extent, p, s) for p, s in zip(placed_pos, placed_size)):
            return False
        placed_pos.append(pos)
        placed_size.append(extent)
        objects.append((img, pos))
        return True

    def random_triangle() -> Tuple[Image.Image, Tuple[int, int]]:
        tri_min = max(16, round(24 * scale))
        tri_size = rng.randint(tri_min, max(tri_min + 1, round(48 * scale)))
        img = _draw_triangle(tri_size, rng.randrange(NUM_CLASSES) * ANGLE_STEP_DEG)
        half = max(img.width, img.height) // 2
        t_lo, t_hi = margin + half, image_size - margin - half
        if t_lo > t_hi:
            t_lo, t_hi = margin, image_size - margin
        return img, (rng.randint(t_lo, t_hi), rng.randint(t_lo, t_hi))

    placed = sum(try_place(img, pos) for img, pos in candidates)
    # Colliding candidates are replaced by triangles so that the scene keeps
    # ``num_distractors`` distractors; the last ones may overlap other objects.
    attempts = 0
    while placed < num_distractors and attempts < max(num_distractors * 10, 20):
        placed += try_place(*random_triangle())
        attempts += 1
    while placed < num_distractors:
        placed += try_place(*random_triangle(), force=True)

    rng.shuffle(objects)
    for img, (cx, cy) in objects:
        x = max(0, min(cx - img.width // 2, image_size - img.width))
        y = max(0, min(cy - img.height // 2, image_size - img.height))
        scene.paste(img, (x, y), img)
    return scene, label


# ---------------------------------------------------------------------------
# Stage 2: grid reasoning
# ---------------------------------------------------------------------------

@dataclass
class GridSample:
    image: Image.Image
    label: int                               # orientation class of the blank panel
    rule: str                                # "rotation" | "mirror"
    grid_size: int
    grid_angles: List[List[Optional[int]]]   # orientation class per panel, None = blank


def allowed_rules(n: int) -> Tuple[str, ...]:
    """Rules available for an n x n grid (rotation only for n = 2)."""
    return ("rotation",) if n == 2 else RULES


def _partition(total: int, parts: int) -> List[int]:
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _render_arrow_panel(width: int, height: int, rotation_deg: float, rng: random.Random,
                        background: Tuple[int, int, int]) -> Image.Image:
    panel = Image.new("RGB", (width, height), background)
    min_dim = max(24, min(width, height))
    # Smaller panels get relatively larger arrows so they stay legible.
    if min_dim <= 60:
        low, high = 0.65, 0.88
    elif min_dim <= 80:
        low, high = 0.55, 0.78
    else:
        low, high = 0.45, 0.62
    size_min = int(min_dim * low)
    size_max = max(int(min_dim * high), size_min + 1)
    arrow = draw_arrow(rng.randint(size_min, size_max), rotation_deg)
    bbox = arrow.getchannel("A").getbbox()
    if bbox is not None:
        arrow = arrow.crop(bbox)

    margin = max(int(min_dim * 0.06), 2)
    max_w = max(width - 2 * margin, 8)
    max_h = max(height - 2 * margin, 8)
    if arrow.width > max_w or arrow.height > max_h:
        s = min(max_w / arrow.width, max_h / arrow.height)
        arrow = arrow.resize((max(8, round(arrow.width * s)), max(8, round(arrow.height * s))), Image.BICUBIC)

    x = max(margin, min((width - arrow.width) // 2, width - arrow.width - margin))
    y = max(margin, min((height - arrow.height) // 2, height - arrow.height - margin))
    panel.paste(arrow, (x, y), arrow)
    return panel


def render_grid(
    grid_angles: Sequence[Sequence[Optional[int]]],
    rng: random.Random,
    image_size: int = 224,
    angle_jitter_deg: float = 0.5,
    background: Tuple[int, int, int] = (245, 245, 245),
    line_color: Tuple[int, int, int] = (210, 210, 210),
    line_width: int = 2,
) -> Image.Image:
    rows, cols = len(grid_angles), len(grid_angles[0])
    col_widths = _partition(image_size, cols)
    row_heights = _partition(image_size, rows)
    grid = Image.new("RGB", (image_size, image_size), background)
    draw = ImageDraw.Draw(grid)

    y = 0
    for r in range(rows):
        x = 0
        for c in range(cols):
            w, h = col_widths[c], row_heights[r]
            angle = grid_angles[r][c]
            if angle is None:
                panel = Image.new("RGB", (w, h), background)
                inset = max(3, min(w, h) // 12)
                ImageDraw.Draw(panel).rectangle([inset, inset, w - inset - 1, h - inset - 1],
                                                outline=line_color, width=2)
            else:
                jitter = rng.uniform(-angle_jitter_deg, angle_jitter_deg) if angle_jitter_deg > 0 else 0.0
                panel = _render_arrow_panel(w, h, angle * ANGLE_STEP_DEG + jitter, rng, background)
            grid.paste(panel, (x, y))
            if line_width > 0:
                draw.rectangle([x, y, x + w - 1, y + h - 1], outline=line_color, width=line_width)
            x += w
        y += row_heights[r]
    return grid


def generate_stage2_sample(
    rng: random.Random,
    n: int = 3,
    rules: Optional[Sequence[str]] = None,
    rotation_steps: Sequence[int] = DEFAULT_ROTATION_STEPS,
    image_size: int = 224,
    angle_jitter_deg: float = 0.5,
) -> GridSample:
    """Render one n x n reasoning grid whose rows share a hidden rule."""
    if n not in (2, 3, 4):
        raise ValueError("grid size n must be 2, 3 or 4")
    options = tuple(rules) if rules is not None else allowed_rules(n)
    for rule in options:
        if rule not in allowed_rules(n):
            raise ValueError(f"rule '{rule}' is not available for a {n}x{n} grid")
    rule = rng.choice(options)

    grid: List[List[Optional[int]]] = []
    if rule == "rotation":
        step = rng.choice(tuple(rotation_steps))
        for _ in range(n):
            start = rng.randrange(NUM_CLASSES)
            grid.append([(start + c * step) % NUM_CLASSES for c in range(n)])
    else:
        for _ in range(n):
            left = [rng.randrange(NUM_CLASSES) for _ in range(n // 2)]
            center = [rng.randrange(NUM_CLASSES)] if n % 2 == 1 else []
            grid.append(left + center + left[::-1])

    label = grid[-1][-1]
    grid[-1][-1] = None
    image = render_grid(grid, rng, image_size=image_size, angle_jitter_deg=angle_jitter_deg)
    return GridSample(image=image, label=int(label), rule=rule, grid_size=n, grid_angles=grid)


def compute_panel_labels(n: int, image_size: int = 224, patch_size: int = 16) -> torch.Tensor:
    """Panel index (row-major, in [0, n*n)) of every ViT patch, shape [num_patches]."""
    side = image_size // patch_size
    if side * patch_size != image_size:
        raise ValueError("image_size must be divisible by patch_size")
    bounds = [int(round(i * side / n)) for i in range(n + 1)]
    labels = torch.zeros(side, side, dtype=torch.long)
    for r in range(n):
        for c in range(n):
            labels[bounds[r]:bounds[r + 1], bounds[c]:bounds[c + 1]] = r * n + c
    return labels.flatten()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _to_tensor(image: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


class _SeededDataset(Dataset):
    def __init__(self, num_samples: int, split: str, seed: int) -> None:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        if num_samples >= VAL_SEED_OFFSET:
            raise ValueError("num_samples is too large for disjoint train/val seed ranges")
        self.num_samples = int(num_samples)
        self.split = split
        self.base_seed = int(seed) + (VAL_SEED_OFFSET if split == "val" else 0)

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, index: int) -> random.Random:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        return random.Random(self.base_seed + index)


class Stage1Dataset(_SeededDataset):
    """Single-panel pretext: classify the sum of two arrow orientations."""

    def __init__(self, num_samples: int = 50_000, split: str = "train", seed: int = 0,
                 image_size: int = 224) -> None:
        super().__init__(num_samples, split, seed)
        self.image_size = image_size

    def render(self, index: int) -> Tuple[Image.Image, int]:
        # Training labels are balanced by index; validation labels are random.
        label = index % NUM_CLASSES if self.split == "train" else None
        return generate_stage1_sample(self._rng(index), label=label, image_size=self.image_size)

    def __getitem__(self, index: int) -> dict:
        image, label = self.render(index)
        return {"pixel_values": _to_tensor(image), "label": label}


class Stage2Dataset(_SeededDataset):
    """Grid reasoning on an n x n grid; also returns the per-patch panel index."""

    def __init__(self, n: int = 3, num_samples: int = 50_000, split: str = "train", seed: int = 0,
                 rules: Optional[Sequence[str]] = None,
                 rotation_steps: Sequence[int] = DEFAULT_ROTATION_STEPS,
                 image_size: int = 224, patch_size: int = 16) -> None:
        super().__init__(num_samples, split, seed)
        self.n = n
        self.rules = tuple(rules) if rules is not None else allowed_rules(n)
        self.rotation_steps = tuple(rotation_steps)
        self.image_size = image_size
        self.panel_labels = compute_panel_labels(n, image_size, patch_size)

    def render(self, index: int) -> GridSample:
        return generate_stage2_sample(self._rng(index), n=self.n, rules=self.rules,
                                      rotation_steps=self.rotation_steps, image_size=self.image_size)

    def __getitem__(self, index: int) -> dict:
        sample = self.render(index)
        return {
            "pixel_values": _to_tensor(sample.image),
            "label": sample.label,
            "panel_labels": self.panel_labels,
        }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Write a few benchmark samples as PNG files.")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--num", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stage1 = Stage1Dataset(num_samples=args.num, split="val")
    for i in range(args.num):
        image, label = stage1.render(i)
        image.save(os.path.join(args.out, f"stage1_{i}_sum{label * 45}.png"))
    for n in (2, 3, 4):
        stage2 = Stage2Dataset(n=n, num_samples=args.num, split="val")
        for i in range(args.num):
            sample = stage2.render(i)
            sample.image.save(os.path.join(args.out, f"stage2_{n}x{n}_{i}_{sample.rule}_answer{sample.label * 45}.png"))
    print(f"wrote samples to {args.out}/")
