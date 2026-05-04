"""DreamBooth 3-panel canvas builder for plain FluxFill + LoRA + OPRO.

The release supplementary (Sec. A) follows a *3-panel layout* for
subject-driven generation:

    ┌─────────┬─────────┬─────────┐
    │ ref 0   │ ref 1   │ TARGET  │
    │ (visible)│ (visible)│ (masked)│
    └─────────┴─────────┴─────────┘

The model sees the two references on the left, must reconstruct the
masked third panel from the same subject, and OPRO modulates the
inter-panel attention so the synthesis transfers identity from the
references rather than relying on local inpainting cues.

This module exposes:

* ``build_three_panel_canvas`` — pure-PIL canvas + binary mask construction
  used by both the training data loader and the inference helper.
* ``DreamBoothThreePanelDataset`` — a thin reference Dataset that samples
  pairs of references from a directory of subjects. Users can swap in
  any HF Datasets / WebDataset loader of their own.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------
# Pure canvas + mask builder (no FluxFill dependency)
# ---------------------------------------------------------------

def _resize_square(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    return img.resize((size, size), Image.BICUBIC)


def build_three_panel_canvas(
    ref0: Image.Image,
    ref1: Image.Image,
    target: Optional[Image.Image] = None,
    panel_size: int = 512,
    mask_target: bool = True,
) -> tuple[Image.Image, Image.Image]:
    """Tile ``[ref0, ref1, target]`` left-to-right and build the FluxFill mask.

    Args:
        ref0:        Reference image for the left panel.
        ref1:        Reference image for the middle panel.
        target:      Optional ground-truth target panel. If ``None`` and
                     ``mask_target=True`` the right panel is filled white
                     (model has to imagine it from the references).
        panel_size:  Side length of each panel (default 512 → 512x1536 canvas).
        mask_target: If True, the third panel is masked (binary mask = 1 there)
                     so FluxFill inpaints it; everything else stays visible.

    Returns:
        ``(canvas, mask)`` PIL images. The mask is a single-channel L image
        with values in {0, 255}: 255 = inpaint here, 0 = keep.
    """
    canvas = Image.new("RGB", (panel_size * 3, panel_size), "white")
    canvas.paste(_resize_square(ref0, panel_size), (0, 0))
    canvas.paste(_resize_square(ref1, panel_size), (panel_size, 0))
    if target is not None:
        canvas.paste(_resize_square(target, panel_size), (panel_size * 2, 0))

    mask = Image.new("L", (panel_size * 3, panel_size), 0)
    if mask_target:
        # White on the rightmost panel
        right = Image.new("L", (panel_size, panel_size), 255)
        mask.paste(right, (panel_size * 2, 0))
    return canvas, mask


# ---------------------------------------------------------------
# Reference Dataset for multi-shot DreamBooth subjects
# ---------------------------------------------------------------

@dataclass
class SubjectShots:
    name: str
    images: List[Path]


def discover_subjects(root: str | os.PathLike) -> List[SubjectShots]:
    """Discover subject directories under ``root``.

    Expected layout (matching the official DreamBooth release)::

        <root>/<subject_name>/*.{jpg,jpeg,png,webp}

    Returns subjects sorted by name with at least 2 shots.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"DreamBooth root not found: {root}")
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    subjects = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        shots = sorted(p for p in sub.iterdir() if p.suffix.lower() in exts)
        if len(shots) >= 2:
            subjects.append(SubjectShots(name=sub.name, images=shots))
    return subjects


class DreamBoothThreePanelDataset(Dataset):
    """Sample two references + one target from each DreamBooth subject.

    Each ``__getitem__`` returns a dict with the canvas, mask, prompt, and
    panel grid metadata used downstream by ``compute_panel_ids``.

    Args:
        subjects:     Output of ``discover_subjects(...)`` (or a manual list).
        panel_size:   Per-panel resolution (FluxFill standard: 512).
        prompt_template: Format string with ``{name}`` placeholder.
        epoch_length: Number of samples per epoch (we resample with replacement).
    """

    def __init__(
        self,
        subjects: Sequence[SubjectShots],
        panel_size: int = 512,
        prompt_template: str = (
            "A triptych of three side-by-side images of the same {name}. "
            "The first two panels show the {name} from different viewpoints; "
            "the third panel must depict the same {name} consistently."
        ),
        epoch_length: int = 1024,
    ) -> None:
        if not subjects:
            raise ValueError("DreamBoothThreePanelDataset requires >=1 subject")
        self.subjects = list(subjects)
        self.panel_size = int(panel_size)
        self.prompt_template = prompt_template
        self.epoch_length = int(epoch_length)

    def __len__(self) -> int:
        return self.epoch_length

    def _sample_three(self, subject: SubjectShots) -> tuple[Image.Image, Image.Image, Image.Image]:
        chosen = random.sample(subject.images, k=min(3, len(subject.images)))
        if len(chosen) < 3:
            # Re-use the first reference if the subject has only 2 shots
            chosen = chosen + [chosen[0]]
        imgs = [Image.open(p) for p in chosen[:3]]
        return imgs[0], imgs[1], imgs[2]

    def __getitem__(self, idx: int) -> dict:
        subject = self.subjects[idx % len(self.subjects)]
        ref0, ref1, target = self._sample_three(subject)
        canvas, mask = build_three_panel_canvas(
            ref0, ref1, target,
            panel_size=self.panel_size, mask_target=True,
        )

        canvas_t = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        target_t = torch.from_numpy(np.array(_resize_square(target, self.panel_size)))
        target_t = target_t.permute(2, 0, 1).float() / 255.0

        return {
            "canvas": canvas_t,                  # [3, H, W*3]
            "mask": mask_t,                      # [1, H, W*3]
            "target": target_t,                  # [3, H, W]  (right panel ground truth)
            "prompt": self.prompt_template.format(name=subject.name.replace("_", " ")),
            "panel_rows": 1,
            "panel_cols": 3,
            "subject": subject.name,
        }


# ---------------------------------------------------------------
# Convenience: build a Flux-shaped img_ids tensor for a 3-panel canvas
# ---------------------------------------------------------------

def make_flux_img_ids(panel_size: int, num_panels: int = 3) -> torch.Tensor:
    """Build a Flux-style ``img_ids`` tensor ``[seq_len, 3]`` for a 1xN-panel canvas.

    FluxFill patchifies at stride 2 in latent space (latent stride 16 in pixel
    space). For panel_size=512 → latent 64x64 → patch grid 32x(32*N).

    Each row is ``[batch_idx, latent_y, latent_x]``. Caller can pass the result
    straight into ``opro.compute_panel_ids(..., panel_rows=1, panel_cols=N)``.
    """
    latent_h = panel_size // 16
    latent_w = (panel_size * num_panels) // 16
    ids = torch.zeros(latent_h * latent_w, 3)
    for r in range(latent_h):
        for c in range(latent_w):
            idx = r * latent_w + c
            ids[idx, 1] = r
            ids[idx, 2] = c
    return ids
