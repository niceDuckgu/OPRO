"""Helpers to attach OPRO to UNO's dual-stream Flux DiT.

UNO splits image and text into independent single-stream blocks. We apply
OPRO on the image-stream Q/K only (the text stream has no panels). The
context lives on the DiT under ``_opro_ctx`` so each ``DoubleStreamBlock``
processor reads it without changing its forward signature.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from opro import OPROLieLowRank, OPROConfig


def build_uno_opro_modules(
    dit: nn.Module,
    num_panels: int = 2,
    panel_rows: int = 1,
    panel_cols: int = 2,
    rank: int = 32,
    variant: str = "opro",
) -> nn.ModuleList:
    """Create per-layer OPRO modules sized to the UNO DiT (image stream only)."""
    head_dim = getattr(dit, "head_dim", None)
    if head_dim is None:
        # Fall back: hidden_size / num_heads from the DiT config.
        head_dim = int(dit.params.hidden_size) // int(dit.params.num_heads)

    blocks = list(getattr(dit, "double_blocks", [])) + list(getattr(dit, "single_blocks", []))
    n = len(blocks)
    if n == 0:
        raise ValueError("No DoubleStream/SingleStream blocks discovered on UNO DiT.")

    modules = nn.ModuleList([
        OPROLieLowRank(num_panels=num_panels, head_dim=head_dim, rank=rank, variant=variant)
        for _ in range(n)
    ])
    dit.opro_modules = modules
    dit.opro_config = OPROConfig(
        num_panels=num_panels, panel_rows=panel_rows, panel_cols=panel_cols,
        rank=rank, variant=variant,
    )
    return modules


def install_panel_context(
    dit: nn.Module,
    opro_modules: nn.ModuleList,
    panel_ids: torch.Tensor,
) -> None:
    """Attach the context that ``DoubleStreamBlockOproProcessor`` reads."""
    dit._opro_ctx = {
        "modules": opro_modules,
        "img_panel_ids": panel_ids,
    }
