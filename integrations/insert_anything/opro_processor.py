"""Helpers to attach OPRO to InsertAnything's FluxFill + Redux transformer.

The block reads its OPRO context from ``model_config["_opro_ctx"]`` so we
keep the existing block signature intact. The dict is populated once per
forward by the patched ``transformer_forward`` (see ``patch.diff``).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from opro import OPROLieLowRank, OPROConfig


def build_insert_opro_modules(
    transformer: nn.Module,
    num_panels: int = 2,
    panel_rows: int = 1,
    panel_cols: int = 2,
    rank: int = 32,
    variant: str = "opro",
) -> nn.ModuleList:
    inner_dim = getattr(transformer, "inner_dim", None) or transformer.config.inner_dim
    num_heads = getattr(transformer, "num_attention_heads", None) or transformer.config.num_attention_heads
    head_dim = int(inner_dim) // int(num_heads)

    blocks = list(getattr(transformer, "transformer_blocks", [])) + \
             list(getattr(transformer, "single_transformer_blocks", []))
    n = len(blocks)
    if n == 0:
        raise ValueError("No transformer blocks discovered on InsertAnything model.")

    modules = nn.ModuleList([
        OPROLieLowRank(num_panels=num_panels, head_dim=head_dim, rank=rank, variant=variant)
        for _ in range(n)
    ])
    transformer.opro_modules = modules
    transformer.opro_config = OPROConfig(
        num_panels=num_panels, panel_rows=panel_rows, panel_cols=panel_cols,
        rank=rank, variant=variant,
    )
    return modules


def install_panel_context(
    transformer: nn.Module,
    opro_modules: nn.ModuleList,
    panel_ids: torch.Tensor,
    img_start_token: int,
) -> None:
    """Attach the panel context that the patched ``attn_forward`` reads."""
    transformer._opro_ctx = {
        "manager": opro_modules,
        "img_panel_ids": panel_ids,
        "img_start": int(img_start_token),
    }
