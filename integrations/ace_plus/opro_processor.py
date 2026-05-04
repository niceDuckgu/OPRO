"""Helpers to attach OPRO to an ACE++ FluxFill transformer.

ACE++ uses a single-stream Flux attention block where text and image
tokens share the joint Q/K/V tensors. OPRO is applied to the image
token slice (`img_start:`) post-RoPE.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from opro import OPROLieLowRank, OPROConfig


def build_ace_opro_modules(
    transformer: nn.Module,
    num_panels: int = 2,
    panel_rows: int = 1,
    panel_cols: int = 2,
    rank: int = 32,
    variant: str = "opro",
) -> nn.ModuleList:
    """Create per-layer OPRO modules sized to the ACE++ transformer."""
    inner_dim = getattr(transformer, "inner_dim", None) or transformer.config.inner_dim
    num_heads = getattr(transformer, "num_attention_heads", None) or transformer.config.num_attention_heads
    head_dim = int(inner_dim) // int(num_heads)

    blocks = list(getattr(transformer, "transformer_blocks", [])) + \
             list(getattr(transformer, "single_transformer_blocks", []))
    n_layers = len(blocks)
    if n_layers == 0:
        raise ValueError("No transformer blocks discovered on ACE++ transformer.")

    modules = nn.ModuleList([
        OPROLieLowRank(num_panels=num_panels, head_dim=head_dim, rank=rank, variant=variant)
        for _ in range(n_layers)
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
    img_start: int = 0,
) -> None:
    """Stash the per-step panel context the patched ``attention_forward`` reads.

    ACE++'s patched forward grabs ``opro_kwargs`` from ``kwargs``; in the
    pipeline path this dict is constructed from these attributes.
    """
    transformer._opro_modules = opro_modules
    transformer._opro_panel_ids = panel_ids
    transformer._opro_img_start = int(img_start)


def opro_kwargs_for_layer(
    transformer: nn.Module,
    layer_idx: int,
) -> dict:
    """Build the ``opro_kwargs`` dict the patched attention reads at layer ``layer_idx``."""
    return {
        "opro_module": transformer._opro_modules[layer_idx],
        "panel_ids": transformer._opro_panel_ids,
        "img_start": transformer._opro_img_start,
    }
