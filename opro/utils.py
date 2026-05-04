"""
OPRO Utilities
===============

Panel ID computation and config for integrating OPRO into tiled-panel layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence
import json

import torch


__all__ = [
    "OPROConfig",
    "compute_panel_ids",
    "remap_panel_ids",
]


@dataclass
class OPROConfig:
    """Configuration for OPRO integration.

    Attributes:
        num_panels:    Number of distinct panel operators.
        panel_rows:    Grid rows for panel ID computation.
        panel_cols:    Grid columns for panel ID computation.
        rank:          Low-rank dimension (rho) for the Lie factors.
        variant:       "opro" (default) | "abl1" (APB) | "abl2" (Asymmetric).
        share_across_layers: If True, all layers share a single OPRO module.
        panel_aliases: Optional remapping table for panel IDs.
    """
    num_panels: int = 2
    panel_rows: int = 1
    panel_cols: int = 2
    rank: int = 32
    variant: str = "opro"
    share_across_layers: bool = False
    panel_aliases: Optional[tuple[int, ...]] = None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OPROConfig":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "OPROConfig":
        """Parse from a YAML/dict config (supports both UPPER and lower keys)."""
        def get(key: str, default):
            return raw.get(key.upper(), raw.get(key, default))
        cfg = cls()
        cfg.num_panels = int(get("num_panels", cfg.num_panels))
        cfg.panel_rows = int(get("panel_rows", cfg.panel_rows))
        cfg.panel_cols = int(get("panel_cols", cfg.panel_cols))
        cfg.rank = int(get("lie_rank", get("rank", cfg.rank)))
        cfg.variant = str(get("variant", cfg.variant)).lower()
        cfg.share_across_layers = bool(get("share_across_layers", cfg.share_across_layers))
        alias_raw = raw.get("PANEL_ALIASES", raw.get("panel_aliases"))
        if alias_raw is not None:
            cfg.panel_aliases = tuple(int(v) for v in alias_raw)
        return cfg


def compute_panel_ids(
    ids: torch.Tensor,
    panel_rows: int,
    panel_cols: int,
    mask: Optional[torch.Tensor] = None,
    panel_aliases: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Compute per-token panel IDs from Flux-style positional coordinates.

    In Flux, each token has coordinates [batch_idx, row, col] stored in ``img_ids``.
    This function bins those coordinates into a ``panel_rows x panel_cols`` grid.

    Args:
        ids:          Token coordinates ``[batch, seq_len, >=3]`` or ``[seq_len, >=3]``.
                      Column 1 = row, column 2 = col.
        panel_rows:   Number of panel rows.
        panel_cols:   Number of panel columns.
        mask:         Optional boolean mask for valid tokens.
        panel_aliases: Optional remapping (e.g. for role-sharing).

    Returns:
        Panel IDs ``[batch, seq_len]`` with -1 for invalid tokens.
    """
    squeeze_batch = False
    if ids.ndim == 2:
        ids = ids.unsqueeze(0)
        squeeze_batch = True
        if mask is not None and mask.ndim == 1:
            mask = mask.unsqueeze(0)

    device = ids.device
    bsz, seq = ids.shape[:2]
    panel_ids = torch.full((bsz, seq), -1, device=device, dtype=torch.long)

    if mask is None:
        mask = torch.ones(bsz, seq, device=device, dtype=torch.bool)
    else:
        mask = mask.to(device=device, dtype=torch.bool)

    for b in range(bsz):
        valid = mask[b]
        if not torch.any(valid):
            continue
        rows = ids[b, valid, 1].to(torch.long)
        cols = ids[b, valid, 2].to(torch.long)
        grid_rows = max(int(rows.max().item()) + 1, 1)
        grid_cols = max(int(cols.max().item()) + 1, 1)

        row_bucket = (rows * panel_rows) // grid_rows
        col_bucket = (cols * panel_cols) // grid_cols
        row_bucket.clamp_(0, panel_rows - 1)
        col_bucket.clamp_(0, panel_cols - 1)
        panel_ids[b, valid] = (row_bucket * panel_cols + col_bucket).long()

    if squeeze_batch:
        panel_ids = panel_ids.squeeze(0)
    return remap_panel_ids(panel_ids, panel_aliases)


def remap_panel_ids(
    panel_ids: torch.Tensor,
    panel_aliases: Optional[Sequence[int]],
) -> torch.Tensor:
    """Remap panel IDs through an alias table (e.g. for multi-reference role-sharing)."""
    if not panel_aliases:
        return panel_ids
    alias_tensor = torch.as_tensor(panel_aliases, device=panel_ids.device, dtype=torch.long)
    result = panel_ids.clone()
    valid = result >= 0
    if valid.any():
        result[valid] = alias_tensor[result[valid]]
    return result
