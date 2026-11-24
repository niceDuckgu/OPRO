"""Tests for opro/utils.py — OPROConfig, compute_panel_ids, remap_panel_ids."""

import json
import tempfile
import os

import pytest
import torch

from opro.utils import OPROConfig, compute_panel_ids, remap_panel_ids


# ---------------------------------------------------------------
# OPROConfig
# ---------------------------------------------------------------

class TestOPROConfig:
    def test_defaults(self):
        cfg = OPROConfig()
        assert cfg.num_panels == 2
        assert cfg.panel_rows == 1
        assert cfg.panel_cols == 2
        assert cfg.rank == 32
        assert cfg.variant == "opro"

    def test_from_dict_upper_keys(self):
        raw = {"NUM_PANELS": 3, "PANEL_ROWS": 1, "PANEL_COLS": 3, "LIE_RANK": 16, "VARIANT": "abl1"}
        cfg = OPROConfig.from_dict(raw)
        assert cfg.num_panels == 3
        assert cfg.rank == 16
        assert cfg.variant == "abl1"

    def test_from_dict_lower_keys(self):
        raw = {"num_panels": 4, "panel_rows": 2, "panel_cols": 2, "rank": 8}
        cfg = OPROConfig.from_dict(raw)
        assert cfg.num_panels == 4
        assert cfg.rank == 8

    def test_from_dict_aliases(self):
        raw = {"NUM_PANELS": 2, "PANEL_ALIASES": [0, 0]}
        cfg = OPROConfig.from_dict(raw)
        assert cfg.panel_aliases == (0, 0)

    def test_save_load_roundtrip(self):
        cfg = OPROConfig(num_panels=3, rank=16, variant="abl2", panel_aliases=(0, 1, 0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cfg.save(path)
            loaded = OPROConfig.load(path)
            assert loaded.num_panels == 3
            assert loaded.rank == 16
            assert loaded.variant == "abl2"
            assert loaded.panel_aliases == [0, 1, 0]  # json converts tuple to list


# ---------------------------------------------------------------
# compute_panel_ids
# ---------------------------------------------------------------

def _make_grid_ids(rows: int, cols: int) -> torch.Tensor:
    """Create Flux-style img_ids [seq_len, 3] for a rows x cols grid."""
    ids = torch.zeros(rows * cols, 3)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            ids[idx, 1] = r
            ids[idx, 2] = c
    return ids


class TestComputePanelIds:
    def test_diptych_1x2(self):
        """10x20 grid → 1x2 panels: left half = 0, right half = 1."""
        ids = _make_grid_ids(10, 20)
        panel_ids = compute_panel_ids(ids, panel_rows=1, panel_cols=2)
        assert panel_ids.shape == (10 * 20,)
        # Left half (cols 0-9) → panel 0, right half (cols 10-19) → panel 1
        for idx in range(200):
            r, c = idx // 20, idx % 20
            expected = 0 if c < 10 else 1
            assert panel_ids[idx].item() == expected, f"row={r}, col={c}"

    def test_triptych_1x3(self):
        """9x9 grid → 1x3 panels."""
        ids = _make_grid_ids(9, 9)
        panel_ids = compute_panel_ids(ids, panel_rows=1, panel_cols=3)
        assert set(panel_ids.tolist()) == {0, 1, 2}

    def test_2x2_grid(self):
        """10x10 grid → 2x2 panels."""
        ids = _make_grid_ids(10, 10)
        panel_ids = compute_panel_ids(ids, panel_rows=2, panel_cols=2)
        assert set(panel_ids.tolist()) == {0, 1, 2, 3}
        # Top-left corner → panel 0
        assert panel_ids[0].item() == 0
        # Bottom-right corner → panel 3
        assert panel_ids[-1].item() == 3

    def test_batched_input(self):
        """Batched ids [B, seq, 3]."""
        ids = _make_grid_ids(8, 8).unsqueeze(0).expand(3, -1, -1).contiguous()
        panel_ids = compute_panel_ids(ids, panel_rows=2, panel_cols=2)
        assert panel_ids.shape == (3, 64)
        # All batches should be identical
        assert torch.equal(panel_ids[0], panel_ids[1])

    def test_with_mask(self):
        """Masked tokens get panel_id = -1."""
        ids = _make_grid_ids(4, 4)
        mask = torch.ones(16, dtype=torch.bool)
        mask[0] = False
        mask[15] = False
        panel_ids = compute_panel_ids(ids, panel_rows=2, panel_cols=2, mask=mask)
        assert panel_ids[0].item() == -1
        assert panel_ids[15].item() == -1
        assert panel_ids[1].item() >= 0


# ---------------------------------------------------------------
# remap_panel_ids
# ---------------------------------------------------------------

class TestRemapPanelIds:
    def test_no_aliases_passthrough(self):
        ids = torch.tensor([0, 1, 2, 0, 1])
        result = remap_panel_ids(ids, None)
        assert torch.equal(ids, result)

    def test_identity_alias(self):
        ids = torch.tensor([0, 1, 2])
        result = remap_panel_ids(ids, [0, 1, 2])
        assert torch.equal(ids, result)

    def test_merge_panels(self):
        """Alias [0, 0, 1] merges panels 0 and 1 into role 0."""
        ids = torch.tensor([0, 1, 2, 0])
        result = remap_panel_ids(ids, [0, 0, 1])
        assert result.tolist() == [0, 0, 1, 0]

    def test_preserves_negative(self):
        """Tokens with id=-1 (masked) stay -1."""
        ids = torch.tensor([-1, 0, 1, -1])
        result = remap_panel_ids(ids, [1, 0])
        assert result.tolist() == [-1, 1, 0, -1]


class TestGridPanelIds:
    def test_diptych_matches_coordinate_binning(self):
        from opro import grid_panel_ids
        rows, cols = torch.meshgrid(torch.arange(4), torch.arange(8), indexing="ij")
        ids = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1).reshape(-1, 3)
        expected = compute_panel_ids(ids, panel_rows=1, panel_cols=2)
        assert torch.equal(grid_panel_ids(4, 8, 1, 2), expected)

    def test_2x2(self):
        from opro import grid_panel_ids
        assert grid_panel_ids(2, 2, 2, 2).tolist() == [0, 1, 2, 3]
