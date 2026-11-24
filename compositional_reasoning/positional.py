"""Positional encodings applied to attention queries and keys.

All three encodings rotate consecutive channel pairs of Q and K by a
position-dependent angle (a block-diagonal orthogonal map), so attention
depends on relative positions. They differ in how the angles are produced:

    rope     fixed axial frequencies: the first half of the pairs encodes x,
             the second half encodes y
    liere    learnable per-pair coefficients for x and y (a block-diagonal
             instance of LieRE's learnable generators)
    comrope  learnable per-pair frequencies with learnable x / y multipliers,
             every pair mixes both axes (a block-diagonal, commuting instance
             of ComRoPE)

These are the lightweight variants used in our compositional reasoning
experiments, not full re-implementations of LieRE or ComRoPE.

Coordinates ``xy`` are in [-1, 1]; the class token sits at the origin and is
therefore left unrotated.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def rotate_pairs(t: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs (2i, 2i+1) of ``t`` [B, H, S, D] by ``angles`` [B, S, D/2]."""
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    even, odd = t[..., 0::2], t[..., 1::2]
    out = torch.empty_like(t)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out


class RoPE2D(nn.Module):
    """Axial 2D RoPE with fixed frequencies (no parameters)."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
        idx = torch.arange(head_dim // 4, dtype=torch.float32)
        self.register_buffer("inv_freq", base ** (-2 * idx / (head_dim // 2)), persistent=False)

    def angles(self, xy: torch.Tensor) -> torch.Tensor:
        ax = xy[..., 0:1] * self.inv_freq
        ay = xy[..., 1:2] * self.inv_freq
        return torch.cat([ax, ay], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = self.angles(xy)
        return rotate_pairs(q, angles), rotate_pairs(k, angles)


class LieRE2D(nn.Module):
    """Learnable per-pair x / y coefficients, initialized uniformly in [0, 2*pi)."""

    def __init__(self, head_dim: int, num_heads: int, init_scale: float = 2 * math.pi) -> None:
        super().__init__()
        self.generators = nn.Parameter(torch.rand(2, num_heads, head_dim // 2) * init_scale)

    def angles(self, xy: torch.Tensor) -> torch.Tensor:
        coeff = self.generators.mean(dim=1)  # shared across heads: [2, pairs]
        return xy[..., 0:1] * coeff[0] + xy[..., 1:2] * coeff[1]

    def forward(self, q: torch.Tensor, k: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = self.angles(xy)
        return rotate_pairs(q, angles), rotate_pairs(k, angles)


class ComRoPE2D(nn.Module):
    """Learnable per-pair frequencies with learnable axis multipliers."""

    def __init__(self, head_dim: int, num_heads: int, init_std: float = 0.02, base: float = 10000.0) -> None:
        super().__init__()
        pairs = head_dim // 2
        idx = torch.arange(pairs, dtype=torch.float32)
        self.register_buffer("base_inv_freq", base ** (-2 * idx / (2 * pairs)), persistent=False)
        self.freqs = nn.Parameter(torch.randn(num_heads, 1, pairs) * init_std)
        self.multiplier = nn.Parameter(torch.randn(1, 2, pairs))

    def angles(self, xy: torch.Tensor) -> torch.Tensor:
        inv_freq = self.base_inv_freq * torch.exp(self.freqs.squeeze(1))  # [heads, pairs]
        inv_freq = inv_freq.mean(dim=0)  # shared across heads
        mult_x, mult_y = self.multiplier[0, 0], self.multiplier[0, 1]
        return (xy[..., 0:1] * mult_x + xy[..., 1:2] * mult_y) * inv_freq

    def forward(self, q: torch.Tensor, k: torch.Tensor, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = self.angles(xy)
        return rotate_pairs(q, angles), rotate_pairs(k, angles)


def build_positional(pos_type: str, head_dim: int, num_heads: int) -> nn.Module:
    if pos_type == "rope":
        return RoPE2D(head_dim)
    if pos_type == "liere":
        return LieRE2D(head_dim, num_heads)
    if pos_type == "comrope":
        return ComRoPE2D(head_dim, num_heads)
    raise ValueError(f"Unknown pos_type '{pos_type}'")
