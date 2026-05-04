"""
ViT with OPRO + 4 Positional Encodings for Compositional Reasoning (Section 4.1).

Backbones supported (controlled benchmark, Table 1):
  - APE      : learnable absolute positional embedding added to input.
  - RoPE     : 2D Rotary Positional Embedding (axis-wise).
  - LieRE    : Lie-group rotational PE (per-axis skew generators, block-diagonal).
  - ComRoPE  : commuting-axis variant of LieRE-style rotation (ComRoPE-LD form).

Adapters supported (controlled benchmark, Tables 1 & 2):
  - LoRA on QKV projection.
  - OPRO (full SO(d_h) low-rank Lie parameterization)              -> opro.OPROLieLowRank.
  - OPRO-BD (block-diagonal RoPE-aligned phase form, supp Sec B)   -> opro.OPROBlockDiagonal.
  - APB (Additive Panel Bias, Tab 2 ablation)                      -> variant="abl1".
  - Asym-OPRO (independent U_q, U_k, Tab 2 ablation)               -> variant="abl2".
  - OPRO w/o Zero Init                                              -> symmetric_init=True.

The 4 PE classes share a uniform `apply(q_img, k_img, H, W) -> (q', k')` interface
(applied to image tokens only; CLS is concatenated back outside).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opro import OPROLieLowRank
from opro.opro import OPROBlockDiagonal


VIT_CONFIGS = {
    "tiny":  {"depth": 6,  "heads": 3,  "dim": 192},
    "small": {"depth": 12, "heads": 6,  "dim": 384},
    "base":  {"depth": 12, "heads": 12, "dim": 768},
    "large": {"depth": 24, "heads": 16, "dim": 1024},
}

PE_TYPES = ("ape", "rope", "liere", "comrope")


# ---------------------------------------------------------------
# Patch embed + LoRA
# ---------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear (frozen base + low-rank delta)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


# ---------------------------------------------------------------
# 2D RoPE (no learnable params)
# ---------------------------------------------------------------

def apply_rope_2d(x: torch.Tensor, H: int, W: int, base: float = 10000.0) -> torch.Tensor:
    """Apply axis-wise 2D RoPE: first half dims encode x-axis, second half y-axis."""
    B, heads, S, d = x.shape
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, 2, device=x.device).float() / half))

    pos_y = torch.arange(H, device=x.device).float().unsqueeze(1).expand(H, W).reshape(-1)
    pos_x = torch.arange(W, device=x.device).float().unsqueeze(0).expand(H, W).reshape(-1)

    angles = torch.cat([torch.outer(pos_x, inv_freq), torch.outer(pos_y, inv_freq)], dim=-1)
    cos_a = angles.cos().unsqueeze(0).unsqueeze(0)
    sin_a = angles.sin().unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], dim=-1)


# ---------------------------------------------------------------
# LieRE — Lie-group block-diagonal rotational PE
# ---------------------------------------------------------------

class LieREPositionalEncoder(nn.Module):
    """LieRE [Ostmeier et al.]: per-layer learnable skew generators applied via matrix exp.

    For each token at normalized position (x, y) we form a block-diagonal generator
    ``G_pos = x * G_x + y * G_y``, exponentiate it to ``U = exp(G_pos)``, and rotate Q, K
    block-wise. Block size B trades fidelity against compute (default 4).

    This implementation matches LieRE's per-axis, per-block parameterization while
    keeping the controlled-benchmark code self-contained (no external deps).
    """

    def __init__(self, head_dim: int, num_axes: int = 2, block_size: int = 4,
                 init_scale: float = 2 * math.pi):
        super().__init__()
        if head_dim % block_size != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by block_size ({block_size})")
        self.head_dim = head_dim
        self.num_axes = num_axes
        self.block_size = block_size
        self.num_blocks = head_dim // block_size
        # Learnable raw params; skew-symmetrized in forward (ensures generator in so(b))
        self.G_raw = nn.Parameter(
            torch.randn(num_axes, self.num_blocks, block_size, block_size) * (init_scale / block_size)
        )

    @staticmethod
    def _normalized_positions(H: int, W: int, device, dtype) -> torch.Tensor:
        ys = torch.arange(H, device=device, dtype=dtype) / max(H - 1, 1)
        xs = torch.arange(W, device=device, dtype=dtype) / max(W - 1, 1)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [S, 2]

    def _rotation(self, S: int, H: int, W: int, device, dtype) -> torch.Tensor:
        positions = self._normalized_positions(H, W, device, torch.float32)  # [S, 2]
        G_skew = (self.G_raw - self.G_raw.transpose(-1, -2)).to(torch.float32)
        # [S, num_axes] @ [num_axes, num_blocks*b*b] -> [S, num_blocks, b, b]
        G_pos = positions @ G_skew.flatten(1)
        G_pos = G_pos.reshape(S, self.num_blocks, self.block_size, self.block_size)
        U = torch.matrix_exp(G_pos)  # [S, num_blocks, b, b]
        return U.to(dtype)

    def apply(self, q: torch.Tensor, k: torch.Tensor, H: int, W: int):
        B, heads, S, d = q.shape
        assert S == H * W, f"LieRE expects S=H*W ({H*W}), got {S}"
        U = self._rotation(S, H, W, q.device, q.dtype)  # [S, num_blocks, b, b]
        q_blk = q.reshape(B, heads, S, self.num_blocks, self.block_size)
        k_blk = k.reshape(B, heads, S, self.num_blocks, self.block_size)
        q_rot = torch.einsum("snij,bhsnj->bhsni", U, q_blk).reshape(B, heads, S, d)
        k_rot = torch.einsum("snij,bhsnj->bhsni", U, k_blk).reshape(B, heads, S, d)
        return q_rot, k_rot


# ---------------------------------------------------------------
# ComRoPE — commuting-axis (ComRoPE-LD form)
# ---------------------------------------------------------------

class ComRoPEPositionalEncoder(nn.Module):
    """ComRoPE-LD [Yu et al.]: commuting axis generators via shared base.

    Per axis j, the generator is ``G_j = m_j * P`` for a learnable scalar ``m_j`` and
    a shared learnable skew base ``P``. Since all G_j are scalar multiples of the same
    skew matrix, ``[G_x, G_y] = 0`` (they commute by construction), guaranteeing
    additive composition: ``exp(x*G_x + y*G_y) = exp(x*G_x) @ exp(y*G_y)``.
    """

    def __init__(self, head_dim: int, num_axes: int = 2, block_size: int = 4,
                 init_std: float = 0.5):
        super().__init__()
        if head_dim % block_size != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by block_size ({block_size})")
        self.head_dim = head_dim
        self.num_axes = num_axes
        self.block_size = block_size
        self.num_blocks = head_dim // block_size
        # Shared base skew
        self.P_raw = nn.Parameter(
            torch.randn(self.num_blocks, block_size, block_size) * init_std
        )
        # Per-axis scalars (per block) -> commuting because all share the same base P
        self.multiplier = nn.Parameter(torch.randn(num_axes, self.num_blocks) * 0.5)

    def _rotation(self, S: int, H: int, W: int, device, dtype) -> torch.Tensor:
        positions = LieREPositionalEncoder._normalized_positions(H, W, device, torch.float32)
        P = (self.P_raw - self.P_raw.transpose(-1, -2)).to(torch.float32)  # [num_blocks, b, b]
        # G[axis, blk] = multiplier[axis, blk] * P[blk]   -> all commute (scalar multiples of P)
        G = self.multiplier.unsqueeze(-1).unsqueeze(-1) * P.unsqueeze(0)  # [num_axes, num_blocks, b, b]
        G_pos = positions @ G.flatten(1)
        G_pos = G_pos.reshape(S, self.num_blocks, self.block_size, self.block_size)
        U = torch.matrix_exp(G_pos)
        return U.to(dtype)

    def apply(self, q: torch.Tensor, k: torch.Tensor, H: int, W: int):
        B, heads, S, d = q.shape
        assert S == H * W, f"ComRoPE expects S=H*W ({H*W}), got {S}"
        U = self._rotation(S, H, W, q.device, q.dtype)
        q_blk = q.reshape(B, heads, S, self.num_blocks, self.block_size)
        k_blk = k.reshape(B, heads, S, self.num_blocks, self.block_size)
        q_rot = torch.einsum("snij,bhsnj->bhsni", U, q_blk).reshape(B, heads, S, d)
        k_rot = torch.einsum("snij,bhsnj->bhsni", U, k_blk).reshape(B, heads, S, d)
        return q_rot, k_rot


def make_pe(pos_type: str, head_dim: int) -> Optional[nn.Module]:
    """Construct a PE module for rotational types; APE/RoPE return None (handled inline)."""
    if pos_type in ("ape", "rope"):
        return None
    if pos_type == "liere":
        return LieREPositionalEncoder(head_dim)
    if pos_type == "comrope":
        return ComRoPEPositionalEncoder(head_dim)
    raise ValueError(f"Unknown pos_type '{pos_type}'. Expected one of {PE_TYPES}.")


# ---------------------------------------------------------------
# Attention with PE swap + OPRO injection
# ---------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, pos_type: str = "rope"):
        super().__init__()
        if pos_type not in PE_TYPES:
            raise ValueError(f"pos_type must be one of {PE_TYPES}, got '{pos_type}'")
        self.heads = heads
        self.d_h = dim // heads
        self.scale = self.d_h ** -0.5
        self.pos_type = pos_type
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.pe = make_pe(pos_type, self.d_h)
        # OPRO is attached lazily by ViTWithOPRO._inject_opro
        self.opro: Optional[nn.Module] = None

    def forward(
        self,
        x: torch.Tensor,
        H: int, W: int,
        panel_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.heads, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Positional encoding (image tokens only — preserve CLS at index 0)
        if S > H * W:
            q_cls, q_img = q[:, :, :1], q[:, :, 1:]
            k_cls, k_img = k[:, :, :1], k[:, :, 1:]
        else:
            q_cls = k_cls = None
            q_img, k_img = q, k

        if self.pos_type == "rope":
            q_img = apply_rope_2d(q_img, H, W)
            k_img = apply_rope_2d(k_img, H, W)
        elif self.pe is not None:
            q_img, k_img = self.pe.apply(q_img, k_img, H, W)
        # APE is added at the input embedding stage, nothing to do here.

        if q_cls is not None:
            q = torch.cat([q_cls, q_img], dim=2)
            k = torch.cat([k_cls, k_img], dim=2)
        else:
            q, k = q_img, k_img

        # OPRO: panel-relative orthogonal modulation
        if self.opro is not None and panel_labels is not None:
            q, k = self.opro.apply_to_qk(q, k, panel_labels)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, pos_type: str = "rope", mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, pos_type)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x, H, W, panel_labels=None):
        x = x + self.attn(self.norm1(x), H, W, panel_labels)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------
# Full ViT with adapter injection
# ---------------------------------------------------------------

OPRO_VARIANTS = ("opro", "opro_bd", "abl1", "abl2")


class ViTWithOPRO(nn.Module):
    """ViT-B variant for the controlled compositional-reasoning benchmark.

    Args:
        size:           'tiny' | 'small' | 'base' | 'large'.
        pos_type:       'ape' | 'rope' | 'liere' | 'comrope'.
        num_panels:     Number of panel operators (= grid_size**2 in Stage 2).
        opro_rank:      Low-rank dimension rho for OPRO (ignored by 'opro_bd').
        opro_variant:   'opro' | 'opro_bd' | 'abl1' (APB) | 'abl2' (Asym).
        opro_symmetric_init: True activates the "OPRO w/o Zero Init" ablation.
        use_opro / use_lora / lora_rank:  adapter toggles.
    """

    def __init__(
        self,
        size: str = "base",
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 8,
        pos_type: str = "rope",
        num_panels: int = 9,
        opro_rank: int = 8,
        opro_variant: str = "opro",
        opro_symmetric_init: bool = False,
        use_opro: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        if pos_type not in PE_TYPES:
            raise ValueError(f"pos_type must be one of {PE_TYPES}, got '{pos_type}'")
        if opro_variant not in OPRO_VARIANTS:
            raise ValueError(f"opro_variant must be one of {OPRO_VARIANTS}, got '{opro_variant}'")

        cfg = VIT_CONFIGS[size]
        dim, heads, depth = cfg["dim"], cfg["heads"], cfg["depth"]
        self.d_h = dim // heads
        self.pos_type = pos_type

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        self.H = self.W = img_size // patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        if pos_type == "ape":
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, dim) * 0.02
            )
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, pos_type) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        if use_opro:
            self._inject_opro(num_panels, opro_rank, opro_variant, opro_symmetric_init)
        if use_lora:
            self._inject_lora(lora_rank)

    def _inject_opro(self, num_panels: int, rank: int, variant: str,
                     symmetric_init: bool) -> None:
        for block in self.blocks:
            if variant == "opro_bd":
                block.attn.opro = OPROBlockDiagonal(
                    num_panels=num_panels, head_dim=self.d_h,
                )
            else:
                block.attn.opro = OPROLieLowRank(
                    num_panels=num_panels, head_dim=self.d_h,
                    rank=rank, variant=variant,
                    symmetric_init=symmetric_init,
                )

    def _inject_lora(self, rank: int) -> None:
        for block in self.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank)

    def freeze_backbone(self) -> None:
        """Stage 2 setup: freeze backbone, keep adapters + classification head trainable."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(True)
        for block in self.blocks:
            if block.attn.opro is not None:
                for p in block.attn.opro.parameters():
                    p.requires_grad_(True)
            if isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv.A.requires_grad_(True)
                block.attn.qkv.B.requires_grad_(True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        panel_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = pixel_values.size(0)
        x = self.patch_embed(pixel_values)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :x.size(1)]

        # Prepend CLS panel ID (always 0; OPRO same-panel invariance handles it).
        if panel_labels is not None:
            if panel_labels.dim() == 1:
                panel_labels = panel_labels.unsqueeze(0).expand(B, -1)
            cls_panel = torch.zeros(B, 1, dtype=panel_labels.dtype, device=panel_labels.device)
            panel_labels = torch.cat([cls_panel, panel_labels], dim=1)

        for block in self.blocks:
            x = block(x, self.H, self.W, panel_labels)

        x = self.norm(x)
        return self.head(x[:, 0])
