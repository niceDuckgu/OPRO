"""ViT classifier for the two-stage compositional reasoning benchmark (Sec. 4.1).

A standard pre-norm ViT whose attention layers can host

  - a positional encoding on Q/K (``rope`` | ``liere`` | ``comrope``); ``ape``
    uses only the learned absolute position embeddings,
  - LoRA on the query and key projections,
  - OPRO, applied to the position-aware Q/K right before the attention product.

As in our experiments, every backbone keeps the ViT's learned absolute position
embeddings at the input and the rotary variants add their Q/K rotation on top.
One positional-encoding module and one set of panel operators are shared by
all layers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from opro import OPROLieLowRank

from .positional import build_positional

VIT_CONFIGS = {
    "tiny": {"dim": 192, "heads": 3, "depth": 12},
    "small": {"dim": 384, "heads": 6, "depth": 12},
    "base": {"dim": 768, "heads": 12, "depth": 12},
    "large": {"dim": 1024, "heads": 16, "depth": 24},
}
POS_TYPES = ("ape", "rope", "liere", "comrope")


class LoRALinear(nn.Module):
    """``nn.Linear`` with a frozen base weight and a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_down = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.lora_down, std=1e-4)
        self.scaling = alpha / max(1, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_down.t()) @ self.lora_up.t()


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        xy: Optional[torch.Tensor],
        positional: Optional[nn.Module],
        operator: Optional[OPROLieLowRank],
        panel_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, S, D = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, self.heads, D // self.heads).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        if positional is not None:
            q, k = positional(q, k, xy)
        if operator is not None and panel_labels is not None:
            q, k = operator.apply_to_qk(q, k, panel_labels)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, S, D))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x, xy, positional, operator, panel_labels):
        x = x + self.attn(self.norm1(x), xy, positional, operator, panel_labels)
        return x + self.mlp(self.norm2(x))


class ViTClassifier(nn.Module):
    """ViT with a selectable positional encoding and optional LoRA / OPRO.

    Args:
        num_classes:  Output classes (8 for both benchmark stages).
        size:         "tiny" | "small" | "base" | "large".
        pos_type:     "ape" | "rope" | "liere" | "comrope".
    """

    def __init__(
        self,
        num_classes: int = 8,
        size: str = "base",
        pos_type: str = "rope",
        image_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        if pos_type not in POS_TYPES:
            raise ValueError(f"pos_type must be one of {POS_TYPES}")
        cfg = VIT_CONFIGS[size]
        dim, heads, depth = cfg["dim"], cfg["heads"], cfg["depth"]
        self.pos_type = pos_type
        self.head_dim = dim // heads
        self.grid = image_size // patch_size

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.positional = None if pos_type == "ape" else build_positional(pos_type, self.head_dim, heads)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.classifier = nn.Linear(dim, num_classes)

        self.operator: Optional[OPROLieLowRank] = None
        self.register_buffer("xy", self._patch_coordinates(self.grid), persistent=False)

    @staticmethod
    def _patch_coordinates(grid: int) -> torch.Tensor:
        """Patch centres in [-1, 1]^2, with the class token at the origin. Shape [1, 1+G*G, 2]."""
        centres = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid * 2 - 1
        y, x = torch.meshgrid(centres, centres, indexing="ij")
        xy = torch.stack([x.flatten(), y.flatten()], dim=-1)
        return torch.cat([torch.zeros(1, 2), xy], dim=0).unsqueeze(0)

    # ------------------------------------------------------------------
    # Stage-2 adapters
    # ------------------------------------------------------------------

    def add_lora(self, rank: int = 8, alpha: float = 16.0) -> None:
        """Wrap the query and key projections of every layer with LoRA."""
        for block in self.blocks:
            block.attn.query = LoRALinear(block.attn.query, rank, alpha)
            block.attn.key = LoRALinear(block.attn.key, rank, alpha)

    def add_opro(self, num_panels: int, rank: int = 8, variant: str = "opro",
                 symmetric_init: bool = False) -> None:
        """Attach one set of panel operators, shared by all attention layers."""
        self.operator = OPROLieLowRank(num_panels, self.head_dim, rank, variant=variant,
                                       symmetric_init=symmetric_init)

    def freeze_backbone(self) -> None:
        """Leave only the classifier, LoRA and OPRO parameters trainable."""
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(("classifier.", "operator.")) or "lora_" in name)

    def load_backbone(self, state_dict: dict) -> None:
        """Load Stage-1 weights, keeping the freshly initialized classifier."""
        state = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith(("classifier.", "operator.")) and "lora_" not in k]
        if missing or unexpected:
            raise RuntimeError(f"Backbone mismatch. Missing: {missing}. Unexpected: {unexpected}")

    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor, panel_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pixel_values:  [B, 3, H, W].
            panel_labels:  [B, num_patches] or [num_patches] panel index per patch;
                           required for OPRO, ignored otherwise.
        """
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed

        if self.operator is not None and panel_labels is not None:
            if panel_labels.dim() == 1:
                panel_labels = panel_labels.unsqueeze(0).expand(B, -1)
            # The class token shares the operator of panel 0.
            cls_label = torch.zeros(B, 1, dtype=panel_labels.dtype, device=panel_labels.device)
            panel_labels = torch.cat([cls_label, panel_labels], dim=1)
        else:
            panel_labels = None

        xy = self.xy.expand(B, -1, -1) if self.positional is not None else None
        for block in self.blocks:
            x = block(x, xy, self.positional, self.operator, panel_labels)
        return self.classifier(self.norm(x)[:, 0])
