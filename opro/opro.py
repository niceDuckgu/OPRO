"""Orthogonal Panel-Relative Operators (OPRO).

Implements the low-rank Lie-exponential parameterization of Sec. 3.3 of

    OPRO: Orthogonal Panel-Relative Operators
    for Panel-Aware In-Context Image Generation  (CVPR 2026)

For every panel p we learn two factors L_p, R_p in R^{d_h x rho} and build

    A_p = L_p R_p^T - R_p L_p^T        skew-symmetric generator   (Eq. 5)
    U_p = exp(A_p)  in SO(d_h)         orthogonal panel operator  (Eq. 6)

The operator of a token's panel is applied to its position-aware query and
key (Eq. 3), so the attention logit between tokens i and j becomes

    q_i^T U_{p(i)}^T U_{p(j)} k_j / sqrt(d_h)                     (Eq. 4)

which leaves same-panel logits untouched and preserves all Q/K norms.

Zero-interference initialization (Eq. 7): L_p = 0, R_p ~ N(0, sigma^2), hence
A_p = 0 and U_p = I at step 0 while dLoss/dL_p is non-zero.

Two ablation variants from Table 2 are included:
    "abl1"  additive panel bias (APB): neither isometric nor same-panel invariant
    "abl2"  independent operators for Q and K: isometric, not same-panel invariant
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


OPROVariant = Literal["opro", "abl1", "abl2"]
_VARIANTS = ("opro", "abl1", "abl2")


class OPROLieLowRank(nn.Module):
    """Panel-specific orthogonal operators with a low-rank skew generator.

    Args:
        num_panels:  Number of panels P.
        head_dim:    Attention head dimension d_h.
        rank:        Rank rho of the factors (clamped to ``head_dim``).
        variant:     "opro" | "abl1" (additive bias) | "abl2" (asymmetric Q/K).
        init_std:    Std of the non-zero factor at initialization.
        symmetric_init:  Initialize both factors randomly ("w/o Zero Init" in
                     Table 2). The default keeps U_p = I at step 0.
        max_skew_fro_norm:  Generators with a larger Frobenius norm are rescaled
                     to this value, which bounds the rotation angle.
    """

    def __init__(
        self,
        num_panels: int,
        head_dim: int,
        rank: int = 8,
        *,
        variant: str = "opro",
        init_std: float = 1e-3,
        symmetric_init: bool = False,
        max_skew_fro_norm: float = 5.0,
    ) -> None:
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown variant '{variant}' (expected one of {_VARIANTS})")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        self.num_panels = int(num_panels)
        self.head_dim = int(head_dim)
        self.rank = int(min(rank, head_dim))
        self.variant: OPROVariant = variant  # type: ignore[assignment]
        self.init_std = float(init_std)
        self.symmetric_init = bool(symmetric_init)
        self.max_skew_fro_norm = float(max_skew_fro_norm)

        shape = (self.num_panels, self.head_dim, self.rank)
        self.L = nn.Parameter(torch.zeros(shape))
        self.R = nn.Parameter(torch.zeros(shape))

        if self.variant == "abl2":
            self.L_key = nn.Parameter(torch.zeros(shape))
            self.R_key = nn.Parameter(torch.zeros(shape))
        else:
            self.register_parameter("L_key", None)
            self.register_parameter("R_key", None)

        if self.variant == "abl1":
            self.panel_bias = nn.Parameter(torch.zeros(self.num_panels, self.head_dim))
        else:
            self.register_parameter("panel_bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.init_std / max(1, self.rank)
        pairs = [(self.L, self.R)]
        if self.L_key is not None:
            pairs.append((self.L_key, self.R_key))
        with torch.no_grad():
            for left, right in pairs:
                right.normal_(mean=0.0, std=std)
                if self.symmetric_init:
                    left.normal_(mean=0.0, std=std)
                else:
                    left.zero_()
            if self.panel_bias is not None:
                self.panel_bias.zero_()

    # ------------------------------------------------------------------
    # Generator and operator
    # ------------------------------------------------------------------

    def _skew(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """A_p = L_p R_p^T - R_p L_p^T, norm-capped. Shape [P, d_h, d_h]."""
        a = left @ right.transpose(-2, -1)
        a = a - a.transpose(-2, -1)
        fro = torch.linalg.norm(a, dim=(-2, -1), keepdim=True)
        return a * torch.clamp(self.max_skew_fro_norm / (fro + 1e-6), max=1.0)

    def _factors(self, target: Literal["q", "k"]) -> Tuple[torch.Tensor, torch.Tensor]:
        if target == "k" and self.variant == "abl2":
            return self.L_key, self.R_key
        return self.L, self.R

    def skew_matrices(self, target: Literal["q", "k"] = "q") -> torch.Tensor:
        """Skew-symmetric generators A_p, shape [P, d_h, d_h]."""
        return self._skew(*self._factors(target))

    def panel_operators(self, target: Literal["q", "k"] = "q", *, gate: float = 1.0) -> torch.Tensor:
        """Orthogonal operators U_p = exp(gate * A_p), shape [P, d_h, d_h].

        The exponential is always evaluated in float32, outside autocast.
        """
        skew = self.skew_matrices(target).float() * gate
        with torch.autocast(device_type=skew.device.type, enabled=False):
            return torch.linalg.matrix_exp(skew)

    # ------------------------------------------------------------------
    # Application to queries and keys
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate(x: torch.Tensor, ops: torch.Tensor, panel_labels: torch.Tensor) -> torch.Tensor:
        """Apply ``ops[panel_labels[b, s]]`` to ``x[b, :, s]``; labels < 0 are left as is."""
        out_dtype = x.dtype
        x = x.to(ops.dtype)
        out = x
        shared = bool((panel_labels == panel_labels[:1]).all())
        for p in range(ops.shape[0]):
            mask = panel_labels == p
            if not bool(mask.any()):
                continue
            if shared:
                # Same layout for every batch element: rotate only this panel's tokens.
                idx = mask[0].nonzero(as_tuple=True)[0]
                if out is x:
                    out = x.clone()
                out[:, :, idx] = x[:, :, idx] @ ops[p].transpose(0, 1)
            else:
                out = torch.where(mask[:, None, :, None], x @ ops[p].transpose(0, 1), out)
        return out.to(out_dtype)

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        panel_labels: torch.Tensor,
        *,
        gate: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the panel operators to position-aware queries and keys.

        Args:
            q, k:          [batch, heads, seq_len, head_dim], after the backbone's
                           positional encoding.
            panel_labels:  [batch, seq_len] (or [seq_len]) integer panel index per
                           token. Tokens labelled with a negative index (e.g. text
                           tokens) are passed through unchanged.
            gate:          Scalar multiplier on the generators (1.0 = full operator).

        Returns:
            ``(q_hat, k_hat)`` with the shapes and dtypes of the inputs.
        """
        if q.shape != k.shape:
            raise ValueError("q and k must have the same shape")
        if panel_labels.dim() == 1:
            panel_labels = panel_labels.unsqueeze(0)
        if panel_labels.shape[0] == 1 and q.shape[0] != 1:
            panel_labels = panel_labels.expand(q.shape[0], -1)
        if panel_labels.shape != (q.shape[0], q.shape[2]):
            raise ValueError("panel_labels must be [batch, seq_len]")
        panel_labels = panel_labels.to(device=q.device, dtype=torch.long)
        if int(panel_labels.max()) >= self.num_panels:
            raise ValueError("panel_labels contains an index >= num_panels")

        if self.variant == "abl1":
            valid = (panel_labels >= 0)[:, None, :, None]
            bias = self.panel_bias[panel_labels.clamp_min(0)].unsqueeze(1).to(q.dtype)
            bias = bias * valid.to(q.dtype)
            return q + bias, k + bias

        u_q = self.panel_operators("q", gate=gate)
        u_k = self.panel_operators("k", gate=gate) if self.variant == "abl2" else u_q
        return self._rotate(q, u_q, panel_labels), self._rotate(k, u_k, panel_labels)

    forward = apply_to_qk

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @torch.no_grad()
    def panel_matrix(
        self,
        panel_index: int,
        *,
        target: Literal["q", "k"] = "q",
        gate: float = 1.0,
    ) -> torch.Tensor:
        """U_p for a single panel (detached)."""
        return self.panel_operators(target, gate=gate)[panel_index]

    @torch.no_grad()
    def verify_orthogonality(self, atol: float = 1e-4) -> bool:
        """Check U_p^T U_p = I for all panels."""
        u = self.panel_operators("q")
        eye = torch.eye(self.head_dim, device=u.device, dtype=u.dtype)
        return bool(((u.transpose(1, 2) @ u - eye).abs().max() < atol).item())

    def extra_repr(self) -> str:
        return (
            f"num_panels={self.num_panels}, head_dim={self.head_dim}, "
            f"rank={self.rank}, variant='{self.variant}'"
        )
