"""
OPROLieLowRank: Low-Rank Lie Algebra Parameterization of Orthogonal Panel-Relative Operators.

This is the core module described in Section 3.3 of the paper:

    OPRO: Orthogonal Panel-Relative Operators
    for Panel-Aware In-Context Image Generation  (CVPR 2026)

For each panel p, we learn low-rank factors L_p, R_p in R^{d_h x rho} and construct:
    A_p = L_p R_p^T - R_p L_p^T        (skew-symmetric generator, Eq. 5)
    U_p = exp(A_p) in SO(d_h)           (orthogonal operator, Eq. 6)

Zero-interference initialization (Eq. 7):
    L_p ~ N(0, sigma^2),  R_p = 0  =>  A_p = 0  =>  U_p = I

Supports three variants:
    - "opro"  (default): Same U applied to Q and K. Satisfies Isometry + Same-Panel Invariance.
    - "abl1"  (APB):     Additive panel bias. Breaks both properties (Table 2 ablation).
    - "abl2"  (Asym):    Independent U_q, U_k. Preserves Isometry, breaks Same-Panel Invariance.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


OPROVariant = Literal["opro", "abl1", "abl2"]


def _deg_to_rad(value_deg: float) -> float:
    return float(value_deg) * math.pi / 180.0


class OPROLieLowRank(nn.Module):
    """Low-rank Lie algebra parameterization with matrix exponential.

    Args:
        num_panels:  Number of distinct panels (P).
        head_dim:    Attention head dimension (d_h).
        rank:        Low-rank dimension (rho). Defaults to 8.
        variant:     "opro" | "abl1" (additive bias) | "abl2" (asymmetric Q/K).
        sigma_deg:   Initialization scale in degrees for the Lie factors.
        symmetric_init: If True, initialize both L and R with random normals.
                        If False (default), L ~ N(0, sigma^2), R = 0 (zero-interference).
    """

    def __init__(
        self,
        num_panels: int,
        head_dim: int,
        rank: int = 8,
        *,
        variant: str = "opro",
        sigma_deg: float = 0.0,
        symmetric_init: bool = False,
    ) -> None:
        super().__init__()
        if variant not in ("opro", "abl1", "abl2"):
            raise ValueError(f"Unknown variant '{variant}' (expected 'opro', 'abl1', or 'abl2')")

        self.num_panels = int(num_panels)
        self.head_dim = int(head_dim)
        self.rank = int(min(rank, head_dim))
        self.variant: OPROVariant = variant  # type: ignore[assignment]
        self.symmetric_init = bool(symmetric_init)
        self.max_skew_fro_norm: float = 5.0

        # --- Main OPRO parameters (Eq. 5): L_p, R_p ---
        self.L = nn.Parameter(torch.zeros(num_panels, head_dim, self.rank))
        self.R = nn.Parameter(torch.zeros(num_panels, head_dim, self.rank))

        # --- Ablation: abl2 (Asymmetric) uses independent L_k, R_k ---
        if self.variant == "abl2":
            scale = 1.0 / max(1, self.rank)
            self.L_key = nn.Parameter(torch.randn(num_panels, head_dim, self.rank) * scale)
            self.R_key = nn.Parameter(torch.randn(num_panels, head_dim, self.rank) * scale)
        else:
            self.register_parameter("L_key", None)
            self.register_parameter("R_key", None)

        # --- Ablation: abl1 (Additive Panel Bias) ---
        if self.variant == "abl1":
            self.panel_bias = nn.Parameter(torch.zeros(num_panels, head_dim))
        else:
            self.register_parameter("panel_bias", None)

        self._reset_parameters(sigma_deg)

    def _reset_parameters(self, sigma_deg: float) -> None:
        """Zero-interference initialization (Eq. 7)."""
        std = _deg_to_rad(sigma_deg)
        if std <= 0:
            std = 1e-3
        scale = std / max(1, self.rank)
        with torch.no_grad():
            if self.L.numel() > 0:
                self.L.normal_(mean=0.0, std=scale)
                if self.symmetric_init:
                    self.R.normal_(mean=0.0, std=scale)
                else:
                    self.R.zero_()
            if self.variant == "abl2" and self.L_key is not None and self.R_key is not None:
                if self.L_key.numel() > 0:
                    self.L_key.normal_(mean=0.0, std=scale)
                    if self.symmetric_init:
                        self.R_key.normal_(mean=0.0, std=scale)
                    else:
                        self.R_key.zero_()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_skew(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute skew-symmetric generator A_p = L R^T - R L^T (Eq. 5).

        Includes Frobenius norm capping for training stability.
        """
        lr_t = torch.matmul(l, r.transpose(-2, -1))
        rl_t = torch.matmul(r, l.transpose(-2, -1))
        a = lr_t - rl_t
        a = 0.5 * (a - a.transpose(-2, -1))  # enforce exact skew-symmetry

        # Frobenius norm capping for stability
        fro = torch.linalg.norm(a, dim=(1, 2), keepdim=True)
        cap = torch.as_tensor(self.max_skew_fro_norm, dtype=torch.float32, device=a.device)
        scale = torch.clamp(cap / (fro + 1e-6), max=1.0)
        return a * scale

    def _compute_u(
        self,
        l: torch.Tensor,
        r: torch.Tensor,
        *,
        scale: float = 1.0,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Compute orthogonal operator U_p = exp(A_p) (Eq. 6).

        ``out_dtype`` lets callers request the runtime dtype of the
        downstream Q/K (e.g. bfloat16 for FluxFill).
        """
        scale_tensor = torch.as_tensor(scale, dtype=l.dtype, device=l.device)
        skew = self._compute_skew(l, r) * scale_tensor
        device_type = "cuda" if skew.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            u = torch.matrix_exp(skew.to(torch.float32))
        return u.to(out_dtype if out_dtype is not None else skew.dtype)

    def _select_params(self, target: Literal["q", "k"]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select L, R factors for query or key."""
        if target == "q":
            return self.L, self.R
        if self.variant != "abl2":
            raise RuntimeError("Requested key parameters but variant is not 'abl2'")
        assert self.L_key is not None and self.R_key is not None
        return self.L_key, self.R_key

    def skew_matrices(self) -> torch.Tensor:
        """Return the skew-symmetric generators A_p for the query branch."""
        l, r = self._select_params("q")
        return self._compute_skew(l, r)

    # ------------------------------------------------------------------
    # Forward: apply_to_qk
    # ------------------------------------------------------------------

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        panel_labels: torch.Tensor,
        *,
        gate: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply panel-specific orthogonal rotations to queries and keys.

        Args:
            q:  Query tensor  [batch, heads, seq_len, head_dim].
            k:  Key tensor    [batch, heads, seq_len, head_dim].
            panel_labels:  Panel indices [batch, seq_len], values in [0, P).
            gate:  Scalar multiplier on skew-symmetric generators (for warmup).

        Returns:
            (q_rot, k_rot) with the same shape as inputs.
        """
        if q.shape != k.shape:
            raise ValueError("q and k must have the same shape")
        if panel_labels.shape != (q.shape[0], q.shape[2]):
            raise ValueError("panel_labels must be [batch, seq_len]")
        if panel_labels.dtype != torch.long:
            panel_labels = panel_labels.long()

        # --- Ablation: abl1 (Additive Panel Bias) ---
        if self.variant == "abl1":
            bias = self.panel_bias[panel_labels].unsqueeze(1)  # [B, 1, S, d_h]
            return q + bias, k + bias

        # --- Main OPRO / abl2 ---
        l_q, r_q = self._select_params("q")
        u_q = self._compute_u(l_q, r_q, scale=gate, out_dtype=q.dtype)  # [P, d_h, d_h]
        u_q_tok = u_q[panel_labels]                                       # [B, S, d_h, d_h]
        q_rot = torch.einsum("bsde,bhse->bhsd", u_q_tok, q)

        if self.variant == "abl2":
            l_k, r_k = self._select_params("k")
            u_k = self._compute_u(l_k, r_k, scale=gate, out_dtype=k.dtype)
        else:
            u_k = u_q if k.dtype == q.dtype else u_q.to(k.dtype)
        u_k_tok = u_k[panel_labels]
        k_rot = torch.einsum("bsde,bhse->bhsd", u_k_tok, k)

        return q_rot, k_rot

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
        """Return the orthogonal matrix U_p for a single panel (detached)."""
        l, r = self._select_params(target)
        u = self._compute_u(l, r, scale=gate)
        return u[panel_index]

    @torch.no_grad()
    def verify_orthogonality(self, atol: float = 1e-4) -> bool:
        """Check U_p^T U_p ≈ I for all panels."""
        l, r = self._select_params("q")
        u = self._compute_u(l, r)
        I = torch.eye(self.head_dim, device=u.device, dtype=u.dtype).unsqueeze(0)
        err = (u.transpose(1, 2) @ u - I).abs().max()
        return err.item() < atol

    def extra_repr(self) -> str:
        return (
            f"num_panels={self.num_panels}, head_dim={self.head_dim}, "
            f"rank={self.rank}, variant='{self.variant}'"
        )


# ======================================================================
# OPRO-BD: RoPE-aligned block-diagonal specialization (Supp. Sec B)
# ======================================================================


class OPROBlockDiagonal(nn.Module):
    """Block-diagonal SO(2) specialization of OPRO (Supp. Sec B, Table 2).

    Per panel p, learn ``d_h/2`` phase angles ``phi_{p,k}`` and apply the
    panel rotation as a block-diagonal stack of 2x2 SO(2) rotations.
    Strictly weaker than ``OPROLieLowRank`` (cannot mix channel pairs)
    but uses only ``P * d_h/2`` parameters and yields a closed-form
    panel-relative phase shift on top of RoPE-style backbones.

    All structural guarantees are preserved by construction:

        * Isometry: each 2x2 block is a rotation, so norms are exact.
        * Same-Panel Invariance: identical panels yield identical phases,
          collapsing the relative rotation to identity.
        * Zero-init identity: all phases start at zero.
    """

    def __init__(
        self,
        num_panels: int,
        head_dim: int,
        *,
        sigma_deg: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for block-diagonal OPRO, got {head_dim}")
        self.num_panels = int(num_panels)
        self.head_dim = int(head_dim)
        self.num_blocks = head_dim // 2

        # Zero-init: phi=0 -> rotation = identity at step 0.
        self.phi = nn.Parameter(torch.zeros(num_panels, self.num_blocks))
        if sigma_deg > 0:
            with torch.no_grad():
                self.phi.normal_(mean=0.0, std=_deg_to_rad(sigma_deg))

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        panel_labels: torch.Tensor,
        *,
        gate: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape:
            raise ValueError("q and k must have the same shape")
        if panel_labels.shape != (q.shape[0], q.shape[2]):
            raise ValueError("panel_labels must be [batch, seq_len]")
        if panel_labels.dtype != torch.long:
            panel_labels = panel_labels.long()

        # [B, S, num_blocks]
        phi_tok = self.phi[panel_labels] * gate
        cos = phi_tok.cos().unsqueeze(1).unsqueeze(-1).to(q.dtype)  # [B, 1, S, B, 1]
        sin = phi_tok.sin().unsqueeze(1).unsqueeze(-1).to(q.dtype)

        B, H, S, d = q.shape
        q_pair = q.reshape(B, H, S, self.num_blocks, 2)
        k_pair = k.reshape(B, H, S, self.num_blocks, 2)

        def _rotate(x):
            x0, x1 = x[..., :1], x[..., 1:2]
            return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)

        return _rotate(q_pair).reshape(B, H, S, d), _rotate(k_pair).reshape(B, H, S, d)

    @torch.no_grad()
    def verify_orthogonality(self, atol: float = 1e-5) -> bool:
        """Block rotations are orthogonal by construction."""
        cos, sin = self.phi.cos(), self.phi.sin()
        det = cos * cos + sin * sin
        return (det - 1).abs().max().item() < atol

    def extra_repr(self) -> str:
        return f"num_panels={self.num_panels}, head_dim={self.head_dim} (BD)"
