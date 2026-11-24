"""Tests for opro/opro.py — OPROLieLowRank core module."""

import pytest
import torch

from opro import OPROLieLowRank


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture(params=[32, 64, 128])
def head_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def num_panels(request):
    return request.param


@pytest.fixture(params=[4, 8, 32])
def rank(request):
    return request.param


def _make_qk(batch=2, heads=4, seq=16, d_h=64):
    q = torch.randn(batch, heads, seq, d_h)
    k = torch.randn(batch, heads, seq, d_h)
    return q, k


# ---------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------

class TestInit:
    def test_default_creates_identity(self):
        """Zero-interference init: U_p = I at step 0 (Eq. 7)."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8)
        for p in range(2):
            U = opro.panel_matrix(p)
            I = torch.eye(64)
            assert torch.allclose(U, I, atol=1e-6), f"Panel {p} not identity at init"

    def test_L_zero_R_random(self):
        """Default init (Eq. 7): L = 0, R ~ N(0, sigma^2)."""
        opro = OPROLieLowRank(num_panels=3, head_dim=64, rank=8)
        assert opro.L.abs().max().item() == 0.0, "L should be zero"
        assert opro.R.abs().sum().item() > 0.0, "R should be non-zero"

    def test_symmetric_init(self):
        """symmetric_init=True: both L and R are non-zero."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, symmetric_init=True)
        assert opro.R.abs().sum().item() > 0.0, "R should be non-zero with symmetric init"
        assert opro.L.abs().sum().item() > 0.0

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            OPROLieLowRank(num_panels=2, head_dim=64, rank=8, variant="bad")

    def test_rank_clamped_to_head_dim(self):
        opro = OPROLieLowRank(num_panels=2, head_dim=32, rank=999)
        assert opro.rank == 32


# ---------------------------------------------------------------
# Orthogonality (Proposition 1 prerequisite)
# ---------------------------------------------------------------

class TestOrthogonality:
    def test_orthogonal_at_init(self, head_dim, num_panels):
        opro = OPROLieLowRank(num_panels=num_panels, head_dim=head_dim, rank=8)
        assert opro.verify_orthogonality(atol=1e-5)

    def test_orthogonal_after_gradient_step(self):
        """Orthogonality must hold after parameter updates."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8)
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))

        # Simulate a gradient step
        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()
        with torch.no_grad():
            for p in opro.parameters():
                if p.grad is not None:
                    p.add_(p.grad * 0.1)

        assert opro.verify_orthogonality(atol=1e-4)

    def test_skew_symmetric(self):
        """A_p must be skew-symmetric: A^T = -A."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, symmetric_init=True)
        A = opro.skew_matrices()
        err = (A + A.transpose(-2, -1)).abs().max().item()
        assert err < 1e-6, f"Skew-symmetry violation: {err}"


# ---------------------------------------------------------------
# Isometry (Proposition 1)
# ---------------------------------------------------------------

class TestIsometry:
    def test_norm_preservation(self, head_dim):
        """||q_hat|| = ||q_tilde|| for all tokens (Proposition 1)."""
        opro = OPROLieLowRank(num_panels=3, head_dim=head_dim, rank=8, symmetric_init=True)
        q, k = _make_qk(d_h=head_dim, seq=32)
        panel_ids = torch.randint(0, 3, (2, 32))

        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)

        q_norm_err = (q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item()
        k_norm_err = (k.norm(dim=-1) - k_rot.norm(dim=-1)).abs().max().item()
        assert q_norm_err < 1e-4, f"Q norm not preserved: {q_norm_err}"
        assert k_norm_err < 1e-4, f"K norm not preserved: {k_norm_err}"


# ---------------------------------------------------------------
# Same-Panel Invariance (Proposition 2)
# ---------------------------------------------------------------

class TestSamePanelInvariance:
    def test_same_panel_attention_unchanged(self):
        """Attention scores between same-panel tokens must be identical."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, symmetric_init=True)
        q, k = _make_qk(batch=1, heads=4, seq=20, d_h=64)
        panel_ids = torch.zeros(1, 20, dtype=torch.long)
        panel_ids[:, 10:] = 1

        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)

        orig_attn = torch.einsum("bhid,bhjd->bhij", q, k)
        rot_attn = torch.einsum("bhid,bhjd->bhij", q_rot, k_rot)

        same_mask = (panel_ids[0].unsqueeze(0) == panel_ids[0].unsqueeze(1))
        diff = (orig_attn[0] - rot_attn[0])[:, same_mask].abs()
        assert diff.max().item() < 1e-4, f"Same-panel invariance broken: {diff.max().item()}"

    def test_cross_panel_attention_changes(self):
        """Inter-panel attention should differ after training."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, symmetric_init=True)
        q, k = _make_qk(batch=1, heads=4, seq=20, d_h=64)
        panel_ids = torch.zeros(1, 20, dtype=torch.long)
        panel_ids[:, 10:] = 1

        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)

        orig_attn = torch.einsum("bhid,bhjd->bhij", q, k)
        rot_attn = torch.einsum("bhid,bhjd->bhij", q_rot, k_rot)

        cross_mask = (panel_ids[0].unsqueeze(0) != panel_ids[0].unsqueeze(1))
        diff = (orig_attn[0] - rot_attn[0])[:, cross_mask].abs()
        # With symmetric init, cross-panel should change (non-zero rotation)
        assert diff.max().item() > 1e-6, "Cross-panel attention should differ"


# ---------------------------------------------------------------
# Variant: abl1 (Additive Panel Bias)
# ---------------------------------------------------------------

class TestAbl1:
    def test_abl1_output_shape(self):
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, variant="abl1")
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))
        q_out, k_out = opro.apply_to_qk(q, k, panel_ids)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_abl1_changes_norms(self):
        """APB breaks isometry (adds bias to Q/K)."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, variant="abl1")
        # Manually set non-zero bias
        with torch.no_grad():
            opro.panel_bias.fill_(0.5)
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))
        q_out, _ = opro.apply_to_qk(q, k, panel_ids)
        norm_diff = (q.norm(dim=-1) - q_out.norm(dim=-1)).abs().max().item()
        assert norm_diff > 0.01, "APB should break isometry"


# ---------------------------------------------------------------
# Variant: abl2 (Asymmetric Q/K)
# ---------------------------------------------------------------

class TestAbl2:
    def test_abl2_output_shape(self):
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, variant="abl2")
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))
        q_out, k_out = opro.apply_to_qk(q, k, panel_ids)
        assert q_out.shape == q.shape

    def test_abl2_has_separate_key_params(self):
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, variant="abl2")
        assert opro.L_key is not None
        assert opro.R_key is not None

    def test_abl2_preserves_norms(self):
        """Asymmetric variant still uses orthogonal matrices → isometry holds."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, variant="abl2")
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))
        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)
        norm_err = (q.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item()
        assert norm_err < 1e-4


# ---------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------

class TestGradient:
    def test_gradients_flow(self):
        """All parameters receive gradients."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8, symmetric_init=True)
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))
        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)
        loss = q_rot.sum() + k_rot.sum()
        loss.backward()
        assert opro.L.grad is not None
        assert opro.R.grad is not None
        assert opro.L.grad.abs().sum().item() > 0

    def test_gradients_flow_at_zero_init(self):
        """With zero-interference init (L = 0), L still receives a non-zero gradient."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8)
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))
        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)
        loss = (q_rot * torch.randn_like(q_rot)).sum() + (k_rot * torch.randn_like(k_rot)).sum()
        loss.backward()
        assert opro.L.grad.abs().sum().item() > 0
        assert opro.R.grad.abs().sum().item() == 0

    def test_gate_scales_rotation(self):
        """gate=0 should produce identity, gate=1 full rotation."""
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8,
                              init_std=1.0, symmetric_init=True)
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 16))

        q0, k0 = opro.apply_to_qk(q, k, panel_ids, gate=0.0)
        assert torch.allclose(q, q0, atol=1e-5), "gate=0 should give identity"

        q1, _ = opro.apply_to_qk(q, k, panel_ids, gate=1.0)
        # With large sigma and symmetric init, gate=1 should differ
        assert not torch.allclose(q, q1, atol=1e-3)


# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

class TestEdgeCases:
    def test_single_panel(self):
        """With 1 panel, output should equal input (no inter-panel to modulate)."""
        opro = OPROLieLowRank(num_panels=1, head_dim=64, rank=8)
        q, k = _make_qk(d_h=64)
        panel_ids = torch.zeros(2, 16, dtype=torch.long)
        q_rot, k_rot = opro.apply_to_qk(q, k, panel_ids)
        assert torch.allclose(q, q_rot, atol=1e-5)

    def test_mismatched_shapes_raises(self):
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8)
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 20, 64)  # different seq_len
        panel_ids = torch.randint(0, 2, (2, 16))
        with pytest.raises(ValueError, match="same shape"):
            opro.apply_to_qk(q, k, panel_ids)

    def test_wrong_panel_labels_shape_raises(self):
        opro = OPROLieLowRank(num_panels=2, head_dim=64, rank=8)
        q, k = _make_qk(d_h=64)
        panel_ids = torch.randint(0, 2, (2, 999))  # wrong seq dim
        with pytest.raises(ValueError, match="panel_labels"):
            opro.apply_to_qk(q, k, panel_ids)
