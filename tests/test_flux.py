"""Tests for opro/flux.py on a tiny, randomly initialized FLUX transformer."""

import pytest
import torch

from opro import add_opro_to_flux, compute_panel_ids, load_opro_for_flux

diffusers = pytest.importorskip("diffusers")

TXT_LEN, GRID_H, GRID_W = 5, 4, 8  # 1x2 diptych of 4x4-token panels


def _tiny_flux():
    torch.manual_seed(0)
    return diffusers.FluxTransformer2DModel(
        patch_size=1, in_channels=16, num_layers=2, num_single_layers=3,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
        pooled_projection_dim=32, guidance_embeds=False, axes_dims_rope=(4, 6, 6),
    ).eval()


def _inputs(batch=2):
    torch.manual_seed(1)
    rows, cols = torch.meshgrid(torch.arange(GRID_H), torch.arange(GRID_W), indexing="ij")
    img_ids = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1).reshape(-1, 3).float()
    kwargs = dict(
        hidden_states=torch.randn(batch, GRID_H * GRID_W, 16),
        encoder_hidden_states=torch.randn(batch, TXT_LEN, 32),
        pooled_projections=torch.randn(batch, 32),
        timestep=torch.full((batch,), 0.5),
        img_ids=img_ids,
        txt_ids=torch.zeros(TXT_LEN, 3),
    )
    return kwargs, img_ids


def _forward(model, kwargs):
    with torch.no_grad():
        return model(**kwargs, return_dict=False)[0]


def _randomize(opro, std=0.3):
    with torch.no_grad():
        for operator in opro.operators:
            operator.L.normal_(std=std)
            operator.R.normal_(std=std)


def test_one_operator_per_attention_layer():
    model = _tiny_flux()
    opro = add_opro_to_flux(model, num_panels=2, rank=4)
    assert len(opro.operators) == 2 + 3
    assert all(op.head_dim == 16 for op in opro.operators)
    assert sum(p.numel() for p in opro.parameters()) == 5 * 2 * 16 * 4 * 2
    # The operators are not registered inside the backbone.
    assert not any("operator" in name or "opro" in name for name, _ in model.named_parameters())


def test_paper_parameter_count():
    """FLUX: 57 layers, d_h = 128, rho = 32, 2 panels -> +0.93M parameters."""
    assert 57 * 2 * 128 * 32 * 2 == 933_888


def test_zero_init_reproduces_backbone():
    model = _tiny_flux()
    kwargs, img_ids = _inputs()
    reference = _forward(model, kwargs)
    opro = add_opro_to_flux(model, num_panels=2, rank=4)
    opro.set_panel_ids(compute_panel_ids(img_ids, panel_rows=1, panel_cols=2))
    assert torch.allclose(_forward(model, kwargs), reference, atol=1e-5)


def test_trained_operators_change_output_and_can_be_disabled():
    model = _tiny_flux()
    kwargs, img_ids = _inputs()
    reference = _forward(model, kwargs)
    opro = add_opro_to_flux(model, num_panels=2, rank=4)
    opro.set_panel_ids(compute_panel_ids(img_ids, panel_rows=1, panel_cols=2))
    _randomize(opro)
    assert not torch.allclose(_forward(model, kwargs), reference, atol=1e-4)
    opro.enabled = False
    assert torch.allclose(_forward(model, kwargs), reference, atol=1e-5)


def test_single_panel_layout_is_invariant():
    """If every image token is in one panel, any operator leaves image-image logits unchanged.

    Text tokens are not rotated, so the text-image logits would change; we therefore
    check the invariance on the attention processor directly, on image tokens only.
    """
    model = _tiny_flux()
    opro = add_opro_to_flux(model, num_panels=2, rank=4)
    _randomize(opro)
    operator = opro.operators[0]
    q, k = torch.randn(1, 2, 32, 16), torch.randn(1, 2, 32, 16)
    q_hat, k_hat = operator.apply_to_qk(q, k, torch.ones(1, 32, dtype=torch.long))
    assert torch.allclose(q_hat @ k_hat.transpose(-1, -2), q @ k.transpose(-1, -2), atol=1e-4)


def test_gradients_reach_operators_only():
    model = _tiny_flux()
    model.requires_grad_(False)
    kwargs, img_ids = _inputs()
    opro = add_opro_to_flux(model, num_panels=2, rank=4)
    opro.set_panel_ids(compute_panel_ids(img_ids, panel_rows=1, panel_cols=2))
    model(**kwargs, return_dict=False)[0].pow(2).mean().backward()
    assert all(op.L.grad is not None and op.L.grad.abs().sum() > 0 for op in opro.operators)


def test_shared_operators():
    model = _tiny_flux()
    opro = add_opro_to_flux(model, num_panels=2, rank=4, share_across_layers=True)
    assert len(opro.operators) == 1


def test_save_and_load(tmp_path):
    model = _tiny_flux()
    kwargs, img_ids = _inputs()
    panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=2)
    opro = add_opro_to_flux(model, num_panels=2, rank=4)
    opro.set_panel_ids(panel_ids)
    _randomize(opro)
    expected = _forward(model, kwargs)
    opro.save(str(tmp_path))

    restored_model = _tiny_flux()
    restored = load_opro_for_flux(restored_model, str(tmp_path))
    restored.set_panel_ids(panel_ids)
    assert torch.allclose(_forward(restored_model, kwargs), expected, atol=1e-5)
