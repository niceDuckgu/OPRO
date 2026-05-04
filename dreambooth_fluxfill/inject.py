"""
Inject OPRO into a Flux Fill Transformer
==========================================

This module shows how to attach OPRO to an existing Flux Fill model.
The integration has three steps:

  1. Create per-layer OPROLieLowRank modules (one per Transformer layer).
  2. Compute panel IDs from Flux's ``img_ids`` coordinate tensor.
  3. Patch the attention forward to apply OPRO to the image-token Q/K.

Usage:
    from diffusers import FluxFillPipeline
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", ...)

    opro_modules = inject_opro(pipe.transformer, num_panels=2, rank=32)
    # opro_modules is an nn.ModuleList — add its parameters to your optimizer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opro import OPROLieLowRank, OPROConfig, compute_panel_ids


# ---------------------------------------------------------------
# Step 1: Create OPRO modules
# ---------------------------------------------------------------

def create_opro_modules(
    head_dim: int,
    num_layers: int,
    config: OPROConfig,
) -> nn.ModuleList:
    """Create one OPROLieLowRank module per Transformer layer.

    Args:
        head_dim:    Attention head dimension (e.g. 128 for Flux).
        num_layers:  Total number of attention layers
                     (= len(transformer_blocks) + len(single_transformer_blocks)).
        config:      OPRO configuration.

    Returns:
        nn.ModuleList of OPROLieLowRank modules.
    """
    if config.share_across_layers:
        shared = OPROLieLowRank(
            num_panels=config.num_panels,
            head_dim=head_dim,
            rank=config.rank,
            variant=config.variant,
        )
        return nn.ModuleList([shared] * num_layers)

    return nn.ModuleList([
        OPROLieLowRank(
            num_panels=config.num_panels,
            head_dim=head_dim,
            rank=config.rank,
            variant=config.variant,
        )
        for _ in range(num_layers)
    ])


# ---------------------------------------------------------------
# Step 2: Compute panel IDs
# ---------------------------------------------------------------

def compute_flux_panel_ids(
    img_ids: torch.Tensor,
    config: OPROConfig,
    batch_size: int = 1,
) -> torch.Tensor:
    """Compute panel IDs from Flux-style image coordinate tensor.

    In Flux, ``img_ids`` has shape ``[seq_len, 3]`` where each row is
    ``[batch_idx, row, col]``.  This function bins those coordinates
    into the panel grid specified by ``config``.

    Args:
        img_ids:     Flux image ID tensor ``[seq_len, 3]`` or ``[batch, seq_len, 3]``.
        config:      OPRO configuration (panel_rows, panel_cols).
        batch_size:  Batch size to expand the result to.

    Returns:
        Panel IDs ``[batch, img_seq_len]``.
    """
    panel_ids = compute_panel_ids(
        img_ids,
        panel_rows=config.panel_rows,
        panel_cols=config.panel_cols,
        panel_aliases=config.panel_aliases,
    )
    # Expand to batch dimension
    if panel_ids.ndim == 1:
        panel_ids = panel_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    return panel_ids


# ---------------------------------------------------------------
# Step 3: Apply OPRO in attention
# ---------------------------------------------------------------

def apply_opro_to_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    opro_module: OPROLieLowRank,
    panel_ids: torch.Tensor,
    txt_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply OPRO rotations to the image portion of Q/K tensors.

    In Flux attention, the token sequence is ``[text_tokens, image_tokens]``.
    OPRO is applied only to the image tokens.

    Args:
        query:        Full Q tensor ``[batch, heads, txt+img, head_dim]``.
        key:          Full K tensor ``[batch, heads, txt+img, head_dim]``.
        opro_module:  OPROLieLowRank for this layer.
        panel_ids:    Panel IDs ``[batch, img_seq_len]``.
        txt_seq_len:  Number of text tokens (image starts after this).

    Returns:
        (query, key) with OPRO applied to the image portion.
    """
    img_start = txt_seq_len
    img_end = img_start + panel_ids.shape[1]

    if img_end > query.shape[2]:
        return query, key

    panel_ids = panel_ids.to(device=query.device, dtype=torch.long)

    # Extract image Q/K
    q_img = query[:, :, img_start:img_end, :]
    k_img = key[:, :, img_start:img_end, :]

    # Apply OPRO rotation
    q_rot, k_rot = opro_module.apply_to_qk(q_img, k_img, panel_ids)

    # Replace in full sequence
    query = query.clone()
    key = key.clone()
    query[:, :, img_start:img_end, :] = q_rot
    key[:, :, img_start:img_end, :] = k_rot

    return query, key


# ---------------------------------------------------------------
# High-level: inject into Flux Transformer
# ---------------------------------------------------------------

def inject_opro(
    transformer: nn.Module,
    num_panels: int = 2,
    panel_rows: int = 1,
    panel_cols: int = 2,
    rank: int = 32,
    variant: str = "opro",
    share_across_layers: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
) -> nn.ModuleList:
    """Inject OPRO modules into a Flux Transformer and return them.

    This function:
      1. Detects the model's head_dim and layer count.
      2. Creates per-layer OPROLieLowRank modules.
      3. Stores them on the transformer for use during forward pass.

    Args:
        transformer:  A FluxTransformer2DModel (from diffusers).
        num_panels:   Number of panels (2 for diptych, 3 for triptych, etc.).
        panel_rows:   Panel grid rows.
        panel_cols:   Panel grid columns.
        rank:         OPRO low-rank dimension (rho).
        variant:      "opro" | "abl1" | "abl2".
        share_across_layers: Share one module across all layers.
        dtype:        Parameter dtype (float32 recommended for stability).
        device:       Device string (auto-detected if None).

    Returns:
        nn.ModuleList of OPRO modules. Add ``.parameters()`` to your optimizer.

    Example::

        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", ...)
        opro_modules = inject_opro(pipe.transformer, num_panels=2, rank=32)

        # Add to optimizer with separate LR
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": 1e-4},
            {"params": opro_modules.parameters(), "lr": 1e-4},
        ])
    """
    config = OPROConfig(
        num_panels=num_panels,
        panel_rows=panel_rows,
        panel_cols=panel_cols,
        rank=rank,
        variant=variant,
        share_across_layers=share_across_layers,
    )

    # Detect head_dim
    inner_dim = getattr(transformer, "inner_dim", None)
    if inner_dim is None:
        cfg = getattr(transformer, "config", {})
        inner_dim = getattr(cfg, "inner_dim", None) or getattr(cfg, "hidden_size", None)
    num_heads = getattr(transformer, "num_attention_heads", None)
    if num_heads is None:
        cfg = getattr(transformer, "config", {})
        num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "num_heads", None)

    if inner_dim is None or num_heads is None:
        raise ValueError(
            "Cannot detect head_dim from transformer. "
            "Pass a FluxTransformer2DModel or set inner_dim/num_attention_heads manually."
        )
    head_dim = int(inner_dim) // int(num_heads)

    # Count total layers
    n_joint = len(getattr(transformer, "transformer_blocks", []))
    n_single = len(getattr(transformer, "single_transformer_blocks", []))
    total_layers = n_joint + n_single

    if total_layers == 0:
        raise ValueError("No transformer blocks found.")

    # Create modules
    modules = create_opro_modules(head_dim, total_layers, config)

    if device is None:
        try:
            device = next(transformer.parameters()).device
        except StopIteration:
            device = "cpu"
    modules = modules.to(device=device, dtype=dtype)

    # Store on transformer for access during forward
    transformer.opro_modules = modules
    transformer.opro_config = config

    n_params = sum(p.numel() for p in modules.parameters())
    print(f"[OPRO] Injected {total_layers} layers | "
          f"head_dim={head_dim}, rank={rank}, panels={num_panels} | "
          f"+{n_params / 1e6:.2f}M params")

    return modules


# ---------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------

def save_opro(modules: nn.ModuleList, config: OPROConfig, path: str) -> None:
    """Save OPRO weights and config to a directory."""
    import os
    os.makedirs(path, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in modules.state_dict().items()}
    torch.save(state, os.path.join(path, "opro.pt"))
    config.save(os.path.join(path, "opro_config.json"))


def load_opro(
    transformer: nn.Module,
    path: str,
    dtype: torch.dtype = torch.float32,
    device: Optional[str] = None,
) -> nn.ModuleList:
    """Load OPRO weights from a checkpoint directory.

    Args:
        transformer:  The Flux Transformer to inject into.
        path:         Directory containing ``opro.pt`` and ``opro_config.json``.
        dtype:        Parameter dtype.
        device:       Target device.

    Returns:
        nn.ModuleList with loaded weights.

    Notes:
        Production checkpoints written by the OPROManager wrapper use a
        ``_orpo_layers.{i}.L`` key prefix. We strip that prefix on load so
        the same file works in both the release format (bare ModuleList
        keys ``{i}.L``) and the production format.
    """
    import os
    import json

    cfg_path = os.path.join(path, "opro_config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    # Production configs use lowercase keys ('lie_rank', 'mode'); release uses 'rank', 'variant'.
    # Normalize so OPROConfig.from_dict accepts either layout.
    config = OPROConfig.from_dict(raw)

    modules = inject_opro(
        transformer,
        num_panels=config.num_panels,
        panel_rows=config.panel_rows,
        panel_cols=config.panel_cols,
        rank=config.rank,
        variant=config.variant,
        share_across_layers=config.share_across_layers,
        dtype=dtype,
        device=device,
    )
    state = torch.load(os.path.join(path, "opro.pt"), map_location="cpu", weights_only=True)

    # Strip production wrapper prefix if present.
    if any(k.startswith("_orpo_layers.") for k in state.keys()):
        state = {k.replace("_orpo_layers.", "", 1): v for k, v in state.items()}

    # Drop keys that don't exist in the release schema (production has extras like
    # gate state, hyper-network params for unused modes, etc.).
    target_keys = set(modules.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in target_keys}
    missing, unexpected = modules.load_state_dict(filtered, strict=False)
    if missing:
        # Production may not save L_key/R_key/panel_bias for ablation variants we
        # don't enable; only warn for genuinely missing main params.
        critical = [m for m in missing if not (".L_key" in m or ".R_key" in m or ".panel_bias" in m)]
        if critical:
            raise RuntimeError(f"Missing OPRO state keys: {critical[:5]} ...")
    return modules
