"""OPRO for FLUX transformers (diffusers).

``add_opro_to_flux`` replaces the attention processor of every joint and single
block of a ``FluxTransformer2DModel`` with one that applies the panel operators
to the image-token queries and keys, right after the backbone's RoPE and right
before the attention product (Eq. 3-4 of the paper):

    from diffusers import FluxFillPipeline
    from opro import add_opro_to_flux, compute_panel_ids

    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev")
    opro = add_opro_to_flux(pipe.transformer, num_panels=2, rank=32)   # +0.93M params
    opro.set_panel_ids(compute_panel_ids(img_ids, panel_rows=1, panel_cols=2))

``opro`` is a regular ``nn.Module`` that owns the operators. It is deliberately
*not* registered inside the transformer: it stays in float32 when the backbone
is cast to half precision and is untouched by ``transformer.requires_grad_``.

Tested with diffusers 0.32 (``FluxAttnProcessor2_0`` call signature).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .opro import OPROLieLowRank
from .utils import OPROConfig


class OPROFluxAttnProcessor:
    """``FluxAttnProcessor2_0`` with OPRO applied to the image-token Q/K."""

    def __init__(self, operator: OPROLieLowRank, owner: "FluxOPRO") -> None:
        self.operator = operator
        self.owner = owner

    def _apply_opro(self, query: torch.Tensor, key: torch.Tensor):
        panel_ids = self.owner.panel_ids
        if panel_ids is None or not self.owner.enabled:
            return query, key
        num_img = panel_ids.shape[-1]
        seq_len = query.shape[2]
        if num_img > seq_len:
            raise ValueError(
                f"panel_ids covers {num_img} image tokens but the sequence has only {seq_len}"
            )
        # FLUX orders the joint sequence as [text tokens, image tokens].
        start = seq_len - num_img
        q_img, k_img = self.operator.apply_to_qk(
            query[:, :, start:], key[:, :, start:], panel_ids, gate=self.owner.gate
        )
        query = torch.cat([query[:, :, :start], q_img], dim=2)
        key = torch.cat([key[:, :, :start], k_img], dim=2)
        return query, key

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        batch_size = hidden_states.shape[0]

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, -1, attn.heads, x.shape[-1] // attn.heads).transpose(1, 2)

        query = split_heads(attn.to_q(hidden_states))
        key = split_heads(attn.to_k(hidden_states))
        value = split_heads(attn.to_v(hidden_states))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Joint blocks receive the text stream separately; single blocks do not.
        if encoder_hidden_states is not None:
            text_query = split_heads(attn.add_q_proj(encoder_hidden_states))
            text_key = split_heads(attn.add_k_proj(encoder_hidden_states))
            text_value = split_heads(attn.add_v_proj(encoder_hidden_states))
            if attn.norm_added_q is not None:
                text_query = attn.norm_added_q(text_query)
            if attn.norm_added_k is not None:
                text_key = attn.norm_added_k(text_key)
            query = torch.cat([text_query, query], dim=2)
            key = torch.cat([text_key, key], dim=2)
            value = torch.cat([text_value, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query, key = self._apply_opro(query, key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, value.shape[1] * value.shape[-1])
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            num_text = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states[:, :num_text], hidden_states[:, num_text:]
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


class FluxOPRO(nn.Module):
    """Container for the per-layer operators attached to a FLUX transformer.

    Attributes:
        operators:  ``nn.ModuleList`` with one ``OPROLieLowRank`` per attention layer
                    (joint blocks first, then single blocks).
        panel_ids:  Panel index of every image token, set with ``set_panel_ids``.
        gate:       Scalar multiplier on the generators (1.0 = full operator).
        enabled:    Set to ``False`` to recover the unmodified backbone.
    """

    def __init__(self, operators: nn.ModuleList, config: OPROConfig) -> None:
        super().__init__()
        self.operators = operators
        self.config = config
        self.panel_ids: Optional[torch.Tensor] = None
        self.gate: float = 1.0
        self.enabled: bool = True

    def set_panel_ids(self, panel_ids: Optional[torch.Tensor]) -> None:
        """Set the panel index of every image token: ``[num_img]`` or ``[batch, num_img]``.

        The image tokens are assumed to be the last ``num_img`` tokens of the
        attention sequence. Use a negative index for tokens that should not be
        rotated.
        """
        self.panel_ids = panel_ids

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.operators.state_dict().items()},
                   os.path.join(path, "opro.pt"))
        self.config.save(os.path.join(path, "opro_config.json"))

    def load_weights(self, path: str) -> None:
        state = torch.load(os.path.join(path, "opro.pt"), map_location="cpu", weights_only=True)
        self.operators.load_state_dict(state)


def _attention_modules(transformer: nn.Module) -> list:
    blocks = list(getattr(transformer, "transformer_blocks", [])) + list(
        getattr(transformer, "single_transformer_blocks", [])
    )
    if not blocks:
        raise ValueError("No transformer blocks found; expected a FluxTransformer2DModel.")
    return [block.attn for block in blocks]


def add_opro_to_flux(
    transformer: nn.Module,
    num_panels: int = 2,
    rank: int = 32,
    *,
    panel_rows: int = 1,
    panel_cols: Optional[int] = None,
    variant: str = "opro",
    share_across_layers: bool = False,
    symmetric_init: bool = False,
) -> FluxOPRO:
    """Attach OPRO to every attention layer of a FLUX transformer.

    Args:
        transformer:  ``FluxTransformer2DModel``.
        num_panels:   Number of panel operators P.
        rank:         Rank rho of the generators (32 in the paper's FLUX experiments).
        panel_rows, panel_cols:  Panel grid stored in the config for
                      ``compute_panel_ids`` (defaults to one row of ``num_panels``).
        variant:      "opro" | "abl1" | "abl2".
        share_across_layers:  Use a single operator set for all layers.
        symmetric_init:  Random (non-identity) initialization.

    Returns:
        A ``FluxOPRO`` module on the transformer's device, in float32.
    """
    attns = _attention_modules(transformer)
    head_dim = attns[0].to_q.out_features // attns[0].heads
    config = OPROConfig(
        num_panels=num_panels,
        panel_rows=panel_rows,
        panel_cols=panel_cols if panel_cols is not None else num_panels,
        rank=rank,
        variant=variant,
        share_across_layers=share_across_layers,
    )

    def make() -> OPROLieLowRank:
        return OPROLieLowRank(num_panels, head_dim, rank, variant=variant, symmetric_init=symmetric_init)

    if share_across_layers:
        operators = nn.ModuleList([make()])
        per_layer = [operators[0]] * len(attns)
    else:
        operators = nn.ModuleList([make() for _ in attns])
        per_layer = list(operators)

    opro = FluxOPRO(operators, config)
    opro.to(device=next(transformer.parameters()).device, dtype=torch.float32)
    for attn, operator in zip(attns, per_layer):
        attn.set_processor(OPROFluxAttnProcessor(operator, opro))
    return opro


def load_opro_for_flux(transformer: nn.Module, path: str) -> FluxOPRO:
    """Re-create the operators saved by ``FluxOPRO.save`` and attach them."""
    config = OPROConfig.load(os.path.join(path, "opro_config.json"))
    opro = add_opro_to_flux(
        transformer,
        num_panels=config.num_panels,
        rank=config.rank,
        panel_rows=config.panel_rows,
        panel_cols=config.panel_cols,
        variant=config.variant,
        share_across_layers=config.share_across_layers,
    )
    opro.load_weights(path)
    return opro
