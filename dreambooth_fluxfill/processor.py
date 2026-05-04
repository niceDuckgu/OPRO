"""OPRO-aware drop-in replacement for diffusers' ``FluxAttnProcessor2_0``.

Tested against ``diffusers==0.32.0``. The processor mirrors the upstream
implementation byte-for-byte, then inserts a single OPRO call on the
image-token slice of (Q, K) right after the rotary embedding is applied
and before ``scaled_dot_product_attention``.

Wiring pattern:

  1. Build per-layer OPRO modules with ``inject_opro(...)`` (see ``inject.py``).
  2. Stash a panel-id tensor on the transformer:
        ``transformer._opro_ctx = {"panel_ids": ..., "txt_seq_len": ...}``
  3. Call ``install_opro_processors(transformer, opro_modules)`` once.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opro import OPROLieLowRank


class OPROFluxAttnProcessor:
    """Drop-in for FluxAttnProcessor2_0 that applies OPRO on the image slice."""

    def __init__(self, layer_idx: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("OPROFluxAttnProcessor requires PyTorch >= 2.0")
        self.layer_idx = int(layer_idx)

    def _resolve_ctx(self, attn) -> Optional[dict]:
        """Find the transformer's OPRO context.

        We walk up via ``attn._opro_ctx_owner`` (set by ``install_opro_processors``)
        and fall back to inspecting the attached attention module hierarchy.
        """
        ctx = getattr(attn, "_opro_ctx_owner", None)
        if ctx is None:
            return None
        return getattr(ctx, "_opro_ctx", None)

    def _resolve_modules(self, attn) -> Optional[nn.ModuleList]:
        ctx_owner = getattr(attn, "_opro_ctx_owner", None)
        if ctx_owner is None:
            return None
        return getattr(ctx_owner, "opro_modules", None)

    def __call__(
        self,
        attn,                       # diffusers Attention
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # ---- Mirror FluxAttnProcessor2_0 up to the RoPE step ----------------
        bsz, _, _ = (encoder_hidden_states if encoder_hidden_states is not None else hidden_states).shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            ehs_q = attn.add_q_proj(encoder_hidden_states)
            ehs_k = attn.add_k_proj(encoder_hidden_states)
            ehs_v = attn.add_v_proj(encoder_hidden_states)
            ehs_q = ehs_q.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            ehs_k = ehs_k.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            ehs_v = ehs_v.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                ehs_q = attn.norm_added_q(ehs_q)
            if attn.norm_added_k is not None:
                ehs_k = attn.norm_added_k(ehs_k)
            txt_seq_len = ehs_q.shape[2]
            query = torch.cat([ehs_q, query], dim=2)
            key = torch.cat([ehs_k, key], dim=2)
            value = torch.cat([ehs_v, value], dim=2)
        else:
            # Single block: text+image already concatenated; need txt_seq_len from context.
            ctx = self._resolve_ctx(attn) or {}
            txt_seq_len = int(ctx.get("txt_seq_len", 0))

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # ---- OPRO insert: image slice only ---------------------------------
        modules = self._resolve_modules(attn)
        ctx = self._resolve_ctx(attn) or {}
        panel_ids = ctx.get("panel_ids")
        if modules is not None and panel_ids is not None and txt_seq_len > 0:
            img_start = txt_seq_len
            img_end = img_start + panel_ids.shape[1]
            if img_end <= query.shape[2]:
                pids = panel_ids.to(device=query.device, dtype=torch.long)
                if pids.shape[0] != query.shape[0]:
                    pids = pids.expand(query.shape[0], -1).contiguous()
                q_img = query[:, :, img_start:img_end, :]
                k_img = key[:, :, img_start:img_end, :]
                q_img, k_img = modules[self.layer_idx].apply_to_qk(q_img, k_img, pids)
                query = query.clone()
                key = key.clone()
                query[:, :, img_start:img_end, :] = q_img
                key[:, :, img_start:img_end, :] = k_img

        # ---- Standard attention completion ---------------------------------
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        out = out.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
        out = out.to(query.dtype)

        if encoder_hidden_states is not None:
            ehs_out, hs_out = out[:, :encoder_hidden_states.shape[1]], out[:, encoder_hidden_states.shape[1]:]
            hs_out = attn.to_out[0](hs_out)
            hs_out = attn.to_out[1](hs_out)
            ehs_out = attn.to_add_out(ehs_out)
            return hs_out, ehs_out
        return out


def install_opro_processors(transformer: nn.Module, opro_modules: nn.ModuleList) -> None:
    """Replace every block's attention processor with ``OPROFluxAttnProcessor``.

    The transformer is expected to expose ``transformer_blocks`` (joint) and
    ``single_transformer_blocks`` (single-stream) — the FluxFill / Flux layout.

    Per-step state (panel_ids + txt_seq_len) lives on
    ``transformer._opro_ctx``; processors read it through
    ``attn._opro_ctx_owner`` (set here once via ``object.__setattr__`` to
    bypass nn.Module's child-registration — otherwise we'd create a
    transformer→attn→transformer cycle that breaks ``.train()``/``.eval()``).
    """
    if not hasattr(transformer, "opro_modules"):
        object.__setattr__(transformer, "opro_modules", opro_modules)
    if not hasattr(transformer, "_opro_ctx"):
        object.__setattr__(transformer, "_opro_ctx", {})

    layer_idx = 0
    for block in getattr(transformer, "transformer_blocks", []):
        object.__setattr__(block.attn, "_opro_ctx_owner", transformer)
        block.attn.set_processor(OPROFluxAttnProcessor(layer_idx))
        layer_idx += 1
    for block in getattr(transformer, "single_transformer_blocks", []):
        object.__setattr__(block.attn, "_opro_ctx_owner", transformer)
        block.attn.set_processor(OPROFluxAttnProcessor(layer_idx))
        layer_idx += 1
