# Integrations: dropping OPRO into existing in-context generation baselines

This directory contains **minimal patches** for the four baselines from the
paper. We do not vendor any baseline — each subdirectory ships:

* a short `README.md` describing the upstream files to modify,
* an `opro_processor.py` (or equivalent) implementing the OPRO-aware
  attention processor / forward hook,
* a `patch.diff` showing the exact upstream diff (when applicable).

| Baseline | Path | Backbone | Attention pattern |
|---|---|---|---|
| ACE++ | [`ace_plus/`](ace_plus) | FluxFill (single stream) | inline `apply()` inside `attention_forward` |
| InsertAnything | [`insert_anything/`](insert_anything) | FluxFill + Redux | custom processor read from `model_config["_opro_ctx"]` |
| UNO | [`uno/`](uno) | Flux + dual-stream | `DoubleStreamBlockOproProcessor` |

The shared OPRO module lives at the repo root: `opro/`. Each integration
imports `OPROLieLowRank`, `OPROConfig`, `compute_panel_ids` from there;
the baselines themselves are unchanged on disk apart from the documented
diff.

---

## Common shape of every integration

Every patch boils down to this 3-step pattern, applied at each attention
layer of the backbone:

```python
# 1) After RoPE is applied to q, k:
q_img = q[:, :, img_start:img_end]
k_img = k[:, :, img_start:img_end]

# 2) Apply OPRO on the image-token slice:
q_img, k_img = opro_modules[layer_idx].apply_to_qk(q_img, k_img, panel_ids)

# 3) Put them back and proceed to scaled_dot_product_attention as usual:
q[:, :, img_start:img_end] = q_img
k[:, :, img_start:img_end] = k_img
```

The differences between baselines come down to (a) where RoPE is applied,
(b) whether the architecture is single-stream or dual-stream, and (c) how
the panel context is plumbed in.
