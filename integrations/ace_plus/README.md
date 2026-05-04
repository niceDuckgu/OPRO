# Integration: ACE++

Drop OPRO into ACE++'s FluxFill-based pipeline. ACE++ uses a single-stream
attention block, so the patch is a one-place insert.

## Setup

```bash
git clone https://github.com/ali-vilab/ACE_plus /your/ace_plus
cd /your/ace_plus && pip install -e .
```

## Files to modify

ACE++ release locates its FluxFill blocks at `modules/layers.py`. The
inline patch sits in the joint `attention_forward` (around line 381–428
in the version we tested). The diff is in [`patch.diff`](patch.diff).

The high-level shape:

```python
# modules/layers.py: attention_forward(...)
opro_kwargs = kwargs.get("opro_kwargs", {})
opro = opro_kwargs.get("opro_module")
panel_ids = opro_kwargs.get("panel_ids")
img_start = opro_kwargs.get("img_start", 0)

if opro is not None and panel_ids is not None:
    q_img = q[:, :, img_start:]
    k_img = k[:, :, img_start:]
    q_img, k_img = opro.apply_to_qk(q_img, k_img, panel_ids)
    q[:, :, img_start:] = q_img
    k[:, :, img_start:] = k_img

attn = scaled_dot_product_attention(q, k, v, ...)
```

## How to attach OPRO at runtime

```python
from opro import compute_panel_ids
from integrations.ace_plus.opro_processor import build_ace_opro_modules, install_panel_context

# 1) Inject OPRO modules into the ACE++ transformer.
opro_modules = build_ace_opro_modules(ace_pipe.transformer, num_panels=2, rank=32)

# 2) Compute panel ids from the diptych canvas geometry.
panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=2)

# 3) Stash them on the transformer; the patched forward will pick them up.
install_panel_context(ace_pipe.transformer, opro_modules, panel_ids)
```
