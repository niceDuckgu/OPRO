# Integration: UNO

UNO ("Less-to-More Generalization", Wu et al., 2025) uses a *dual-stream*
Flux: image and text run through separate single-stream blocks before
joining. OPRO is applied on the image branch only, post-RoPE, inside a
custom `DoubleStreamBlockOproProcessor`.

## Setup

```bash
git clone https://github.com/bytedance/UNO /your/uno
cd /your/uno && pip install -r requirements.txt
```

## Files to modify

* `uno/flux/util.py` — attach `opro_modules` to the DiT during model build.
* `uno/flux/modules/layers.py` — register `DoubleStreamBlockOproProcessor`
  on each `DoubleStreamBlock`.

The diff is in [`patch.diff`](patch.diff).

## How to attach OPRO at runtime

```python
from opro import compute_panel_ids
from integrations.uno.opro_processor import (
    build_uno_opro_modules, install_panel_context,
)

opro_modules = build_uno_opro_modules(model.dit, num_panels=2, rank=32)
panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=2)
install_panel_context(model.dit, opro_modules, panel_ids)
```

## Why dual-stream matters

In UNO each block computes Q/K/V for image and text separately. To preserve
intra-stream behavior we apply OPRO **only** on the image-stream Q/K *after*
its RoPE. The text branch is untouched, so text-token same-panel invariance
is trivial (text has no panels). The patch in `patch.diff` shows the slice
extraction + manager call.
