# Integration: InsertAnything

InsertAnything ("Insert Anything: Image Insertion via In-Context Editing")
runs FluxFill + Redux. The text and image streams are processed
separately and joined inside attention. OPRO is applied to the image
slice of joint Q/K, after RoPE, inside a custom attention forward.

## Setup

```bash
git clone https://github.com/song-wensong/insert-anything /your/insert_anything
cd /your/insert_anything && pip install -r requirements.txt
```

## Files to modify

* `src/models/transformer.py` — call `compute_panel_ids` on `img_ids` and
  store the result in `model_config["_opro_ctx"]` so each block can read
  it without changing the block signature.
* `src/models/block.py` — extend `attn_forward()` to read `_opro_ctx` and
  apply OPRO on the image-token slice. The diff is in [`patch.diff`](patch.diff).

## How to attach OPRO at runtime

```python
from opro import compute_panel_ids
from integrations.insert_anything.opro_processor import (
    build_insert_opro_modules, install_panel_context,
)

opro_modules = build_insert_opro_modules(model.transformer, num_panels=2, rank=32)
panel_ids = compute_panel_ids(img_ids, panel_rows=1, panel_cols=2)
install_panel_context(model.transformer, opro_modules, panel_ids,
                      img_start_token=text_seq_len)
```
