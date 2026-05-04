# Instructional Editing (Track B)

Reproduction of the **MagicBrush** result in Sec. 4.2 / Table 3 of the paper:

| Method | L1 ↓ | CLIP-I ↑ | DINO ↑ |
|---|---|---|---|
| ICEdit (LoRA-only) | 0.1189 | 0.8703 | 0.7706 |
| **ICEdit + OPRO (ours)** | **0.0781** | **0.9002** | **0.8531** |

This directory contains:

* `wrapper.py` — inference wrapper around a *user-supplied* ICEdit clone.
  We do **not** vendor ICEdit; you point the wrapper at your own clone.
* `train_joint.py` — reference joint LoRA + OPRO training script that
  produced the released checkpoint.

---

## 1. One-time setup

```bash
# Clone ICEdit (we do not vendor it — license + diff churn).
git clone https://github.com/River-Zhang/ICEdit /your/icedit

# Download our joint LoRA + OPRO weights from HF Hub.
huggingface-cli download <ORG>/opro-icedit-magicbrush --local-dir ./ckpts/opro_icedit
```

The HF Hub repo bundles **both** the LoRA *and* the OPRO weights because
they were trained jointly. You should not mix our OPRO with ICEdit's own
released LoRA — the LoRA in our checkpoint adapted to the OPRO modulation
during joint training.

## 2. Inference

```bash
python -m instructional_editing.wrapper \
    --icedit_path /your/icedit \
    --flux_path /path/to/FLUX.1-Fill-dev \
    --ckpt_dir ./ckpts/opro_icedit \
    --image source.png \
    --instruction "Make the cat wear sunglasses" \
    --out edited.png
```

The wrapper builds the diptych canvas, attaches OPRO panel context, and
delegates the actual diffusion to ICEdit's pipeline factory.

## 3. (Optional) Re-train from scratch

```bash
python -m instructional_editing.train_joint \
    --flux_path /path/to/FLUX.1-Fill-dev \
    --data_root /path/to/magicbrush_train \
    --output_dir ./output/icedit_opro \
    --max_steps 5000
```

Hyper-parameters match the paper Sec. 4.2:

| | Value |
|---|---|
| LoRA rank (r) | 16 |
| OPRO rank (ρ) | 32 |
| Optimizer | AdamW, lr 1e-4, weight decay 0.01 |
| Steps | 5,000 |
| Batch | 1 × 8 grad accum |
| Precision | bfloat16 |

## 4. Why not just publish the inference code?

Reviewer-friendly answer: the OPRO weights alone are not actionable — the
LoRA was learned alongside them. Publishing both as a single ckpt is the
only honest way to make the table reproducible without re-training.

Engineering answer: separating the LoRA into a peft adapter and the OPRO
into our own format keeps the repo small (only +0.93 M parameters for
OPRO; LoRA adds ~22 M for ICEdit). HF download is therefore < 100 MB.
