#!/usr/bin/env bash
# Stage 1: single-panel pretext (50k steps, ViT-B from scratch)
# Usage:  bash scripts/train_stage1.sh rope ./out/stage1
set -euo pipefail

POS_TYPE="${1:-rope}"
OUTPUT_DIR="${2:-./out/stage1_${POS_TYPE}}"

python -m compositional_reasoning.train \
    --stage 1 \
    --pos_type "${POS_TYPE}" \
    --size base \
    --max_steps 50000 \
    --batch_size 256 \
    --lr 1e-3 \
    --output_dir "${OUTPUT_DIR}"
