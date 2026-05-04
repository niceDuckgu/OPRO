#!/usr/bin/env bash
# Stage 2: grid reasoning (2k steps, freeze backbone, train LoRA + OPRO)
# Usage:  bash scripts/train_stage2.sh rope opro 3 out/stage1/best.pt
set -euo pipefail

POS_TYPE="${1:-rope}"
OPRO_VARIANT="${2:-opro}"
GRID_SIZE="${3:-3}"
PRETRAIN_CKPT="${4:-./out/stage1_${POS_TYPE}/best.pt}"
OUTPUT_DIR="${5:-./out/stage2_${POS_TYPE}_${OPRO_VARIANT}_g${GRID_SIZE}}"

python -m compositional_reasoning.train \
    --stage 2 \
    --pos_type "${POS_TYPE}" \
    --size base \
    --grid_size "${GRID_SIZE}" \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --use_lora --lora_rank 8 \
    --use_opro --opro_rank 8 --opro_variant "${OPRO_VARIANT}" \
    --freeze_backbone \
    --max_steps 2000 \
    --batch_size 256 \
    --lr 5e-4 \
    --output_dir "${OUTPUT_DIR}"
