#!/bin/bash
# Two-stage compositional reasoning benchmark (Table 1) for one positional encoding:
# Stage-1 pretraining, then linear probe / LoRA / LoRA + OPRO on 2x2, 3x3 and 4x4 grids.
#
#   bash scripts/run_benchmark.sh rope        # ape | rope | liere | comrope
set -euo pipefail

POS=${1:-rope}
OUT=${OUT:-output}
TRAIN="python -m compositional_reasoning.train --pos_type ${POS} --bf16"

STAGE1=${OUT}/stage1_${POS}
[ -f "${STAGE1}/last.pt" ] || ${TRAIN} --stage 1 --output_dir "${STAGE1}"

for N in 2 3 4; do
    STAGE2="${TRAIN} --stage 2 --grid_size ${N} --pretrain_ckpt ${STAGE1}/last.pt"
    ${STAGE2} --output_dir "${OUT}/${POS}_${N}x${N}_probe"
    ${STAGE2} --use_lora --output_dir "${OUT}/${POS}_${N}x${N}_lora"
    ${STAGE2} --use_lora --use_opro --output_dir "${OUT}/${POS}_${N}x${N}_lora_opro"
done

python scripts/collect_results.py "${OUT}"
