#!/bin/bash
# Design ablations (Table 2): RoPE backbone, 3x3 grid, LoRA r=8, operator rank 4.
# Requires the Stage-1 checkpoint written by `scripts/run_benchmark.sh rope`.
set -euo pipefail

OUT=${OUT:-output}
STAGE2="python -m compositional_reasoning.train --stage 2 --pos_type rope --grid_size 3 --bf16
        --pretrain_ckpt ${OUT}/stage1_rope/last.pt --use_lora"

${STAGE2} --output_dir "${OUT}/ablation_lora"
${STAGE2} --use_opro --opro_rank 4 --opro_variant abl1 --output_dir "${OUT}/ablation_apb"
${STAGE2} --use_opro --opro_rank 4 --opro_variant abl2 --output_dir "${OUT}/ablation_asym"
${STAGE2} --use_opro --opro_rank 4 --opro_symmetric_init --output_dir "${OUT}/ablation_no_zero_init"
${STAGE2} --use_opro --opro_rank 4 --output_dir "${OUT}/ablation_opro"

python scripts/collect_results.py "${OUT}"
