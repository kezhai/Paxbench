#!/usr/bin/env bash
set -euo pipefail

DATA=${1:-data/demo/paxbench_demo.tsv}
OUT=${2:-results/demo}
PYTHON=${PYTHON:-python}
MODELS=(esm calm mrnabert)
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}

"$PYTHON" scripts/run_benchmark.py \
  --data "$DATA" \
  --output "$OUT" \
  --models "${MODELS[@]}" \
  --modes embedding zeroshot lora \
  --learners ridge mlp \
  --epochs 1 \
  --continue-on-error

echo "Demo complete. Scores: $OUT/scores.tsv"
