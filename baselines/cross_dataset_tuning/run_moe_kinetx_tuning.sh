#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${CROSS_TUNING_OUTPUT_ROOT:-${SCRIPT_DIR}/hyperparameter_tuning}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" --model moe
echo "MoE cross-dataset tuning: published 2773 method -> KinetX warm validation"
"${PYTHON_BIN}" "${SCRIPT_DIR}/tune_warm.py" \
  --model moe \
  --dataset KinetX \
  --output-root "${OUTPUT_ROOT}" \
  --n-trials "${N_TRIALS:-30}" \
  --top-k "${TOP_K:-5}" \
  --search-jobs "${SEARCH_JOBS:-1}" \
  --review-jobs "${REVIEW_JOBS:-1}" \
  --max-failed-trials "${MAX_FAILED_TRIALS:-30}" \
  --device "${DEVICE:-auto}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --seed "${SEED:-42}" \
  --sampler-seed "${SAMPLER_SEED:-2026}"

echo "MoE selected parameters: ${OUTPUT_ROOT}/moe/KinetX/best_params.json"
