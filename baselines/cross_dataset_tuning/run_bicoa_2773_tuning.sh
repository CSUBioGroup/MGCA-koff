#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${CROSS_TUNING_OUTPUT_ROOT:-${SCRIPT_DIR}/hyperparameter_tuning}"
PROJECT_ROOT="${BIO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CACHE_ROOT="${BICOA_FEATURE_CACHE:-${PROJECT_ROOT}/bicoa_cross_tuning_cache}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" --model bicoa
echo "BiCoA-Net cache preflight: 2773 warm train/validation only"
"${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_bicoa_2773_warm.py" \
  --device "${DEVICE:-auto}" \
  --cache-root "${CACHE_ROOT}"

echo "BiCoA-Net cross-dataset tuning: published KinetX method -> 2773 warm validation"
BICOA_FEATURE_CACHE="${CACHE_ROOT}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/tune_warm.py" \
  --model bicoa \
  --dataset 2773 \
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

echo "BiCoA-Net selected parameters: ${OUTPUT_ROOT}/bicoa/2773/best_params.json"
