#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${BIO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-${PROJECT_ROOT}/results/reruns/cross_dataset}"
BEST_PARAMS="${BICOA_2773_BEST_PARAMS:-${PROJECT_ROOT}/stage_data/bicoa/2773/best_params.json}"
CACHE_ROOT="${BICOA_FEATURE_CACHE:-${PROJECT_ROOT}/bicoa_cross_tuning_cache}"

if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
  echo "BiCoA-Net feature-cache preflight for all 2773 final splits"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_bicoa_2773_all.py" \
    --device "${DEVICE:-auto}" \
    --project-root "${PROJECT_ROOT}" \
    --cache-root "${CACHE_ROOT}"
fi

ARGS=(
  --model bicoa
  --dataset 2773
  --best-params "${BEST_PARAMS}"
  --output-root "${OUTPUT_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --cache-root "${CACHE_ROOT}"
  --device "${DEVICE:-auto}"
  --num-workers "${NUM_WORKERS:-0}"
  --max-parallel "${BICOA_MAX_PARALLEL:-1}"
  --base-seed "${BASE_SEED:-42}"
  --seed-step "${SEED_STEP:-100}"
  --existing-result-action "${EXISTING_RESULT_ACTION:-skip}"
)
if [[ "${KEEP_CHECKPOINTS:-0}" == "1" ]]; then
  ARGS+=(--keep-checkpoints)
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  ARGS+=(--preflight-only)
fi

echo "BiCoA-Net tuned final benchmark: 2773 warm/drug-cold/protein-cold"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_final_experiments.py" "${ARGS[@]}"
