#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${BIO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-${PROJECT_ROOT}/results/reruns/cross_dataset}"
BEST_PARAMS="${MOE_KINETX_BEST_PARAMS:-${PROJECT_ROOT}/stage_data/moe/KinetX/best_params.json}"

ARGS=(
  --model moe
  --dataset KinetX
  --best-params "${BEST_PARAMS}"
  --output-root "${OUTPUT_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --device "${DEVICE:-auto}"
  --num-workers "${NUM_WORKERS:-0}"
  --max-parallel "${MOE_MAX_PARALLEL:-5}"
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

echo "MoE tuned final benchmark: KinetX warm/drug-cold/protein-cold"
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_final_experiments.py" "${ARGS[@]}"
