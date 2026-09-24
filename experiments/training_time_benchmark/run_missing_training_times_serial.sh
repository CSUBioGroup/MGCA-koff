#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${BIO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${TRAINING_TIME_OUTPUT_ROOT:-${PROJECT_ROOT}/training_time_benchmark_outputs}"
CACHE_ROOT="${BICOA_FEATURE_CACHE:-${PROJECT_ROOT}/bicoa_cross_tuning_cache}"

# This benchmark intentionally exposes no MAX_PARALLEL setting. The Python
# scheduler launches one subprocess, waits for it, and only then launches the next.
CONCURRENCY=1

mkdir -p "${OUTPUT_ROOT}"

echo "Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "Project root: ${PROJECT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Concurrency: ${CONCURRENCY} (enforced serially)"
echo "Planned formal runs: BiCoA/KinetX 15 + BiCoA/2773 tuned 15 + MoE/KinetX tuned 15"

"${PYTHON_BIN}" "${SCRIPT_DIR}/collect_benchmark_environment.py" \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT_ROOT}/benchmark_environment.json"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_missing_training_times.py" \
    --project-root "${PROJECT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --device "${DEVICE:-auto}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --base-seed "${BASE_SEED:-42}" \
    --seed-step "${SEED_STEP:-100}" \
    --existing-result-action "${EXISTING_RESULT_ACTION:-skip}" \
    --preflight-only
  exit 0
fi

if [[ "${SKIP_FEATURE_PRECOMPUTE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_bicoa_training_time_features.py" \
    --project-root "${PROJECT_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --output-dir "${OUTPUT_ROOT}" \
    --device "${DEVICE:-auto}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_missing_training_times.py" \
  --project-root "${PROJECT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --device "${DEVICE:-auto}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --base-seed "${BASE_SEED:-42}" \
  --seed-step "${SEED_STEP:-100}" \
  --existing-result-action "${EXISTING_RESULT_ACTION:-skip}"

SUMMARY_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
)
if [[ -n "${MGCA_REFERENCE_CSV:-}" ]]; then
  SUMMARY_ARGS+=(--reference-final-per-run "${MGCA_REFERENCE_CSV}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_missing_training_times.py" "${SUMMARY_ARGS[@]}"

echo "Training-time benchmark complete: ${OUTPUT_ROOT}"
