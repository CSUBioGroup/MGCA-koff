#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${BIO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${TRAINING_TIME_OUTPUT_ROOT:-${PROJECT_ROOT}/training_time_benchmark_outputs_aligned}"
ESM2_PATH="${ESM2_PATH:-${PROJECT_ROOT}/../pretrained_model/esm2_t36}"
BEST_PARAMS="${MGCA_BEST_PARAMS:-${PROJECT_ROOT}/mgca_hyperparameter_tuning/corrected_model_window_search_results_v4/mgca_morgan/KinetX/best_params.json}"

# Some downloaded local archives used this extra container directory. The
# online project does not; this is only a safe fallback for archive validation.
if [[ ! -f "${BEST_PARAMS}" ]]; then
  FALLBACK_PARAMS="${PROJECT_ROOT}/mgca_hyperparameter_tuning/kinetx_v4_final_results/corrected_model_window_search_results_v4/mgca_morgan/KinetX/best_params.json"
  if [[ -f "${FALLBACK_PARAMS}" ]]; then
    BEST_PARAMS="${FALLBACK_PARAMS}"
  fi
fi

[[ -f "${BEST_PARAMS}" ]] || { echo "ERROR: MGCA best_params.json not found: ${BEST_PARAMS}" >&2; exit 1; }
[[ -e "${ESM2_PATH}" ]] || { echo "ERROR: ESM2 path not found: ${ESM2_PATH}" >&2; exit 1; }
[[ -d "${OUTPUT_ROOT}/moe_kinetx_tuned" ]] || {
  echo "ERROR: existing aligned MoE results are missing: ${OUTPUT_ROOT}/moe_kinetx_tuned" >&2
  exit 1
}

CONCURRENCY=1
mkdir -p "${OUTPUT_ROOT}"

echo "Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "Project root: ${PROJECT_ROOT}"
echo "MGCA config: ${BEST_PARAMS}"
echo "ESM2 path: ${ESM2_PATH}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Concurrency: ${CONCURRENCY} (enforced serially)"
echo "Planned runs: MGCA-Morgan/KinetX tuned 15"

"${PYTHON_BIN}" "${SCRIPT_DIR}/collect_benchmark_environment.py" \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT_ROOT}/mgca_aligned_benchmark_environment.json"

CONFIG_DIR="$(dirname "${BEST_PARAMS}")"
"${PYTHON_BIN}" "${PROJECT_ROOT}/mgca_hyperparameter_tuning/validate_selected_mgca_config.py" \
  --config-dir "${CONFIG_DIR}" --dataset KinetX
"${PYTHON_BIN}" "${PROJECT_ROOT}/mgca_hyperparameter_tuning/verify_corrected_ablation_model.py"

RUNNER_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --best-params "${BEST_PARAMS}"
  --esm2-path "${ESM2_PATH}"
  --device "${DEVICE:-auto}"
  --num-workers "${NUM_WORKERS:-0}"
  --base-seed "${BASE_SEED:-42}"
  --seed-step "${SEED_STEP:-100}"
  --existing-result-action "${EXISTING_RESULT_ACTION:-skip}"
)

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_mgca_aligned_training_times.py" \
    "${RUNNER_ARGS[@]}" --preflight-only
  exit 0
fi

# Cache construction is an explicitly separate preflight and is never included
# in the primary training timer or in the per-run subprocess wall clock.
if [[ "${SKIP_ESM_CACHE_PREFLIGHT:-0}" != "1" ]]; then
  CONFIG_ROOT="$(dirname "${CONFIG_DIR}")"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/mgca_hyperparameter_tuning/precompute_best_params_esm_caches.py" \
    --datasets KinetX \
    --protocols warm drug_cold protein_cold \
    --esm2-path "${ESM2_PATH}" \
    --config-root "${CONFIG_ROOT}" \
    --fingerprint morgan \
    --device "${CACHE_DEVICE:-cuda:0}" \
    --batch-size "${ESM_CACHE_BATCH_SIZE:-2}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_mgca_aligned_training_times.py" "${RUNNER_ARGS[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_mgca_aligned_training_times.py" \
  --output-root "${OUTPUT_ROOT}"

echo "MGCA aligned timing benchmark complete: ${OUTPUT_ROOT}"
