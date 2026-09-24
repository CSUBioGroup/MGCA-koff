#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${BIO_PROJECT_ROOT:-${ONLINE_ROOT}}"
export BIO_PROJECT_ROOT="${DATA_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
ESM2_PATH="${ESM2_PATH:-${DATA_ROOT}/../pretrained_model/esm2_t36}"
CONFIG_ROOT="${CONFIG_ROOT:-${ONLINE_ROOT}/stage_data/mgca_morgan}"
CONFIG_DIR="${CONFIG_ROOT}/KinetX"
RESULT_ROOT="${RESULT_ROOT:-${ONLINE_ROOT}/results/reruns/mgca_kinetx}"
TRAIN_SCRIPT="${SCRIPT_DIR}/ESM_Morgan_Hybrid_Fusion_nonredundant.py"
MODEL_DEPENDENCY_SCRIPT="${SCRIPT_DIR}/ESM_Morgan_Hybrid_Fusion.py"
RUN_JOBS="${RUN_JOBS:-5}"
ESM_BATCH_SIZE="${ESM_BATCH_SIZE:-2}"
SEED="${SEED:-42}"
SEED_STEP="${SEED_STEP:-100}"
SAVE_MODELS="${SAVE_MODELS:-false}"
PRECOMPUTE_ESM="${PRECOMPUTE_ESM:-true}"
read -r -a PROTOCOL_LIST <<< "${PROTOCOLS:-warm drug_cold protein_cold}"
EXTRA_ARGS=("$@")

if (( RUN_JOBS < 1 || RUN_JOBS > 5 )); then
  echo "ERROR: RUN_JOBS must be between 1 and 5" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_SCRIPT}" || ! -f "${MODEL_DEPENDENCY_SCRIPT}" ]]; then
  echo "ERROR: ONLINE MGCA implementation is incomplete" >&2
  exit 1
fi
for protocol in "${PROTOCOL_LIST[@]}"; do
  case "${protocol}" in
    warm|drug_cold|protein_cold) ;;
    *) echo "ERROR: unsupported protocol: ${protocol}" >&2; exit 2 ;;
  esac
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_selected_config.py" \
  --config-dir "${CONFIG_DIR}" --dataset KinetX
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_corrected_ablation_model.py"

# shellcheck disable=SC1090
source "${CONFIG_DIR}/best_params.sh"
for name in MGCA_CONFIG_ID MGCA_LR MGCA_WEIGHT_DECAY MGCA_BATCH_SIZE \
  MGCA_DROPOUT MGCA_WINDOW_SIZE MGCA_WINDOW_LAYOUT MGCA_EPOCHS \
  MGCA_PATIENCE MGCA_VAL_FREQ MGCA_MOE_NUM_EXPERTS; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} missing from ${CONFIG_DIR}/best_params.sh" >&2
    exit 1
  fi
done
if [[ "${MGCA_MOE_NUM_EXPERTS}" != "2" ]]; then
  echo "ERROR: expected the corrected two-expert model" >&2
  exit 1
fi

if [[ "${PRECOMPUTE_ESM}" == "true" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_esm_caches.py" \
    --datasets KinetX --protocols "${PROTOCOL_LIST[@]}" \
    --esm2-path "${ESM2_PATH}" --config-root "${CONFIG_ROOT}" \
    --device "${DEVICE}" --batch-size "${ESM_BATCH_SIZE}"
elif [[ "${PRECOMPUTE_ESM}" != "false" ]]; then
  echo "ERROR: PRECOMPUTE_ESM must be true or false" >&2
  exit 2
fi

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    printf '%s\n' unavailable
  fi
}

TRAIN_HASH="$(file_sha256 "${TRAIN_SCRIPT}")"
DEPENDENCY_HASH="$(file_sha256 "${MODEL_DEPENDENCY_SCRIPT}")"
AUDIT_FILE="${RESULT_ROOT}/selected_config.txt"
mkdir -p "${RESULT_ROOT}"
if [[ -f "${AUDIT_FILE}" ]]; then
  existing_config="$(awk -F= '$1=="config_id" {print $2; exit}' "${AUDIT_FILE}")"
  existing_train_hash="$(awk -F= '$1=="training_script_sha256" {print $2; exit}' "${AUDIT_FILE}")"
  existing_dependency_hash="$(awk -F= '$1=="model_dependency_sha256" {print $2; exit}' "${AUDIT_FILE}")"
  if [[ "${existing_config}" != "${MGCA_CONFIG_ID}" \
      || "${existing_train_hash}" != "${TRAIN_HASH}" \
      || "${existing_dependency_hash}" != "${DEPENDENCY_HASH}" ]]; then
    echo "ERROR: RESULT_ROOT contains an incompatible config or implementation" >&2
    echo "Use a new RESULT_ROOT instead of mixing production runs." >&2
    exit 1
  fi
else
  printf '%s\n' \
    "dataset=KinetX" "selection=warm_validation_mse_only" \
    "config_id=${MGCA_CONFIG_ID}" "lr=${MGCA_LR}" \
    "weight_decay=${MGCA_WEIGHT_DECAY}" "batch_size=${MGCA_BATCH_SIZE}" \
    "dropout=${MGCA_DROPOUT}" "window_size=${MGCA_WINDOW_SIZE}" \
    "window_layout=${MGCA_WINDOW_LAYOUT}" "moe_num_experts=2" \
    "training_script_sha256=${TRAIN_HASH}" \
    "model_dependency_sha256=${DEPENDENCY_HASH}" > "${AUDIT_FILE}"
fi

input_paths() {
  local protocol="$1" run="$2" root
  case "${protocol}" in
    warm)
      root="${DATA_ROOT}/KinetX/random_split_mgca_input"
      printf '%s\n' "${root}/train_run${run}.csv" \
        "${root}/val_run${run}.csv" "${root}/test_run${run}.csv"
      ;;
    drug_cold)
      root="${DATA_ROOT}/KinetX/drug_cold_start_canonical_5fold/fold${run}"
      printf '%s\n' "${root}/train.csv" "${root}/val.csv" "${root}/test.csv"
      ;;
    protein_cold)
      root="${DATA_ROOT}/KinetX/cold_start"
      printf '%s\n' "${root}/train.csv" "${root}/val.csv" "${root}/test.csv"
      ;;
  esac
}

is_complete() {
  local output="$1"
  compgen -G "${output}/metrics_*.txt" >/dev/null 2>&1 \
    && compgen -G "${output}/test_predictions_*.txt" >/dev/null 2>&1
}

run_one() {
  local protocol="$1" run="$2" cold_mode run_seed split output_base run_output log_dir
  mapfile -t paths < <(input_paths "${protocol}" "${run}")
  for path in "${paths[@]}"; do
    [[ -f "${path}" ]] || { echo "ERROR: missing ${path}" >&2; return 1; }
  done
  case "${protocol}" in
    warm) cold_mode=pair; run_seed="${SEED}" ;;
    drug_cold) cold_mode=drug; run_seed="${SEED}" ;;
    protein_cold) cold_mode=target; run_seed=$((SEED + (run - 1) * SEED_STEP)) ;;
  esac
  split="${protocol}_run${run}"
  output_base="${RESULT_ROOT}/KinetX/${protocol}/MGCA_Morgan"
  run_output="${output_base}/fixed_split/${split}"
  log_dir="${RESULT_ROOT}/logs/KinetX/${protocol}"
  mkdir -p "${log_dir}"
  if is_complete "${run_output}"; then
    echo "Skipping completed KinetX/${protocol}/run${run}"
    return 0
  fi
  echo "Running KinetX/${protocol}/run${run}; seed=${run_seed}; config=${MGCA_CONFIG_ID}"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${EXTRA_ARGS[@]}" \
    --train_csv "${paths[0]}" --val_csv "${paths[1]}" --test_csv "${paths[2]}" \
    --split_name "${split}" --esm2_path "${ESM2_PATH}" --output_dir "${output_base}" \
    --fingerprint_type morgan --cold_start_mode "${cold_mode}" --ablation no \
    --lr "${MGCA_LR}" --weight_decay "${MGCA_WEIGHT_DECAY}" \
    --batch_size "${MGCA_BATCH_SIZE}" --dropout "${MGCA_DROPOUT}" \
    --window_size "${MGCA_WINDOW_SIZE}" --window_layout "${MGCA_WINDOW_LAYOUT}" \
    --epochs "${MGCA_EPOCHS}" --patience "${MGCA_PATIENCE}" \
    --val_freq "${MGCA_VAL_FREQ}" --seed "${run_seed}" --device "${DEVICE}" \
    --save_models "${SAVE_MODELS}" 2>&1 | tee "${log_dir}/full_${split}.log"
}

wait_group() {
  local failed=0 index
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
      echo "ERROR: ${LABELS[$index]} failed" >&2
      failed=1
    fi
  done
  PIDS=(); LABELS=()
  return "${failed}"
}

overall_status=0
for protocol in "${PROTOCOL_LIST[@]}"; do
  PIDS=(); LABELS=()
  for run in 1 2 3 4 5; do
    output="${RESULT_ROOT}/KinetX/${protocol}/MGCA_Morgan/fixed_split/${protocol}_run${run}"
    if is_complete "${output}"; then
      echo "Skipping completed KinetX/${protocol}/run${run}"
      continue
    fi
    run_one "${protocol}" "${run}" &
    PIDS+=("$!"); LABELS+=("KinetX/${protocol}/run${run}")
    if (( ${#PIDS[@]} >= RUN_JOBS )); then
      wait_group || overall_status=1
    fi
  done
  if (( ${#PIDS[@]} > 0 )); then
    wait_group || overall_status=1
  fi
done
if (( overall_status != 0 )); then
  echo "ERROR: one or more Full runs failed; rerun to resume incomplete runs" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_kinetx_full.py" \
  --result-root "${RESULT_ROOT}"

echo "============================================================"
echo "KinetX v4-selected Full experiments complete"
echo "Config: ${CONFIG_DIR}/best_params.json"
echo "Results: ${RESULT_ROOT}"
echo "Completed runs are skipped independently on rerun."
echo "============================================================"
