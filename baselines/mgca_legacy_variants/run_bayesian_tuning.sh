#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${BIO_PROJECT_ROOT:-${ONLINE_ROOT}}"
export BIO_PROJECT_ROOT="${DATA_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
ESM2_PATH="${ESM2_PATH:-${DATA_ROOT}/../pretrained_model/esm2_t36}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/hyperparameter_tuning}"
MGCA_SCRIPT="${MGCA_SCRIPT:-${SCRIPT_DIR}/ESM_Morgan_Hybrid_Fusion_nonredundant.py}"
MODEL_DEPENDENCY_SCRIPT="${MODEL_DEPENDENCY_SCRIPT:-${SCRIPT_DIR}/ESM_Morgan_Hybrid_Fusion.py}"
MODEL_VARIANT="${MODEL_VARIANT:-mha_interaction_rmsnorm_sigmoid_fixed3h_2expert}"
PROTOCOL_VERSION="${PROTOCOL_VERSION:-mgca_single_objective_warm_val_mse_tpe_top5_v1}"
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-2}"
WINDOW_SIZES="${WINDOW_SIZES:-1 2 4 6 8}"
WINDOW_LAYOUT="${WINDOW_LAYOUT:-legacy_anchors_v1}"
STUDY_NAME_SUFFIX="${STUDY_NAME_SUFFIX:-online_bundle_warm_mse_v1}"

# Paper defaults: warm-only selection, 30 TPE trials, top-5 review on five runs.
N_TRIALS="${N_TRIALS:-30}"
TOP_K="${TOP_K:-5}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-15}"
SEARCH_JOBS="${SEARCH_JOBS:-1}"
REVIEW_JOBS="${REVIEW_JOBS:-1}"
ESM_BATCH_SIZE="${ESM_BATCH_SIZE:-2}"
MAX_FAILED_TRIALS="${MAX_FAILED_TRIALS:-30}"
SEED="${SEED:-42}"
SAMPLER_SEED="${SAMPLER_SEED:-2026}"
ACCEPT_MODEL_HASH_CHANGE="${ACCEPT_MODEL_HASH_CHANGE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

RESUME_ARGS=()
if [[ "${ACCEPT_MODEL_HASH_CHANGE}" == "1" ]]; then
  RESUME_ARGS+=(--accept-model-hash-change)
elif [[ "${ACCEPT_MODEL_HASH_CHANGE}" != "0" ]]; then
  echo "ERROR: ACCEPT_MODEL_HASH_CHANGE must be 0 or 1" >&2
  exit 2
fi

# Defaults tune MGCA-Morgan on 2773. Set DATASETS="KinetX 2773" for both.
# Examples:
#   DATASETS="KinetX" bash mgca_hyperparameter_tuning/run_bayesian_tuning.sh
#   FINGERPRINTS="morgan fcfp" bash mgca_hyperparameter_tuning/run_bayesian_tuning.sh
read -r -a DATASET_LIST <<< "${DATASETS:-2773}"
read -r -a FINGERPRINT_LIST <<< "${FINGERPRINTS:-morgan}"
read -r -a WINDOW_SIZE_LIST <<< "${WINDOW_SIZES}"
if [[ "${#WINDOW_SIZE_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: WINDOW_SIZES must contain at least one positive integer" >&2
  exit 2
fi
for window_size in "${WINDOW_SIZE_LIST[@]}"; do
  if [[ ! "${window_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid window size: ${window_size}" >&2
    exit 2
  fi
done

if [[ ! -f "${SCRIPT_DIR}/tune_mgca_warm.py" ]]; then
  echo "ERROR: missing ${SCRIPT_DIR}/tune_mgca_warm.py" >&2
  exit 1
fi
if [[ ! -f "${MGCA_SCRIPT}" ]]; then
  echo "ERROR: missing MGCA training entry: ${MGCA_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DEPENDENCY_SCRIPT}" ]]; then
  echo "ERROR: missing MGCA model dependency: ${MODEL_DEPENDENCY_SCRIPT}" >&2
  exit 1
fi
if [[ ! -e "${ESM2_PATH}" ]]; then
  echo "ERROR: ESM2_PATH does not exist: ${ESM2_PATH}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "optuna") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("ERROR: missing Python packages: " + ", ".join(missing))
print("Dependency preflight OK: torch, optuna")
PY

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_corrected_ablation_model.py"

for fingerprint in "${FINGERPRINT_LIST[@]}"; do
  case "${fingerprint}" in
    morgan|fcfp) ;;
    *) echo "ERROR: unsupported fingerprint: ${fingerprint}" >&2; exit 2 ;;
  esac
  for dataset in "${DATASET_LIST[@]}"; do
    case "${dataset}" in
      KinetX|2773) ;;
      *) echo "ERROR: unsupported dataset: ${dataset}" >&2; exit 2 ;;
    esac
    echo "================================================================"
    echo "MGCA Bayesian selection: ${fingerprint} / ${dataset}"
    echo "Selection objective: warm validation MSE only (test data unavailable)"
    echo "Model variant: ${MODEL_VARIANT}"
    echo "ESM2 windows: ${WINDOW_SIZE_LIST[*]} (layout=${WINDOW_LAYOUT})"
    echo "Stage 1: ${N_TRIALS} completed TPE trials on warm run1"
    echo "Stage 2: serial ESM2 cache preflight, then top ${TOP_K} configurations on five warm validation runs"
    echo "Results: ${OUTPUT_ROOT}/mgca_${fingerprint}/${dataset}"
    echo "================================================================"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/tune_mgca_warm.py" \
      --dataset "${dataset}" \
      --fingerprint "${fingerprint}" \
      --output-root "${OUTPUT_ROOT}" \
      --mgca-script "${MGCA_SCRIPT}" \
      --model-dependency-script "${MODEL_DEPENDENCY_SCRIPT}" \
      --model-variant "${MODEL_VARIANT}" \
      --protocol-version "${PROTOCOL_VERSION}" \
      --window-sizes "${WINDOW_SIZE_LIST[@]}" \
      --window-layout "${WINDOW_LAYOUT}" \
      --study-name-suffix "${STUDY_NAME_SUFFIX}" \
      --moe-num-experts "${MOE_NUM_EXPERTS}" \
      --esm2-path "${ESM2_PATH}" \
      --n-trials "${N_TRIALS}" \
      --top-k "${TOP_K}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --search-jobs "${SEARCH_JOBS}" \
      --review-jobs "${REVIEW_JOBS}" \
      --esm-batch-size "${ESM_BATCH_SIZE}" \
      --max-failed-trials "${MAX_FAILED_TRIALS}" \
      --seed "${SEED}" \
      --sampler-seed "${SAMPLER_SEED}" \
      --device "${DEVICE}" \
      "${RESUME_ARGS[@]}"
  done
done

echo "================================================================"
echo "All requested MGCA Bayesian searches completed."
echo "Best configs: ${OUTPUT_ROOT}/mgca_<fingerprint>/<dataset>/best_params.json"
echo "================================================================"
