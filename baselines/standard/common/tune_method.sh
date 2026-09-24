#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash common/tune_method.sh <method> [KinetX|2773] [tune_warm.py arguments...]" >&2
  exit 2
fi
METHOD="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
N_TRIALS="${N_TRIALS:-30}"
TOP_K="${TOP_K:-5}"
TUNE_JOBS="${TUNE_JOBS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_FAILED_TRIALS="${MAX_FAILED_TRIALS:-30}"

case "${METHOD}" in
  deepdta) DEFAULT_REVIEW_JOBS=5 ;;
  graphdta) DEFAULT_REVIEW_JOBS=2 ;;
  attentiondta) DEFAULT_REVIEW_JOBS=1 ;;
  *) echo "ERROR: unsupported method: ${METHOD}" >&2; exit 2 ;;
esac
REVIEW_JOBS="${REVIEW_JOBS:-${DEFAULT_REVIEW_JOBS}}"

if [[ "$#" -gt 0 && ( "$1" == "KinetX" || "$1" == "2773" ) ]]; then
  DATASETS=("$1")
  shift
else
  DATASETS=(KinetX 2773)
fi
EXTRA_ARGS=("$@")

for dataset in "${DATASETS[@]}"; do
  echo "================================================================"
  echo "Warm-only tuning: ${METHOD} / ${dataset}"
  echo "Stage 1: ${N_TRIALS} completed TPE trials; Stage 2: top ${TOP_K} on five warm validation runs"
  echo "================================================================"
  command=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/tune_warm.py"
    --model "${METHOD}"
    --dataset "${dataset}"
    --output-root "${TUNING_OUTPUT_ROOT:-${BASELINE_ROOT}/hyperparameter_tuning}"
    --n-trials "${N_TRIALS}"
    --top-k "${TOP_K}"
    --search-jobs "${TUNE_JOBS}"
    --review-jobs "${REVIEW_JOBS}"
    --max-failed-trials "${MAX_FAILED_TRIALS}"
    --device "${DEVICE}"
    --num-workers "${NUM_WORKERS}"
  )
  if [[ -n "${TUNE_EPOCHS:-}" ]]; then
    command+=(--epochs "${TUNE_EPOCHS}")
  fi
  if [[ -n "${TUNE_PATIENCE:-}" ]]; then
    command+=(--patience "${TUNE_PATIENCE}")
  fi
  if [[ "${KEEP_TUNING_CHECKPOINTS:-0}" == "1" ]]; then
    command+=(--keep-checkpoints)
  fi
  command+=("${EXTRA_ARGS[@]}")
  "${command[@]}"
done
