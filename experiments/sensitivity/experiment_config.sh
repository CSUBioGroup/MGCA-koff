#!/usr/bin/env bash
# Scientific settings are frozen by plan.json. Changing them requires a NEW output root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export ESM2_PATH="${ESM2_PATH:-${PROJECT_ROOT}/../pretrained_model/esm2_t36}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/sensitivity_v1}"
export CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export DEVICE="${DEVICE:-cuda:0}"
export RUN_JOBS="${RUN_JOBS:-5}"
export DATASETS="${DATASETS:-2773 KinetX}"
export PROTOCOLS="${PROTOCOLS:-warm drug_cold protein_cold}"
export N_RUNS="${N_RUNS:-5}"
export SEED="${SEED:-42}"
export SEED_STEP="${SEED_STEP:-100}"
# benchmark: warm/drug = original 5 splits with seed 42; protein = split 1, five seeds.
# paired_seeds: all protocols use seeds 42+100*(run-1), warm/drug cycle five folds.
export SEED_POLICY="${SEED_POLICY:-benchmark}"
export CAP_MULTIPLIERS="${CAP_MULTIPLIERS:-0.5 1 2}"
export INIT_MULTIPLIERS="${INIT_MULTIPLIERS:-0.5 1 2}"
# Optional two extra equal-cap configurations (0.15,0.15), (0.30,0.30): +60 runs.
export INCLUDE_EQUAL_CAPS="${INCLUDE_EQUAL_CAPS:-false}"
export MIN_FREE_GIB="${MIN_FREE_GIB:-5}"
export AMP="${AMP:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export MPLBACKEND=Agg
# all | plan | preflight | cache | run | summary
export PHASE="${PHASE:-all}"
