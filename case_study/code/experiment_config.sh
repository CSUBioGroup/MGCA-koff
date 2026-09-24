#!/usr/bin/env bash
# Scientific settings are frozen in frozen/*.json. Only paths/scheduling below vary.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ESM2_PATH="${ESM2_PATH:-${PROJECT_ROOT}/../pretrained_model/esm2_t36}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/case_study_outputs/mgca_final_unbounded_2773_v1}"
export DEVICE="${DEVICE:-cuda:0}"
export RUN_JOBS="${RUN_JOBS:-5}"
export PHASE="${PHASE:-all}"
export ESM_BATCH_SIZE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=42
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
# Optional separate Python environment with PyMOL; auto uses a labeled matplotlib
# structure rendering fallback if PyMOL is unavailable. Numerical analysis is not skipped.
export STRUCTURE_PYTHON_BIN="${STRUCTURE_PYTHON_BIN:-${PYTHON_BIN}}"
export STRUCTURE_RENDERER="${STRUCTURE_RENDERER:-auto}"
export MIN_FREE_GIB="${MIN_FREE_GIB:-5}"
