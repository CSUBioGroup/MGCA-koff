#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export ESM2_PATH="${ESM2_PATH:-/root/private_data/DP/pretrained_model/esm2_t36}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${HERE}/outputs}"
export DEVICE="${DEVICE:-cuda:0}"
export ESM_BATCH_SIZE=1
export INCLUDE_ESM="${INCLUDE_ESM:-0}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=43
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
