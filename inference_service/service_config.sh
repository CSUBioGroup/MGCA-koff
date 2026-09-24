#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export BUNDLE_DIR="${BUNDLE_DIR:-${HERE}}"
if [[ -d "${BUNDLE_DIR}/esm2_t36" ]]; then
  export ESM2_PATH="${ESM2_PATH:-${BUNDLE_DIR}/esm2_t36}"
else
  export ESM2_PATH="${ESM2_PATH:-/root/private_data/DP/pretrained_model/esm2_t36}"
fi
export DEVICE="${DEVICE:-cuda:0}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-64}"
export MAX_PROTEIN_CACHE="${MAX_PROTEIN_CACHE:-1024}"
export MAX_DRUG_CACHE="${MAX_DRUG_CACHE:-10000}"
export MAX_PAIRS="${MAX_PAIRS:-4096}"
export MAX_BODY_BYTES="${MAX_BODY_BYTES:-8388608}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Set API_KEY in the shell; never put a secret in a distributable archive.
