#!/usr/bin/env bash
# Expanded, identical initialization grids; preserve the previous experiment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/unbounded_warm_hpo_wide_init_v2}"
export CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache}"
export DRUG_INIT_CHOICES="${DRUG_INIT_CHOICES:-0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5}"
export JOINT_INIT_CHOICES="${JOINT_INIT_CHOICES:-0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5}"
# Original LR/WD/batch/dropout/window search, 30 TPE trials, Top-5 review,
# all training parameters and concurrency defaults remain unchanged.
source "${SCRIPT_DIR}/experiment_config.sh"
