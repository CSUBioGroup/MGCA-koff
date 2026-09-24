#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNING_OUTPUT_ROOT="${TUNING_OUTPUT_ROOT:-${SCRIPT_DIR}/kinetx_v4_window_search_results}"
FULL_RESULT_ROOT="${FULL_RESULT_ROOT:-${SCRIPT_DIR}/kinetx_v4_full_results}"

echo "Phase 1/2: KinetX v4 warm-validation-MSE hyperparameter selection"
DATASETS=KinetX FINGERPRINTS=morgan \
MODEL_VARIANT=mha_interaction_rmsnorm_sigmoid_fixed3h_2expert_v4_window_sweep \
PROTOCOL_VERSION=mgca_mha_interaction_2expert_warm_only_tpe_top5_window_v4 \
STUDY_NAME_SUFFIX=mha_interaction_2expert_warm_window_v4 \
MOE_NUM_EXPERTS=2 WINDOW_SIZES="1 2 4 6 8" \
WINDOW_LAYOUT=legacy_anchors_v1 OUTPUT_ROOT="${TUNING_OUTPUT_ROOT}" \
bash "${SCRIPT_DIR}/run_warm_mse_tuning.sh"

echo "Phase 2/2: frozen KinetX Full runs on warm/drug-cold/protein-cold"
CONFIG_ROOT="${TUNING_OUTPUT_ROOT}/mgca_morgan" \
RESULT_ROOT="${FULL_RESULT_ROOT}" \
bash "${SCRIPT_DIR}/run_kinetx_full.sh" "$@"
