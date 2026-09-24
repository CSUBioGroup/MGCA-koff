#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "1/2 MoE: 2773-origin method -> tune on KinetX"
echo "============================================================"
bash "${SCRIPT_DIR}/run_moe_kinetx_tuning.sh"

echo "============================================================"
echo "2/2 BiCoA-Net: KinetX-origin method -> tune on 2773"
echo "============================================================"
bash "${SCRIPT_DIR}/run_bicoa_2773_tuning.sh"

echo "============================================================"
echo "Both cross-dataset Bayesian tuning studies completed."
echo "Results: ${CROSS_TUNING_OUTPUT_ROOT:-${SCRIPT_DIR}/hyperparameter_tuning}"
echo "============================================================"
