#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${BIO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-${PROJECT_ROOT}/results/reruns/cross_dataset}"

echo "============================================================"
echo "Cross-dataset tuned final benchmarks"
echo "  MoE: KinetX"
echo "  BiCoA-Net: 2773"
echo "  Each: warm + drug-cold + protein-cold, five runs"
echo "============================================================"

# Run serially across models to avoid GPU-memory contention.
bash "${SCRIPT_DIR}/run_moe_kinetx_best_params_all.sh"
bash "${SCRIPT_DIR}/run_bicoa_2773_best_params_all.sh"

if [[ "${PREFLIGHT_ONLY:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_final_results.py" \
    --results-root "${OUTPUT_ROOT}"
  echo "Combined mean+/-std: ${OUTPUT_ROOT}/combined_mean_std_display.csv"
fi
echo "All requested final benchmarks finished."
