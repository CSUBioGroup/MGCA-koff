#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}}"
STAGE2_ROOT="${STAGE2_ROOT:-${PROJECT_ROOT}/results/case_study/factor_xa_stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/case_study/factor_xa_stage3_docking}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: python is not available on PATH." >&2
  exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${PYTHON_VERSION}" != "3.10" ]]; then
  echo "ERROR: expected Python 3.10, found ${PYTHON_VERSION} at ${PYTHON_BIN}." >&2
  exit 1
fi

for required in "${WORK_DIR}/factor_xa_stage3.py" "${STAGE2_ROOT}/KinetX/factor_xa_compound_catalog.csv" "${STAGE2_ROOT}/2773/factor_xa_compound_catalog.csv"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required file is missing: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}/logs"

echo "Python: ${PYTHON_BIN} (${PYTHON_VERSION})"
echo "Stage-2 input: ${STAGE2_ROOT}"
echo "Stage-3 output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" "${WORK_DIR}/factor_xa_stage3.py" \
  --work-dir "${WORK_DIR}" \
  --stage2-root "${STAGE2_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --seeds "101,201,301,401,501,601,701,801,901,1001" \
  --cpu 8 \
  --exhaustiveness 32 \
  --n-poses 20 \
  --energy-range 5 \
  --redock-rmsd 2.0 \
  --redock-min-seed-passes 5 \
  --cluster-energy-window 2.0 \
  --cluster-rmsd 2.0 \
  --contact-cutoff 4.0 \
  2>&1 | tee "${OUTPUT_DIR}/logs/stage3_console.log"

echo "Stage-3 completed: ${OUTPUT_DIR}"
