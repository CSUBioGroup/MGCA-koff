#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/config.sh"
mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONUNBUFFERED=1
"${PYTHON_BIN}" "${HERE}/workflow.py" "$@" 2>&1 | tee "${OUTPUT_ROOT}/logs/workflow_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
