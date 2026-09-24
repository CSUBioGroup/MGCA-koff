#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/experiment_config.sh"
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/workflow.py" --phase "${PHASE}"
