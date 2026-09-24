#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./experiment_config.sh
exec "$PYTHON_BIN" -u workflow.py --phase "$PHASE"
