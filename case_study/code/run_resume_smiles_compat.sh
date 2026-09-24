#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./experiment_config.sh
exec "${PYTHON_BIN}" -u resume_case_smiles_compat.py --phase "${PHASE}"
