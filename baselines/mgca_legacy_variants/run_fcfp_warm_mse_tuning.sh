#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FINGERPRINTS="fcfp"
exec bash "${SCRIPT_DIR}/run_bayesian_tuning.sh" "$@"
