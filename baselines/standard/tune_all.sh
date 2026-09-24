#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a METHOD_LIST <<< "${METHODS:-deepdta graphdta attentiondta}"
for method in "${METHOD_LIST[@]}"; do
  bash "${ROOT}/${method}/tune.sh" "$@"
done
