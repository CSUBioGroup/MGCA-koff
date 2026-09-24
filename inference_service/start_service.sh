#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/service_config.sh"
if [[ "${HOST}" != "127.0.0.1" && "${HOST}" != "localhost" && "${HOST}" != "::1" && -z "${API_KEY:-}" ]]; then
  echo 'Non-loopback binding requires API_KEY. Use a trusted network/TLS reverse proxy.' >&2
  exit 2
fi
cd "${HERE}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN}" -m uvicorn api:app --host "${HOST}" --port "${PORT}" --workers 1 --limit-concurrency 16 --timeout-keep-alive 15
