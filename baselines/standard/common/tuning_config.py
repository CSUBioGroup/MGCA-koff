"""Canonical hashing and atomic I/O for frozen tuning configurations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def canonical_config_id(config: dict) -> str:
    identity = {
        key: value for key, value in config.items()
        if key not in {"config_id", "generated_at_unix"}
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
