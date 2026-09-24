from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(payload: dict, length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def atomic_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_torch_save(torch_module, path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch_module.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_capture(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:
        return None


def environment_snapshot(torch_module=None) -> dict:
    snapshot = {
        "created_at_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "executable": sys.executable,
        "pip_freeze": run_capture([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": run_capture([
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
    }
    if torch_module is not None:
        snapshot.update({
            "torch": torch_module.__version__,
            "cuda_runtime": torch_module.version.cuda,
            "cudnn": (
                torch_module.backends.cudnn.version()
                if torch_module.backends.cudnn.is_available() else None
            ),
            "cuda_available": torch_module.cuda.is_available(),
        })
    for module_name in ("rdkit", "transformers", "numpy", "scipy", "sklearn", "optuna"):
        try:
            module = __import__(module_name)
            snapshot[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            snapshot[module_name] = None
    return snapshot
