#!/usr/bin/env python3
"""Record software and hardware context for reproducible training-time reporting."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return "unavailable: %s" % exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(args.project_root.resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "git_commit": command_output(
            ["git", "-C", str(args.project_root.resolve()), "rev-parse", "HEAD"]
        ),
    }
    try:
        import torch

        payload.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as exc:
        payload["torch_probe_error"] = repr(exc)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Benchmark environment: %s" % args.output.resolve())


if __name__ == "__main__":
    main()
