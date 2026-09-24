#!/usr/bin/env python3
"""Fail fast if a frozen MGCA config is not the expected warm-MSE selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "baselines" / "common"))
from tuning_config import canonical_config_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="KinetX", choices=("KinetX", "2773"))
    args = parser.parse_args()

    config_path = args.config_dir / "best_params.json"
    shell_path = args.config_dir / "best_params.sh"
    if not config_path.is_file() or not shell_path.is_file():
        raise FileNotFoundError(f"Missing frozen config in {args.config_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("dataset") != args.dataset:
        raise ValueError(f"Dataset mismatch: {config.get('dataset')} != {args.dataset}")
    if config.get("fingerprint") != "morgan":
        raise ValueError("Only the selected MGCA-Morgan configuration is supported")
    if config.get("selection_protocol") != "warm_only_two_stage":
        raise ValueError("Config was not produced by two-stage warm-only selection")
    if config.get("test_accessed_during_selection") is not False:
        raise ValueError("Refusing a configuration that accessed test data")
    if config.get("stage1", {}).get("objective") != "validation_mse":
        raise ValueError("Stage 1 objective is not validation MSE")
    if config.get("stage2", {}).get("selection_metric") != "mean_validation_mse":
        raise ValueError("Stage 2 selection metric is not mean validation MSE")
    if config.get("fixed_architecture", {}).get("moe_experts") != 2:
        raise ValueError("Expected the corrected two-expert final MoE")

    computed_id = canonical_config_id(config)
    if config.get("config_id") != computed_id:
        raise ValueError(
            f"Config hash mismatch: stored={config.get('config_id')} computed={computed_id}"
        )

    for entry in config.get("warm_data_manifest", []):
        for prefix in ("train", "val"):
            path = Path(entry[f"{prefix}_csv"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != entry[f"{prefix}_sha256"]:
                raise ValueError(f"Warm tuning data changed after selection: {path}")

    implementation_files = [
        ("mgca_script", "mgca_script_sha256"),
        ("model_dependency_script", "model_dependency_script_sha256"),
    ]
    for path_key, hash_key in implementation_files:
        path = Path(config[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != config[hash_key]:
            raise ValueError(f"Implementation changed after tuning: {path}")

    shell_values = dict(re.findall(
        r"^MGCA_([A-Z0-9_]+)='([^']*)'$",
        shell_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    ))
    expected = {
        "CONFIG_ID": config["config_id"],
        "LR": str(config["params"]["lr"]),
        "WEIGHT_DECAY": str(config["params"]["weight_decay"]),
        "BATCH_SIZE": str(config["params"]["batch_size"]),
        "DROPOUT": str(config["params"]["dropout"]),
        "WINDOW_SIZE": str(config["params"]["window_size"]),
        "MOE_NUM_EXPERTS": "2",
    }
    mismatches = {
        key: (shell_values.get(key), value)
        for key, value in expected.items()
        if shell_values.get(key) != value
    }
    if mismatches:
        raise ValueError(f"best_params.sh does not match best_params.json: {mismatches}")

    print("Selected MGCA config audit OK")
    print(f"  dataset={args.dataset}; config_id={config['config_id']}")
    print("  objective=warm validation MSE only; test_accessed=false")
    print(
        "  params="
        + ", ".join(f"{key}={value}" for key, value in config["params"].items())
    )


if __name__ == "__main__":
    main()
