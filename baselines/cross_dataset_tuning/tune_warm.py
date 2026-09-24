#!/usr/bin/env python3
"""Two-stage warm-validation-MSE tuning for cross-dataset MoE and BiCoA-Net."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
COMMON_TUNER = HERE / "common" / "tune_warm.py"


def load_common():
    spec = importlib.util.spec_from_file_location("online_common_tuner", COMMON_TUNER)
    if spec is None or spec.loader is None:
        raise ImportError(COMMON_TUNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_common()
common.BASELINE_ROOT = HERE
common.PROJECT_ROOT = PROJECT_ROOT
common.MODEL_DEFAULTS = {
    "moe": {
        "epochs": 50,
        "patience": 10,
        "published": {
            "lr": 5e-4,
            "weight_decay": 0.0,
            "batch_size": 64,
            "dropout": 0.2,
        },
    },
    "bicoa": {
        "epochs": 250,
        "patience": 50,
        "published": {
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "dropout": 0.15,
        },
    },
}


def suggest_params(trial, model: str) -> dict:
    if model == "moe":
        lr_range = (1e-5, 2e-3)
        batch_sizes = [16, 32, 64, 128]
        dropouts = [0.0, 0.1, 0.2, 0.3]
    else:
        lr_range = (1e-5, 1e-3)
        batch_sizes = [8, 16, 32, 64]
        dropouts = [0.05, 0.1, 0.15, 0.2, 0.3]
    return {
        "lr": trial.suggest_float("lr", *lr_range, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ),
        "batch_size": trial.suggest_categorical("batch_size", batch_sizes),
        "dropout": trial.suggest_categorical("dropout", dropouts),
    }


def search_space_description(model: str) -> dict:
    if model == "moe":
        lr_range, batch_sizes, dropouts = (
            [1e-5, 2e-3],
            [16, 32, 64, 128],
            [0.0, 0.1, 0.2, 0.3],
        )
    else:
        lr_range, batch_sizes, dropouts = (
            [1e-5, 1e-3],
            [8, 16, 32, 64],
            [0.05, 0.1, 0.15, 0.2, 0.3],
        )
    return {
        "lr": {"type": "log_uniform", "range": lr_range},
        "weight_decay": {
            "type": "categorical",
            "values": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        },
        "batch_size": {"type": "categorical", "values": batch_sizes},
        "dropout": {"type": "categorical", "values": dropouts},
    }


def optimizer_name_for_model(model: str) -> str:
    return "Adam" if model == "moe" else "AdamW"


original_parse_args = common.parse_args


def parse_args():
    args = original_parse_args()
    allowed_pair = (args.model == "moe" and args.dataset == "KinetX") or (
        args.model == "bicoa" and args.dataset == "2773"
    )
    if not allowed_pair:
        raise ValueError(
            "Only cross-dataset tuning is allowed: moe/KinetX or bicoa/2773. "
            "Use the original published configuration on moe/2773 and bicoa/KinetX."
        )
    return args


def run_selection(
    *,
    model: str,
    dataset: str,
    run: int,
    params: dict,
    output_dir: Path,
    device: str,
    num_workers: int,
    seed: int,
    epochs: int,
    patience: int,
    keep_checkpoints: bool,
) -> dict:
    train_csv, val_csv = common.warm_paths(dataset, run)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    optimizer = optimizer_name_for_model(model)
    expected = {
        **params,
        "epochs": epochs,
        "patience": patience,
        "optimizer": optimizer,
    }
    expected_input = {
        "train_csv": str(train_csv.resolve()),
        "train_sha256": common.sha256_file(train_csv),
        "val_csv": str(val_csv.resolve()),
        "val_sha256": common.sha256_file(val_csv),
        "test_csv": None,
        "test_sha256": None,
    }
    if (output_dir / ".complete").is_file() and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        identity_matches = (
            metrics.get("model") == model
            and metrics.get("dataset") == dataset
            and metrics.get("split") == "warm"
            and metrics.get("run") == run
            and metrics.get("seed") == seed
            and metrics.get("input") == expected_input
        )
        if (
            not metrics.get("selection_only")
            or not identity_matches
            or not common.params_match(metrics.get("hyperparameters", {}), expected)
        ):
            raise RuntimeError(f"Completed output has incompatible parameters: {output_dir}")
        return metrics

    train_script = HERE / f"{model}_train_selection.py"
    command = [
        sys.executable,
        str(train_script),
        "--model",
        model,
        "--dataset",
        dataset,
        "--split",
        "warm",
        "--run",
        str(run),
        "--seed",
        str(seed),
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--lr",
        str(params["lr"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--batch-size",
        str(params["batch_size"]),
        "--dropout",
        str(params["dropout"]),
        "--selection-only",
    ]
    if not keep_checkpoints:
        command.append("--discard-checkpoint-after-eval")
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(log_lines[-60:])[-6000:]
        raise RuntimeError(
            f"Training failed with exit {completed.returncode}; see {log_path}\n"
            f"--- train.log tail ---\n{log_tail}\n--- end train.log tail ---"
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


common.suggest_params = suggest_params
common.search_space_description = search_space_description
common.optimizer_name_for_model = optimizer_name_for_model
common.parse_args = parse_args
common.run_selection = run_selection


if __name__ == "__main__":
    common.main()
