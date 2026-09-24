#!/usr/bin/env python3
"""Train tuned MoE on one aligned split and evaluate the untouched test CSV."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from gensim.models import word2vec
from torch import nn
from torch.utils.data import DataLoader

from final_common import (
    atomic_complete,
    atomic_write_json,
    load_best_params,
    regression_metrics,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODEL_ROOT = HERE / "models" / "moe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("moe",), required=True)
    parser.add_argument("--dataset", choices=("KinetX",), required=True)
    parser.add_argument("--split", choices=("warm", "drug_cold", "protein_cold"), required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--best-params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--necessary-files", type=Path, default=None)
    parser.add_argument("--discard-checkpoint-after-eval", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def synchronize(device: torch.device) -> None:
    """Make wall-clock GPU timing include all queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    by_lower = {str(column).strip().lower(): column for column in frame.columns}
    missing = {"fasta", "smiles", "pkoff"} - set(by_lower)
    if missing:
        raise ValueError("%s missing columns: %s" % (path, sorted(missing)))
    selected = frame[[by_lower["fasta"], by_lower["smiles"], by_lower["pkoff"]]].copy()
    selected.columns = ["FASTA", "SMILES", "pkoff"]
    selected.insert(0, "source_row", np.arange(len(frame), dtype=int))
    selected = selected.dropna().reset_index(drop=True)
    selected["FASTA"] = selected["FASTA"].astype(str)
    selected["SMILES"] = selected["SMILES"].astype(str)
    selected["pkoff"] = selected["pkoff"].astype(float)
    return selected


def rows_from_frame(frame: pd.DataFrame):
    return list(frame[["FASTA", "SMILES", "pkoff"]].itertuples(index=False, name=None))


@torch.no_grad()
def evaluate_outputs(model, loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels, validities = [], [], []
    for x1, x2, y, valid in loader:
        predictions.append(model(x1.to(device), x2.to(device)).reshape(-1).cpu().numpy())
        labels.append(y.numpy().reshape(-1))
        validities.append(valid.numpy().reshape(-1))
    return (
        np.concatenate(predictions),
        np.concatenate(labels),
        np.concatenate(validities) > 0.5,
    )


def masked_metrics(pred: np.ndarray, true: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    if not np.any(valid):
        raise RuntimeError("No valid samples remain after MoE feature conversion")
    return regression_metrics(true[valid], pred[valid])


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.run not in range(1, 6):
        raise ValueError("run must be 1..5")
    config = load_best_params(args.best_params, "moe", "KinetX")
    params = config["params"]
    training = config["training"]
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    batch_size = int(params["batch_size"])
    if min(epochs, patience, batch_size) <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.train_csv.resolve(), args.val_csv.resolve(), args.test_csv.resolve()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    train_csv, val_csv, test_csv = paths

    source_root = Path(os.environ.get("MOE_SOURCE_ROOT", PROJECT_ROOT / "moe_base")).resolve()
    necessary = args.necessary_files or Path(
        os.environ.get("MOE_NECESSARY_FILES", source_root / "necessary_files")
    )
    necessary = necessary.resolve()
    model_300 = necessary / "model_300dim.pkl"
    residue_file = necessary / "res_list3.txt"
    framework_file = source_root / "cold_start_framework.py"
    for path in (model_300, residue_file, framework_file, MODEL_ROOT / "model_bimodal_regression_moe.py"):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(MODEL_ROOT))
    sys.path.insert(0, str(source_root))
    from model_bimodal_regression_moe import moe
    from cold_start_framework import KineticsDataset, collate_fn, set_seed

    set_seed(args.seed)
    device = resolve_device(args.device)
    mol2vec = word2vec.Word2Vec.load(str(model_300))
    residues = [line.strip() for line in residue_file.read_text(encoding="utf-8").splitlines()]
    train_frame, val_frame, test_frame = [read_frame(path) for path in paths]
    datasets = [
        KineticsDataset(rows_from_frame(frame), mol2vec, residues, 74, 300, 500)
        for frame in (train_frame, val_frame, test_frame)
    ]
    loader_options = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets[0], shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(datasets[1], shuffle=False, **loader_options)
    test_loader = DataLoader(datasets[2], shuffle=False, **loader_options)

    model = moe(
        num_experts=16,
        drop_r=float(params["dropout"]),
        res_list_num=len(residues),
        x1_dim=300,
        x2_dim=30,
        hid_dim=30,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_mse, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup_duration_sec = time.perf_counter() - run_started
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum, train_valid = 0.0, 0.0
        for x1, x2, y, valid in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y, valid = y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_loss = criterion(model(x1, x2).reshape(-1), y)
            loss = (raw_loss * valid).sum() / (valid.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            train_loss_sum += float((raw_loss * valid).sum().detach().cpu())
            train_valid += float(valid.sum().detach().cpu())
        val_pred, val_true, val_valid = evaluate_outputs(model, val_loader, device)
        val_metrics = masked_metrics(val_pred, val_true, val_valid)
        synchronize(device)
        epoch_duration_sec = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "epoch_duration_sec": epoch_duration_sec,
                "train_mse": train_loss_sum / max(train_valid, 1.0),
                **{"val_" + key: value for key, value in val_metrics.items()},
            }
        )
        print("epoch=%04d val_mse=%.8f" % (epoch, val_metrics["mse"]))
        if val_metrics["mse"] < best_mse - 1e-8:
            best_mse, best_epoch, stale = val_metrics["mse"], epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                break
    synchronize(device)
    training_duration_sec = time.perf_counter() - training_started
    training_peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        if device.type == "cuda"
        else None
    )
    if best_state is None:
        raise RuntimeError("MoE training did not produce a finite validation result")
    evaluation_started = time.perf_counter()
    model.load_state_dict(best_state)
    val_pred, val_true, val_valid = evaluate_outputs(model, val_loader, device)
    test_pred, test_true, test_valid = evaluate_outputs(model, test_loader, device)
    val_metrics = masked_metrics(val_pred, val_true, val_valid)
    test_metrics = masked_metrics(test_pred, test_true, test_valid)
    synchronize(device)
    evaluation_duration_sec = time.perf_counter() - evaluation_started

    prediction_frame = test_frame.loc[test_valid, ["source_row", "FASTA", "SMILES"]].copy()
    prediction_frame["y_true"] = test_true[test_valid]
    prediction_frame["y_pred"] = test_pred[test_valid]
    prediction_frame["error"] = prediction_frame["y_pred"] - prediction_frame["y_true"]
    prediction_frame["abs_error"] = prediction_frame["error"].abs()
    prediction_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    if not np.all(test_valid):
        invalid = test_frame.loc[~test_valid, ["source_row", "FASTA", "SMILES", "pkoff"]].copy()
        invalid.rename(columns={"pkoff": "y_true"}).to_csv(
            args.output_dir / "invalid_test_rows.csv", index=False
        )
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)

    checkpoint = args.output_dir / "best_model.pt"
    if args.discard_checkpoint_after_eval:
        if checkpoint.exists():
            checkpoint.unlink()
    else:
        torch.save(best_state, checkpoint)

    run_to_metrics_duration_sec = time.perf_counter() - run_started
    epochs_ran = len(history)

    payload = {
        "model": "moe",
        "dataset": "KinetX",
        "split": args.split,
        "run": args.run,
        "seed": args.seed,
        "selection_only": False,
        "tuning_config_id": config["config_id"],
        "best_params_file": str(args.best_params.resolve()),
        "best_params_sha256": sha256_file(args.best_params.resolve()),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "timing": {
            "protocol": "synchronized_wall_clock_v1",
            "training_duration_sec": training_duration_sec,
            "epochs_ran": epochs_ran,
            "mean_epoch_duration_sec": training_duration_sec / max(epochs_ran, 1),
            "final_evaluation_duration_sec": evaluation_duration_sec,
            "setup_duration_sec": setup_duration_sec,
            "artifact_and_other_duration_sec": max(
                0.0,
                run_to_metrics_duration_sec
                - setup_duration_sec
                - training_duration_sec
                - evaluation_duration_sec,
            ),
            "run_to_metrics_duration_sec": run_to_metrics_duration_sec,
            "training_peak_gpu_memory_mb": training_peak_gpu_memory_mb,
            "feature_precomputation_included": False,
            "training_duration_includes": [
                "all_train_epochs",
                "per_epoch_validation",
                "early_stopping",
                "best_state_copy",
            ],
            "training_duration_excludes": [
                "data_and_feature_setup",
                "model_initialization",
                "final_validation_and_test_evaluation",
                "artifact_writes",
            ],
        },
        "hyperparameters": {
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "batch_size": batch_size,
            "dropout": float(params["dropout"]),
            "epochs": epochs,
            "patience": patience,
            "optimizer": "Adam",
        },
        "architecture": {"num_experts": 16, "x1_num": 74, "x2_num": 500, "hid_dim": 30},
        "input": {
            "train_csv": str(train_csv),
            "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv),
            "val_sha256": sha256_file(val_csv),
            "test_csv": str(test_csv),
            "test_sha256": sha256_file(test_csv),
        },
        "coverage": {
            "train_total": int(len(train_frame)),
            "val_total": int(len(val_valid)),
            "val_valid": int(np.sum(val_valid)),
            "test_total": int(len(test_valid)),
            "test_valid": int(np.sum(test_valid)),
            "test_fraction": float(np.mean(test_valid)),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    atomic_write_json(args.output_dir / "metrics.json", payload)
    atomic_complete(args.output_dir)
    print(json.dumps(payload["test_metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
