#!/usr/bin/env python3
"""Train tuned MGCA-Morgan on one aligned KinetX split with controlled timing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CROSS_ROOT = PROJECT_ROOT / "cross_dataset_bayesian_tuning"
MGCA_ENTRY = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "ESM_Morgan_Hybrid_Fusion_nonredundant.py"
MGCA_DEPENDENCY = PROJECT_ROOT / "local" / "ESM_Morgan_Hybrid_Fusion.py"

sys.path.insert(0, str(CROSS_ROOT))
from final_common import (  # noqa: E402
    atomic_complete,
    atomic_write_json,
    regression_metrics,
    sha256_file,
)
from common.tuning_config import canonical_config_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("KinetX",), required=True)
    parser.add_argument(
        "--split", choices=("warm", "drug_cold", "protein_cold"), required=True
    )
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--best-params", type=Path, required=True)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--discard-checkpoint-after-eval", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_corrected_mgca():
    if not MGCA_ENTRY.is_file() or not MGCA_DEPENDENCY.is_file():
        raise FileNotFoundError(
            "MGCA implementation is incomplete: %s / %s" % (MGCA_ENTRY, MGCA_DEPENDENCY)
        )
    spec = importlib.util.spec_from_file_location("mgca_aligned_timing_model", MGCA_ENTRY)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import corrected MGCA entry point: %s" % MGCA_ENTRY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_and_validate_config(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if str(config.get("model", "")).lower() != "mgca":
        raise ValueError("Expected an MGCA configuration: %s" % path)
    if config.get("dataset") != "KinetX" or config.get("fingerprint") != "morgan":
        raise ValueError("Expected tuned MGCA-Morgan/KinetX configuration: %s" % path)
    if config.get("test_accessed_during_selection") is not False:
        raise ValueError("Configuration does not certify test-set isolation")
    if config.get("config_id") != canonical_config_id(config):
        raise ValueError("Frozen configuration hash mismatch: %s" % path)
    params = config.get("params", {})
    training = config.get("training", {})
    required_params = {"lr", "weight_decay", "batch_size", "dropout", "window_size"}
    if required_params - set(params):
        raise ValueError("Configuration is missing parameters: %s" % sorted(required_params - set(params)))
    if {"epochs", "patience"} - set(training):
        raise ValueError("Configuration is missing training controls")
    architecture = config.get("fixed_architecture", {})
    if int(architecture.get("hidden_dim", -1)) != 512:
        raise ValueError("Expected hidden_dim=512 in final MGCA configuration")
    if int(architecture.get("moe_experts", -1)) != 2:
        raise ValueError("Expected corrected v4 MGCA with two final MoE experts")
    if config.get("esm2_window_layout", "legacy_anchors_v1") != "legacy_anchors_v1":
        raise ValueError("Expected final v4 ESM2 window layout legacy_anchors_v1")

    recorded_entry_hash = config.get("mgca_script_sha256")
    recorded_dependency_hash = config.get("model_dependency_script_sha256")
    actual_entry_hash = sha256_file(MGCA_ENTRY)
    actual_dependency_hash = sha256_file(MGCA_DEPENDENCY)
    if recorded_entry_hash and recorded_entry_hash != actual_entry_hash:
        raise ValueError(
            "MGCA architecture script differs from tuned config: %s != %s"
            % (actual_entry_hash, recorded_entry_hash)
        )
    if recorded_dependency_hash and recorded_dependency_hash != actual_dependency_hash:
        raise ValueError(
            "MGCA dependency script differs from tuned config: %s != %s"
            % (actual_dependency_hash, recorded_dependency_hash)
        )
    return config


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    for smiles_batch, fasta_batch, labels_batch in loader:
        smiles_batch = smiles_batch.float().to(device)
        fasta_batch = fasta_batch.float().to(device)
        labels_batch = labels_batch.float().to(device)
        prediction, _ = model(fasta_batch, smiles_batch)
        predictions.append(prediction.reshape(-1).cpu().numpy())
        labels.append(labels_batch.reshape(-1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def timed_evaluate(model, loader, device: torch.device):
    synchronize(device)
    started = time.perf_counter()
    prediction, labels = evaluate(model, loader, device)
    synchronize(device)
    duration = time.perf_counter() - started
    return prediction, labels, duration


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.run not in range(1, 6):
        raise ValueError("run must be 1..5")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    config = load_and_validate_config(args.best_params)
    params = config["params"]
    training = config["training"]
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    batch_size = int(params["batch_size"])
    if min(epochs, patience, batch_size) <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [args.train_csv.resolve(), args.val_csv.resolve(), args.test_csv.resolve()]
    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.esm2_path.resolve().exists():
        raise FileNotFoundError(args.esm2_path.resolve())
    train_csv, val_csv, test_csv = csv_paths

    corrected = load_corrected_mgca()
    legacy = corrected.legacy
    legacy.set_seed(args.seed)
    device = resolve_device(args.device)
    window_size = int(params["window_size"])
    window_layout = config.get("esm2_window_layout", "legacy_anchors_v1")

    row_groups = [legacy.read_labeled_rows(str(path)) for path in csv_paths]
    sizes = [len(rows) for rows in row_groups]
    merged_rows = row_groups[0] + row_groups[1] + row_groups[2]
    cache_path = Path(
        legacy.get_combined_esm_cache_path(
            [str(path) for path in csv_paths],
            window_size=window_size,
            window_layout=window_layout,
        )
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            "Production ESM cache is missing. Run the one-click precompute first: %s"
            % cache_path
        )
    fasta_features, smiles_features, labels, _ = legacy.preprocess_rows(
        merged_rows,
        str(args.esm2_path.resolve()),
        device,
        esm_cache=str(cache_path),
        cache_label="%s/run%d train+val+test" % (args.split, args.run),
        fingerprint_type="morgan",
        window_size=window_size,
        window_layout=window_layout,
    )
    boundaries = np.cumsum([0] + sizes)
    datasets = []
    for index in range(3):
        start, stop = int(boundaries[index]), int(boundaries[index + 1])
        datasets.append(
            legacy.ESM2MorganDataset(
                smiles_features[start:stop], fasta_features[start:stop], labels[start:stop]
            )
        )
    # Keep the original MGCA data-loader behavior. In particular, the shuffle
    # stream uses the global seed set above, just like the formal v4 runs.
    train_loader = DataLoader(
        datasets[0], batch_size=batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        datasets[1], batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        datasets[2], batch_size=batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = corrected.FullRegressionTransformer(
        proj_dim1=2560,
        proj_dim2=2048,
        hidden_dim=int(config["fixed_architecture"]["hidden_dim"]),
        dropout=float(params["dropout"]),
        nums_of_experts=int(config["fixed_architecture"]["protein_experts"]),
        num_heads=int(config["fixed_architecture"]["attention_heads"]),
        moe_num_experts=int(config["fixed_architecture"]["moe_experts"]),
        ablation="no",
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup_duration_sec = time.perf_counter() - run_started
    training_started = time.perf_counter()
    best_mse, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        synchronize(device)
        epoch_started = time.perf_counter()
        model.train()
        loss_sum, sample_count = 0.0, 0
        for smiles_batch, fasta_batch, labels_batch in train_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)
            optimizer.zero_grad()
            prediction, _ = model(fasta_batch, smiles_batch)
            loss = criterion(prediction.reshape(-1), labels_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            current_n = int(labels_batch.numel())
            loss_sum += float(loss.detach().cpu()) * current_n
            sample_count += current_n

        val_prediction, val_labels = evaluate(model, val_loader, device)
        val_metrics = regression_metrics(val_labels, val_prediction)
        synchronize(device)
        epoch_duration_sec = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "epoch_duration_sec": epoch_duration_sec,
                "train_mse": loss_sum / max(sample_count, 1),
                **{"val_" + key: value for key, value in val_metrics.items()},
            }
        )
        print("epoch=%04d val_mse=%.8f" % (epoch, val_metrics["mse"]))
        if val_metrics["mse"] < best_mse:
            best_mse, best_epoch, stale = val_metrics["mse"], epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience:
                print("early_stop epoch=%d stale=%d" % (epoch, stale))
                break
    synchronize(device)
    training_duration_sec = time.perf_counter() - training_started
    training_peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        if device.type == "cuda"
        else None
    )
    if best_state is None:
        raise RuntimeError("MGCA training did not produce a finite validation result")

    model.load_state_dict(best_state)
    train_prediction, train_labels, train_eval_sec = timed_evaluate(model, train_loader, device)
    val_prediction, val_labels, val_eval_sec = timed_evaluate(model, val_loader, device)
    test_prediction, test_labels, test_eval_sec = timed_evaluate(model, test_loader, device)
    train_metrics = regression_metrics(train_labels, train_prediction)
    val_metrics = regression_metrics(val_labels, val_prediction)
    test_metrics = regression_metrics(test_labels, test_prediction)
    evaluation_duration_sec = train_eval_sec + val_eval_sec + test_eval_sec

    artifact_started = time.perf_counter()
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    prediction_frame = pd.DataFrame(
        {
            "source_row": np.arange(len(test_labels), dtype=int),
            "y_true": test_labels,
            "y_pred": test_prediction,
            "error": test_prediction - test_labels,
            "abs_error": np.abs(test_prediction - test_labels),
        }
    )
    prediction_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    checkpoint = args.output_dir / "best_model.pt"
    if args.discard_checkpoint_after_eval:
        if checkpoint.exists():
            checkpoint.unlink()
    else:
        torch.save(best_state, checkpoint)
    artifact_write_duration_sec = time.perf_counter() - artifact_started
    run_to_metrics_duration_sec = time.perf_counter() - run_started
    epochs_ran = len(history)

    payload: Dict[str, object] = {
        "model": "mgca_morgan",
        "dataset": "KinetX",
        "split": args.split,
        "run": args.run,
        "seed": args.seed,
        "selection_only": False,
        "tuning_config_id": config["config_id"],
        "best_params_file": str(args.best_params.resolve()),
        "best_params_sha256": sha256_file(args.best_params.resolve()),
        "model_entry_file": str(MGCA_ENTRY),
        "model_entry_sha256": sha256_file(MGCA_ENTRY),
        "model_dependency_file": str(MGCA_DEPENDENCY),
        "model_dependency_sha256": sha256_file(MGCA_DEPENDENCY),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "timing": {
            "protocol": "synchronized_wall_clock_v1",
            "training_duration_sec": training_duration_sec,
            "epochs_ran": epochs_ran,
            "mean_epoch_duration_sec": training_duration_sec / max(epochs_ran, 1),
            "setup_duration_sec": setup_duration_sec,
            "final_evaluation_duration_sec": evaluation_duration_sec,
            "train_evaluation_duration_sec": train_eval_sec,
            "validation_evaluation_duration_sec": val_eval_sec,
            "test_evaluation_duration_sec": test_eval_sec,
            "artifact_write_duration_sec": artifact_write_duration_sec,
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
                "final_train_validation_test_evaluation",
                "artifact_writes",
            ],
        },
        "hyperparameters": {
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "batch_size": batch_size,
            "dropout": float(params["dropout"]),
            "window_size": window_size,
            "window_layout": window_layout,
            "hidden_dim": int(config["fixed_architecture"]["hidden_dim"]),
            "epochs": epochs,
            "patience": patience,
            "val_freq": 1,
            "optimizer": "AdamW",
        },
        "architecture": config["fixed_architecture"],
        "input": {
            "train_csv": str(train_csv),
            "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv),
            "val_sha256": sha256_file(val_csv),
            "test_csv": str(test_csv),
            "test_sha256": sha256_file(test_csv),
            "esm_cache": str(cache_path.resolve()),
            "esm_cache_sha256": sha256_file(cache_path.resolve()),
        },
        "coverage": {
            "train_total": sizes[0],
            "val_total": sizes[1],
            "val_valid": sizes[1],
            "test_total": sizes[2],
            "test_valid": sizes[2],
            "test_fraction": 1.0,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    atomic_write_json(args.output_dir / "metrics.json", payload)
    atomic_complete(args.output_dir, "controlled MGCA aligned timing complete\n")
    print(json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
