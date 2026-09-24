#!/usr/bin/env python3
"""Train tuned BiCoA-Net on one aligned split and evaluate its test CSV."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from bicoa_train_selection import (
    PROJECT_ROOT,
    build_feature_cache,
    cache_complete,
    feature_dir,
    load_source,
    require_bicoa_gpu_headroom,
    resolve_device,
)
from final_common import (
    atomic_complete,
    atomic_write_json,
    load_best_params,
    regression_metrics,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bicoa",), required=True)
    parser.add_argument("--dataset", choices=("KinetX", "2773"), required=True)
    parser.add_argument("--split", choices=("warm", "drug_cold", "protein_cold"), required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--best-params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--discard-checkpoint-after-eval", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    """Make wall-clock GPU timing include all queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.run not in range(1, 6):
        raise ValueError("run must be 1..5")
    config = load_best_params(args.best_params, "bicoa", args.dataset)
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
    device = resolve_device(args.device)
    require_bicoa_gpu_headroom(device)
    cache_root = (
        args.cache_root
        or Path(os.environ.get("BICOA_FEATURE_CACHE", PROJECT_ROOT / "bicoa_cross_tuning_cache"))
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    source = load_source()

    missing_cache = [path for path in paths if not cache_complete(cache_root, path)]
    if missing_cache:
        raise RuntimeError(
            "BiCoA final feature cache is incomplete. Run the benchmark feature-cache "
            "precomputation first. "
            "Missing: %s" % [str(path) for path in missing_cache]
        )
    cached = [build_feature_cache(source, cache_root, path, device) for path in paths]
    train_frame, train_smiles, train_protein, train_desc = cached[0]
    val_frame, val_smiles, val_protein, val_desc = cached[1]
    test_frame, test_smiles, test_protein, test_desc = cached[2]

    desc_mean = train_desc.mean(axis=0, keepdims=True)
    desc_std = train_desc.std(axis=0, keepdims=True)
    desc_std[desc_std == 0] = 1.0
    train_desc = (train_desc - desc_mean) / desc_std
    val_desc = (val_desc - desc_mean) / desc_std
    test_desc = (test_desc - desc_mean) / desc_std
    train_set = source.EnhancedFeatureDataset(
        train_smiles, train_protein, train_desc, train_frame["pkoff"].values
    )
    val_set = source.EnhancedFeatureDataset(
        val_smiles, val_protein, val_desc, val_frame["pkoff"].values
    )
    test_set = source.EnhancedFeatureDataset(
        test_smiles, test_protein, test_desc, test_frame["pkoff"].values
    )
    train_mean = train_set.labels.mean().item()
    train_std = train_set.labels.std().item()
    if not math.isfinite(train_std) or train_std <= 0:
        raise RuntimeError("Invalid training-label standard deviation")
    train_set.labels = (train_set.labels - train_mean) / train_std
    val_set.labels = (val_set.labels - train_mean) / train_std
    test_set.labels = (test_set.labels - train_mean) / train_std

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_set, **loader_options)
    test_loader = DataLoader(test_set, **loader_options)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = source.RefinedProteinLigandModel(
        smiles_dim=train_smiles.shape[1],
        protein_dim=train_protein.shape[1],
        descriptor_dim=train_desc.shape[1],
        d_model=768,
        n_blocks=4,
        n_heads=12,
        d_ff=3072,
        dropout=float(params["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    ema = source.EMA(model, decay=0.999)
    warmup_epochs = min(15, max(1, epochs // 5))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        denominator = max(1, epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / denominator
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        train_loss = source.train_epoch_refined(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_mixup=True,
            mixup_alpha=0.2,
            grad_accum_steps=1,
            ema=ema,
        )
        ema.apply_shadow()
        _, _, _, _, _, _, _, val_pred, val_true = source.evaluate(
            model, val_loader, criterion, device
        )
        ema.restore()
        val_pred = val_pred * train_std + train_mean
        val_true = val_true * train_std + train_mean
        val_metrics = regression_metrics(val_true, val_pred)
        scheduler.step()
        synchronize(device)
        epoch_duration_sec = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "epoch_duration_sec": epoch_duration_sec,
                "train_loss": train_loss,
                **{"val_" + key: value for key, value in val_metrics.items()},
            }
        )
        print(
            "epoch=%04d train_loss=%.8f val_mse=%.8f"
            % (epoch, train_loss, val_metrics["mse"])
        )
        if val_metrics["mse"] < best_mse - 1e-8:
            best_mse, best_epoch, stale = val_metrics["mse"], epoch, 0
            ema.apply_shadow()
            best_state = deepcopy(model.state_dict())
            ema.restore()
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
        raise RuntimeError("BiCoA-Net training did not produce a finite validation result")
    evaluation_started = time.perf_counter()
    model.load_state_dict(best_state)
    _, _, _, _, _, _, _, val_pred, val_true = source.evaluate(
        model, val_loader, criterion, device
    )
    _, _, _, _, _, _, _, test_pred, test_true = source.evaluate(
        model, test_loader, criterion, device
    )
    val_pred, val_true = val_pred * train_std + train_mean, val_true * train_std + train_mean
    test_pred, test_true = (
        test_pred * train_std + train_mean,
        test_true * train_std + train_mean,
    )
    val_metrics = regression_metrics(val_true, val_pred)
    test_metrics = regression_metrics(test_true, test_pred)
    synchronize(device)
    evaluation_duration_sec = time.perf_counter() - evaluation_started

    predictions = pd.DataFrame(
        {
            "source_row": np.arange(len(test_frame), dtype=int),
            "FASTA": test_frame["FASTA"].astype(str).values,
            "SMILES": test_frame["smiles"].astype(str).values,
            "y_true": np.asarray(test_true).reshape(-1),
            "y_pred": np.asarray(test_pred).reshape(-1),
        }
    )
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
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
        "model": "bicoa",
        "dataset": args.dataset,
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
                "ema_update_and_best_state_copy",
            ],
            "training_duration_excludes": [
                "feature_precomputation",
                "feature_cache_loading_and_normalization",
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
            "optimizer": "AdamW",
        },
        "architecture": {
            "d_model": 768,
            "n_blocks": 4,
            "n_heads": 12,
            "d_ff": 3072,
            "mixup_alpha": 0.2,
            "ema_decay": 0.999,
        },
        "input": {
            "train_csv": str(train_csv),
            "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv),
            "val_sha256": sha256_file(val_csv),
            "test_csv": str(test_csv),
            "test_sha256": sha256_file(test_csv),
        },
        "feature_cache": {
            "root": str(cache_root),
            "train_key": feature_dir(cache_root, train_csv).name,
            "val_key": feature_dir(cache_root, val_csv).name,
            "test_key": feature_dir(cache_root, test_csv).name,
        },
        "coverage": {
            "train_total": int(len(train_frame)),
            "val_total": int(len(val_true)),
            "val_valid": int(len(val_true)),
            "test_total": int(len(test_true)),
            "test_valid": int(len(test_true)),
            "test_fraction": 1.0,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    atomic_write_json(args.output_dir / "metrics.json", payload)
    atomic_complete(args.output_dir)
    print(json.dumps(payload["test_metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
