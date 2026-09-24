#!/usr/bin/env python3
"""Selection-only BiCoA-Net training with hash-keyed reusable feature caches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SOURCE_FILE = HERE / "models" / "bicoa" / "train_random.py"


def load_source():
    spec = importlib.util.spec_from_file_location("online_bicoa_source", SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bicoa",), required=True)
    parser.add_argument("--dataset", choices=("2773",), required=True)
    parser.add_argument("--split", choices=("warm",), required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--discard-checkpoint-after-eval", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def require_bicoa_gpu_headroom(device: torch.device) -> None:
    """Fail before model construction when another process occupies the GPU."""
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 1024**3
    total_gib = total_bytes / 1024**3
    minimum_gib = float(os.environ.get("BICOA_MIN_FREE_GIB", "18"))
    print(
        f"BiCoA GPU preflight: free={free_gib:.2f} GiB, "
        f"total={total_gib:.2f} GiB, required_free={minimum_gib:.2f} GiB"
    )
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"Insufficient free GPU memory for BiCoA-Net: {free_gib:.2f} GiB free, "
            f"but at least {minimum_gib:.2f} GiB is required. Use nvidia-smi and "
            "stop other GPU jobs before retrying. Do not reduce the published model "
            "architecture merely to share the GPU. Override only after measurement "
            "with BICOA_MIN_FREE_GIB=<GiB>."
        )


def feature_dir(cache_root: Path, csv_path: Path) -> Path:
    return cache_root / sha256_file(csv_path)[:20]


def cache_complete(cache_root: Path, csv_path: Path) -> bool:
    root = feature_dir(cache_root, csv_path)
    required = ("smiles.npy", "protein.npy", "descriptors.npy", "manifest.json")
    if not all((root / name).is_file() for name in required):
        return False
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("csv_sha256") == sha256_file(csv_path)


def build_feature_cache(
    source,
    cache_root: Path,
    csv_path: Path,
    device: torch.device,
    embedders: tuple | None = None,
) -> tuple:
    """Return cached arrays; compute them only when the CSV hash is absent."""
    csv_path = csv_path.resolve()
    frame = source.normalize_input_columns(pd.read_csv(csv_path), str(csv_path))
    root = feature_dir(cache_root, csv_path)
    root.mkdir(parents=True, exist_ok=True)
    if not cache_complete(cache_root, csv_path):
        if embedders is None:
            embedders = (source.MolFormerEmbedder(device=device), source.ESM2Embedder(device=device))
        molformer, esm2 = embedders
        smiles, protein = source.compute_embeddings_for_split(
            frame, molformer, esm2, csv_path.stem
        )
        descriptors = source.compute_descriptors_for_dataset(
            frame, str(root / "descriptors.npy")
        )
        np.save(root / "smiles.npy", smiles)
        np.save(root / "protein.npy", protein)
        atomic_json(
            root / "manifest.json",
            {
                "csv": str(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "rows": len(frame),
                "feature_protocol": "bicoa_original_splitwise_normalization_v1",
                "source_sha256": sha256_file(SOURCE_FILE),
            },
        )
    smiles = np.load(root / "smiles.npy")
    protein = np.load(root / "protein.npy")
    descriptors = np.load(root / "descriptors.npy")
    if not (len(frame) == len(smiles) == len(protein) == len(descriptors)):
        raise RuntimeError(f"Feature-cache row mismatch: {root}")
    return frame, smiles, protein, descriptors


def original_scale_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred, true = np.asarray(pred).reshape(-1), np.asarray(true).reshape(-1)
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    variance = float(np.var(true))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": 1.0 - mse / variance if variance > 0 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if not args.selection_only:
        raise ValueError("This entry is intentionally selection-only")
    if args.run not in range(1, 6):
        raise ValueError("run must be 1..5")
    if min(args.epochs, args.patience, args.batch_size) <= 0:
        raise ValueError("epochs, patience, and batch size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_csv, val_csv = args.train_csv.resolve(), args.val_csv.resolve()
    if not train_csv.is_file() or not val_csv.is_file():
        raise FileNotFoundError((train_csv, val_csv))
    device = resolve_device(args.device)
    require_bicoa_gpu_headroom(device)
    cache_root = (
        args.cache_root
        or Path(
            os.environ.get(
                "BICOA_FEATURE_CACHE", PROJECT_ROOT / "bicoa_cross_tuning_cache"
            )
        )
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    source = load_source()

    missing_cache = not (
        cache_complete(cache_root, train_csv) and cache_complete(cache_root, val_csv)
    )
    embedders = None
    if missing_cache:
        embedders = (source.MolFormerEmbedder(device=device), source.ESM2Embedder(device=device))
    train_frame, train_smiles, train_protein, train_desc = build_feature_cache(
        source, cache_root, train_csv, device, embedders
    )
    val_frame, val_smiles, val_protein, val_desc = build_feature_cache(
        source, cache_root, val_csv, device, embedders
    )
    if embedders is not None:
        del embedders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    desc_mean = train_desc.mean(axis=0, keepdims=True)
    desc_std = train_desc.std(axis=0, keepdims=True)
    desc_std[desc_std == 0] = 1.0
    train_desc = (train_desc - desc_mean) / desc_std
    val_desc = (val_desc - desc_mean) / desc_std
    train_set = source.EnhancedFeatureDataset(
        train_smiles, train_protein, train_desc, train_frame["pkoff"].values
    )
    val_set = source.EnhancedFeatureDataset(
        val_smiles, val_protein, val_desc, val_frame["pkoff"].values
    )
    train_mean = train_set.labels.mean().item()
    train_std = train_set.labels.std().item()
    if not math.isfinite(train_std) or train_std <= 0:
        raise RuntimeError("Invalid training-label standard deviation")
    train_set.labels = (train_set.labels - train_mean) / train_std
    val_set.labels = (val_set.labels - train_mean) / train_std

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ema = source.EMA(model, decay=0.999)
    warmup_epochs = min(15, max(1, args.epochs // 5))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        denominator = max(1, args.epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / denominator
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_mse, best_epoch, best_state, stale = math.inf, 0, None, 0
    for epoch in range(1, args.epochs + 1):
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
        val_metrics = original_scale_metrics(val_pred, val_true)
        scheduler.step()
        print(
            f"epoch={epoch:04d} train_loss={train_loss:.8f} "
            f"val_mse={val_metrics['mse']:.8f}"
        )
        if val_metrics["mse"] < best_mse - 1e-8:
            best_mse, best_epoch, stale = val_metrics["mse"], epoch, 0
            ema.apply_shadow()
            best_state = deepcopy(model.state_dict())
            ema.restore()
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("BiCoA-Net training did not produce a finite validation result")
    model.load_state_dict(best_state)
    _, _, _, _, _, _, _, val_pred, val_true = source.evaluate(
        model, val_loader, criterion, device
    )
    val_metrics = original_scale_metrics(
        val_pred * train_std + train_mean,
        val_true * train_std + train_mean,
    )

    checkpoint = args.output_dir / "best_model.pt"
    if not args.discard_checkpoint_after_eval:
        torch.save(best_state, checkpoint)
    elif checkpoint.exists():
        checkpoint.unlink()
    payload = {
        "model": "bicoa",
        "dataset": "2773",
        "split": "warm",
        "run": args.run,
        "seed": args.seed,
        "selection_only": True,
        "test_metrics": None,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "hyperparameters": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
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
            "test_csv": None,
            "test_sha256": None,
        },
        "feature_cache": {
            "root": str(cache_root),
            "train_key": feature_dir(cache_root, train_csv).name,
            "val_key": feature_dir(cache_root, val_csv).name,
        },
        "val_metrics": val_metrics,
    }
    atomic_json(args.output_dir / "metrics.json", payload)
    complete = args.output_dir / f".complete.tmp.{os.getpid()}"
    complete.write_text("selection complete\n", encoding="utf-8")
    complete.replace(args.output_dir / ".complete")
    print(json.dumps(val_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
