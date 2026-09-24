#!/usr/bin/env python3
"""Selection-only MoE training on an explicit warm train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import word2vec
from torch import nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODEL_ROOT = HERE / "models" / "moe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("moe",), required=True)
    parser.add_argument("--dataset", choices=("KinetX",), required=True)
    parser.add_argument("--split", choices=("warm",), required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--necessary-files", type=Path, default=None)
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


def read_rows(path: Path) -> list[tuple[str, str, float]]:
    frame = pd.read_csv(path)
    by_lower = {str(column).strip().lower(): column for column in frame.columns}
    missing = {"fasta", "smiles", "pkoff"} - set(by_lower)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rows = frame[[by_lower["fasta"], by_lower["smiles"], by_lower["pkoff"]]].copy()
    rows.columns = ["fasta", "smiles", "pkoff"]
    rows = rows.dropna()
    return [
        (str(row.fasta), str(row.smiles), float(row.pkoff))
        for row in rows.itertuples(index=False)
    ]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@torch.no_grad()
def evaluate(model, loader, device, compute_metrics) -> dict[str, float]:
    model.eval()
    predictions, labels, validities = [], [], []
    for x1, x2, y, valid in loader:
        predictions.append(model(x1.to(device), x2.to(device)).reshape(-1).cpu().numpy())
        labels.append(y.numpy())
        validities.append(valid.numpy())
    pred = np.concatenate(predictions)
    true = np.concatenate(labels)
    valid = np.concatenate(validities).reshape(-1) > 0.5
    if not valid.any():
        raise RuntimeError("No valid samples remain after MoE feature conversion")
    return compute_metrics(true[valid], pred[valid])


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
    from cold_start_framework import KineticsDataset, collate_fn, compute_metrics, set_seed

    set_seed(args.seed)
    device = resolve_device(args.device)
    mol2vec = word2vec.Word2Vec.load(str(model_300))
    residues = [line.strip() for line in residue_file.read_text(encoding="utf-8").splitlines()]
    train_rows, val_rows = read_rows(train_csv), read_rows(val_csv)
    train_set = KineticsDataset(train_rows, mol2vec, residues, 74, 300, 500)
    val_set = KineticsDataset(val_rows, mol2vec, residues, 74, 300, 500)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = moe(
        num_experts=16,
        drop_r=args.dropout,
        res_list_num=len(residues),
        x1_dim=300,
        x2_dim=30,
        hid_dim=30,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_mse, best_epoch, best_state, stale = math.inf, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x1, x2, y, valid in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y, valid = y.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_loss = criterion(model(x1, x2).reshape(-1), y)
            loss = (raw_loss * valid).sum() / (valid.sum() + 1e-8)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(model, val_loader, device, compute_metrics)
        val_mse = float(val_metrics["mse"])
        print(f"epoch={epoch:04d} val_mse={val_mse:.8f}")
        if val_mse < best_mse - 1e-8:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("MoE training did not produce a finite validation result")
    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, device, compute_metrics)

    checkpoint = args.output_dir / "best_model.pt"
    if not args.discard_checkpoint_after_eval:
        torch.save(best_state, checkpoint)
    elif checkpoint.exists():
        checkpoint.unlink()

    payload = {
        "model": "moe",
        "dataset": "KinetX",
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
            "optimizer": "Adam",
        },
        "architecture": {"num_experts": 16, "x1_num": 74, "x2_num": 500, "hid_dim": 30},
        "input": {
            "train_csv": str(train_csv),
            "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv),
            "val_sha256": sha256_file(val_csv),
            "test_csv": None,
            "test_sha256": None,
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
