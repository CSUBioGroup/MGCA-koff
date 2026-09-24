#!/usr/bin/env python3
"""Train one explicit train/validation/test run for an aligned DTA baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

SMILES_VOCAB = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64,
}
PROTEIN_VOCAB = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8,
    "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16,
    "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("deepdta", "graphdta", "attentiondta"))
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=("KinetX", "2773"))
    parser.add_argument("--split", required=True, choices=("warm", "drug_cold", "protein_cold"))
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=0, help="0 selects the model-specific default")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0, help="0 selects the model-specific default")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection-only", action="store_true",
                        help="Tune using train/validation only; never read or evaluate a test CSV")
    parser.add_argument("--tuning-config-id", default=None)
    parser.add_argument("--discard-checkpoint-after-eval", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def read_split(path: str) -> pd.DataFrame:
    csv_path = Path(path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    lower_to_original = {str(column).lower(): column for column in frame.columns}
    required = {"fasta": "FASTA", "smiles": "SMILES", "pkoff": "pkoff"}
    missing = [key for key in required if key not in lower_to_original]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}; found {list(frame.columns)}")
    selected = frame[[lower_to_original["fasta"], lower_to_original["smiles"], lower_to_original["pkoff"]]]
    if selected.isna().any().any():
        raise ValueError(f"NaN detected in required columns: {csv_path}")
    normalized = pd.DataFrame({
        "FASTA": frame[lower_to_original["fasta"]].astype(str),
        "SMILES": frame[lower_to_original["smiles"]].astype(str),
        "pkoff": pd.to_numeric(frame[lower_to_original["pkoff"]], errors="raise"),
    })
    normalized.insert(0, "source_row", np.arange(len(normalized), dtype=np.int64))
    if len(normalized) == 0:
        raise ValueError(f"Empty split: {csv_path}")
    return normalized


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(text: str, vocab: dict[str, int], max_len: int) -> np.ndarray:
    output = np.zeros(max_len, dtype=np.int64)
    for index, character in enumerate(str(text)[:max_len]):
        output[index] = vocab.get(character, 0)
    return output


class SequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, protein_len: int):
        self.frame = frame.reset_index(drop=True)
        self.protein_len = protein_len

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        smiles = torch.from_numpy(encode(row.SMILES, SMILES_VOCAB, 100))
        protein = torch.from_numpy(encode(row.FASTA, PROTEIN_VOCAB, self.protein_len))
        return smiles, protein, torch.tensor(float(row.pkoff), dtype=torch.float32), int(row.source_row)


def one_hot_unknown(value, allowed):
    if value not in allowed:
        value = allowed[-1]
    return [value == candidate for candidate in allowed]


def atom_features(atom) -> np.ndarray:
    symbols = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
        "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
        "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr",
        "Cr", "Pt", "Hg", "Pb", "Unknown",
    ]
    values = (
        one_hot_unknown(atom.GetSymbol(), symbols)
        + one_hot_unknown(atom.GetDegree(), list(range(11)))
        + one_hot_unknown(atom.GetTotalNumHs(), list(range(11)))
        + one_hot_unknown(atom.GetImplicitValence(), list(range(11)))
        + [atom.GetIsAromatic()]
    )
    features = np.asarray(values, dtype=np.float32)
    return features / features.sum()


def make_graph_dataset(frame: pd.DataFrame, graph_cache: dict):
    try:
        from rdkit import Chem
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("GraphDTA requires rdkit and torch-geometric; see requirements.txt") from exc

    result = []
    for row in frame.itertuples(index=False):
        if row.SMILES not in graph_cache:
            molecule = Chem.MolFromSmiles(row.SMILES)
            if molecule is None:
                raise ValueError(f"RDKit cannot parse SMILES at source row {row.source_row}: {row.SMILES}")
            nodes = np.stack([atom_features(atom) for atom in molecule.GetAtoms()])
            edges = []
            for bond in molecule.GetBonds():
                left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edges.extend(((left, right), (right, left)))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            if not edges:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            graph_cache[row.SMILES] = (torch.tensor(nodes, dtype=torch.float32), edge_index)
        nodes, edge_index = graph_cache[row.SMILES]
        item = Data(x=nodes.clone(), edge_index=edge_index.clone(), y=torch.tensor([float(row.pkoff)]))
        item.target = torch.from_numpy(encode(row.FASTA, PROTEIN_VOCAB, 1000)).unsqueeze(0)
        item.source_row = torch.tensor([int(row.source_row)], dtype=torch.long)
        result.append(item)
    return result


def compute_metrics(y_true, y_pred):
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    error = pred - true
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else 0.0
    pearson = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 and np.std(true) > 0 and np.std(pred) > 0 else 0.0
    true_rank = pd.Series(true).rank(method="average").to_numpy()
    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1]) if len(true) > 1 and np.std(pred_rank) > 0 else 0.0
    return {
        "n": int(len(true)), "mse": mse, "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))), "r2": r2,
        "pearson": pearson, "spearman": spearman,
    }


def build_model(name: str, dropout: float):
    if name == "deepdta":
        from deepdta.model import DeepDTA
        return DeepDTA(dropout=dropout)
    if name == "attentiondta":
        from attentiondta.model import AttentionDTA
        return AttentionDTA(dropout=dropout)
    from graphdta.model import GraphDTA
    return GraphDTA(dropout=dropout)


def make_loaders(args, frames):
    if args.model == "graphdta":
        from torch_geometric.loader import DataLoader as GraphLoader
        cache = {}
        datasets = [make_graph_dataset(frame, cache) for frame in frames]
        return [
            GraphLoader(dataset, batch_size=args.batch_size, shuffle=(index == 0), num_workers=args.num_workers)
            for index, dataset in enumerate(datasets)
        ]
    protein_len = 1200 if args.model == "attentiondta" else 1000
    datasets = [SequenceDataset(frame, protein_len) for frame in frames]
    return [
        DataLoader(dataset, batch_size=args.batch_size, shuffle=(index == 0), num_workers=args.num_workers)
        for index, dataset in enumerate(datasets)
    ]


def unpack_batch(batch, model_name, device):
    if model_name == "graphdta":
        batch = batch.to(device)
        return batch, batch.y.view(-1, 1).float(), batch.source_row.view(-1)
    smiles, protein, labels, rows = batch
    return (smiles.to(device), protein.to(device)), labels.to(device).view(-1, 1), rows


def forward(model, inputs, model_name):
    return model(inputs) if model_name == "graphdta" else model(*inputs)


@torch.no_grad()
def evaluate(model, loader, model_name, device):
    model.eval()
    all_true, all_pred, all_rows = [], [], []
    for batch in loader:
        inputs, labels, rows = unpack_batch(batch, model_name, device)
        predictions = forward(model, inputs, model_name)
        all_true.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
        all_pred.extend(predictions.detach().cpu().numpy().reshape(-1).tolist())
        all_rows.extend(rows.detach().cpu().numpy().reshape(-1).astype(int).tolist())
    return compute_metrics(all_true, all_pred), np.asarray(all_true), np.asarray(all_pred), np.asarray(all_rows)


def main():
    args = parse_args()
    defaults = {
        "deepdta": {"epochs": 100, "lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.1},
        "attentiondta": {"epochs": 100, "lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.1},
        "graphdta": {"epochs": 1000, "lr": 5e-4, "weight_decay": 0.0, "dropout": 0.2},
    }[args.model]
    args.epochs = args.epochs or defaults["epochs"]
    args.lr = args.lr or defaults["lr"]
    args.weight_decay = defaults["weight_decay"] if args.weight_decay is None else args.weight_decay
    args.dropout = defaults["dropout"] if args.dropout is None else args.dropout
    if not args.selection_only and not args.test_csv:
        raise ValueError("--test-csv is required unless --selection-only is used")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        raise ValueError("lr must be positive, weight-decay non-negative, and dropout in [0, 1)")
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [read_split(args.train_csv), read_split(args.val_csv)]
    if not args.selection_only:
        frames.append(read_split(args.test_csv))
    loaders = make_loaders(args, frames)
    train_loader, val_loader = loaders[:2]
    test_loader = loaders[2] if not args.selection_only else None
    model = build_model(args.model, args.dropout).to(device)
    optimizer_class = torch.optim.Adam if args.model == "graphdta" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = nn.MSELoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    best_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.time()
    checkpoint_path = output_dir / "best_model.pt"

    print(f"model={args.model} dataset={args.dataset} split={args.split} run={args.run}")
    print(f"device={device} parameters={parameter_count:,} seed={args.seed}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            inputs, labels, _ = unpack_batch(batch, args.model, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = forward(model, inputs, args.model)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        val_metrics, _, _, _ = evaluate(model, val_loader, args.model, device)
        train_loss = float(np.mean(batch_losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_metrics["mse"]})
        print(f"epoch={epoch:04d} train_mse={train_loss:.6f} val_mse={val_metrics['mse']:.6f}")
        if val_metrics["mse"] < best_mse - args.min_delta:
            best_mse = val_metrics["mse"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "best_epoch": best_epoch, "best_val_mse": best_mse, "args": vars(args),
                "parameter_count": parameter_count,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}")
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_metrics = None
    if not args.selection_only:
        train_metrics, _, _, _ = evaluate(model, train_loader, args.model, device)
    val_metrics, _, _, _ = evaluate(model, val_loader, args.model, device)
    test_metrics = None
    if not args.selection_only:
        test_metrics, true, pred, source_rows = evaluate(model, test_loader, args.model, device)
        test_source = frames[2].set_index("source_row")
        predictions = pd.DataFrame({"source_row": source_rows, "y_true": true, "y_pred": pred})
        predictions["error"] = predictions.y_pred - predictions.y_true
        predictions["abs_error"] = predictions.error.abs()
        predictions["SMILES"] = predictions.source_row.map(test_source.SMILES)
        predictions["FASTA"] = predictions.source_row.map(test_source.FASTA)
        predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    result = {
        "model": args.model, "dataset": args.dataset, "split": args.split, "run": args.run,
        "seed": args.seed, "device": str(device), "parameter_count": parameter_count,
        "selection_only": args.selection_only, "tuning_config_id": args.tuning_config_id,
        "hyperparameters": {
            "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
            "dropout": args.dropout, "epochs": args.epochs, "patience": args.patience,
            "optimizer": optimizer_class.__name__,
        },
        "best_epoch": best_epoch, "duration_sec": time.time() - started,
        "input": {
            "train_csv": str(Path(args.train_csv).resolve()),
            "train_sha256": sha256_file(args.train_csv),
            "val_csv": str(Path(args.val_csv).resolve()),
            "val_sha256": sha256_file(args.val_csv),
            "test_csv": str(Path(args.test_csv).resolve()) if args.test_csv else None,
            "test_sha256": sha256_file(args.test_csv) if args.test_csv else None,
        },
        "train_metrics": train_metrics, "val_metrics": val_metrics, "test_metrics": test_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    flat_metrics = {
        "model": args.model, "dataset": args.dataset, "split": args.split,
        "run": args.run, "seed": args.seed, "best_epoch": best_epoch,
    }
    metric_sets = [("val", val_metrics)]
    if train_metrics is not None:
        metric_sets.insert(0, ("train", train_metrics))
    if test_metrics is not None:
        metric_sets.append(("test", test_metrics))
    for subset, values in metric_sets:
        flat_metrics.update({f"{subset}_{name}": value for name, value in values.items()})
    pd.DataFrame([flat_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    if args.discard_checkpoint_after_eval and checkpoint_path.exists():
        checkpoint_path.unlink()
    (output_dir / ".complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(val_metrics if args.selection_only else test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
