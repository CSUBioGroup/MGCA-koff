#!/usr/bin/env python3
"""Summarize 15 KinetX Full runs using mean/std plus OOF/seed ensemble."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("mse", "rmse", "mae", "r2", "pearson", "spearman")
PROTOCOLS = ("warm", "drug_cold", "protein_cold")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(true_rank, pred_rank)[0, 1])
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual ** 2) / denominator),
        "pearson": pearson,
        "spearman": spearman,
    }


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = list(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No {pattern} in {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_metrics(path: Path) -> dict[str, float | str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.startswith("Split\t"):
            header = line.split("\t")
            values = lines[index + 1].split("\t")
            record = dict(zip(header, values))
            output: dict[str, float | str] = {"split": record["Split"]}
            for phase in ("train", "val", "test"):
                for metric in METRICS:
                    output[f"{phase}_{metric}"] = float(record[f"{phase}_{metric}"])
            return output
    raise ValueError(f"Metrics table not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Summary destination; defaults to --result-root.",
    )
    args = parser.parse_args()
    root = args.result_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    aggregate_rows = []
    for protocol in PROTOCOLS:
        predictions = []
        truths = []
        for run in range(1, 6):
            run_dir = (
                root / "KinetX" / protocol / "MGCA_Morgan" / "fixed_split"
                / f"{protocol}_run{run}"
            )
            metrics_path = latest_file(run_dir, "metrics_*.txt")
            prediction_path = latest_file(run_dir, "test_predictions_*.txt")
            row = {"dataset": "KinetX", "protocol": protocol, "run": run}
            row.update(read_metrics(metrics_path))
            row["metrics_file"] = str(metrics_path)
            row["prediction_file"] = str(prediction_path)
            per_run.append(row)
            array = np.atleast_2d(np.loadtxt(prediction_path, comments="#"))
            truths.append(array[:, 0])
            predictions.append(array[:, 1])

        if protocol in ("warm", "drug_cold"):
            y_true = np.concatenate(truths)
            y_pred = np.concatenate(predictions)
            aggregation = "5-fold OOF concatenation"
        else:
            if not all(np.allclose(truths[0], current) for current in truths[1:]):
                raise ValueError("Protein-cold seeds do not share one fixed test set")
            y_true = truths[0]
            y_pred = np.mean(np.stack(predictions, axis=0), axis=0)
            aggregation = "5-seed prediction ensemble"
        aggregate_rows.append({
            "dataset": "KinetX",
            "protocol": protocol,
            "aggregation": aggregation,
            "n_runs": 5,
            "n_predictions": len(y_true),
            **compute_metrics(y_true, y_pred),
        })

    per_run_path = output_dir / "per_run_metrics.csv"
    with per_run_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)

    summary_rows = []
    for protocol in PROTOCOLS:
        group = [row for row in per_run if row["protocol"] == protocol]
        result: dict[str, object] = {
            "dataset": "KinetX", "protocol": protocol, "n_runs": len(group)
        }
        for metric in METRICS:
            values = [float(row[f"test_{metric}"]) for row in group]
            result[f"test_{metric}_mean"] = statistics.fmean(values)
            result[f"test_{metric}_std"] = statistics.stdev(values)
        summary_rows.append(result)
    mean_std_path = output_dir / "full_mean_std.csv"
    with mean_std_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate_path = output_dir / "full_oof_or_seed_ensemble_metrics.csv"
    with aggregate_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print("KinetX Full summary complete")
    print(f"  per-run: {per_run_path}")
    print(f"  mean/std: {mean_std_path}")
    print(f"  OOF/ensemble: {aggregate_path}")


if __name__ == "__main__":
    main()
