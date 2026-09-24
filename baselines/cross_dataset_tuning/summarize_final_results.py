#!/usr/bin/env python3
"""Summarize completed tuned final runs as per-run, mean/std, and pooled metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from final_common import METRIC_NAMES, atomic_write_json, regression_metrics


SPLITS = ("warm", "drug_cold", "protein_cold")


def read_run(run_dir: Path) -> Dict:
    if not (run_dir / ".complete").is_file():
        raise FileNotFoundError("Run is incomplete: %s" % run_dir)
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "test_predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Completed run is missing metrics/predictions: %s" % run_dir)
    with metrics_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("selection_only") is not False or not payload.get("test_metrics"):
        raise ValueError("Not a final benchmark metrics file: %s" % metrics_path)
    return payload


def mean_std_row(rows: pd.DataFrame) -> Dict:
    first = rows.iloc[0]
    split = str(first["split"])
    if split == "protein_cold":
        if rows["n_total"].nunique() != 1 or rows["n_valid"].nunique() != 1:
            raise ValueError("Protein-cold repeated seeds must use the same test set size")
        n_total, n_valid = int(first["n_total"]), int(first["n_valid"])
    else:
        n_total, n_valid = int(rows["n_total"].sum()), int(rows["n_valid"].sum())
    output = {
        "model": first["model"],
        "dataset": first["dataset"],
        "split": split,
        "tuning_config_id": first["tuning_config_id"],
        "runs": len(rows),
        "n_valid": n_valid,
        "n_total": n_total,
        "sample_count": "%d/%d" % (n_valid, n_total) if n_valid != n_total else str(n_total),
    }
    for metric in METRIC_NAMES:
        output[metric + "_mean"] = float(rows[metric].mean())
        output[metric + "_std"] = float(rows[metric].std(ddof=1))
    return output


def display_row(summary: Dict) -> Dict:
    output = {
        key: summary[key]
        for key in ("model", "dataset", "split", "tuning_config_id", "runs", "sample_count")
    }
    for metric in METRIC_NAMES:
        output[metric] = "%.4f ± %.4f" % (
            summary[metric + "_mean"],
            summary[metric + "_std"],
        )
    return output


def summarize_split(root: Path) -> None:
    metric_rows: List[Dict] = []
    prediction_frames = []
    identities = set()
    for run in range(1, 6):
        run_dir = root / ("run%d" % run)
        payload = read_run(run_dir)
        if int(payload.get("run", -1)) != run:
            raise ValueError("Run identity mismatch: %s" % run_dir)
        identities.add(
            (
                payload["model"],
                payload["dataset"],
                payload["split"],
                payload["tuning_config_id"],
            )
        )
        coverage = payload["coverage"]
        metric_rows.append(
            {
                "model": payload["model"],
                "dataset": payload["dataset"],
                "split": payload["split"],
                "run": run,
                "seed": payload["seed"],
                "tuning_config_id": payload["tuning_config_id"],
                "best_epoch": payload["best_epoch"],
                "n_valid": coverage["test_valid"],
                "n_total": coverage["test_total"],
                "coverage": coverage["test_fraction"],
                **payload["test_metrics"],
            }
        )
        predictions = pd.read_csv(run_dir / "test_predictions.csv")
        predictions.insert(0, "run", run)
        prediction_frames.append(predictions)
    if len(identities) != 1:
        raise ValueError("Runs have inconsistent identities: %s" % (identities,))
    model, dataset, split, config_id = next(iter(identities))
    if root.name != split:
        raise ValueError("Split directory name mismatch: %s != %s" % (root.name, split))

    per_run = pd.DataFrame(metric_rows)
    summary = mean_std_row(per_run)
    per_run.to_csv(root / "per_run_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(root / "mean_std_metrics.csv", index=False)
    pd.DataFrame([display_row(summary)]).to_csv(root / "mean_std_display.csv", index=False)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    if split == "protein_cold":
        grouped = all_predictions.groupby("source_row", as_index=False).agg(
            y_true=("y_true", "first"),
            y_true_min=("y_true", "min"),
            y_true_max=("y_true", "max"),
            y_pred=("y_pred", "mean"),
            y_pred_std=("y_pred", "std"),
            FASTA=("FASTA", "first"),
            SMILES=("SMILES", "first"),
        )
        y_true_min = grouped.pop("y_true_min")
        y_true_max = grouped.pop("y_true_max")
        if not np.allclose(y_true_min, y_true_max, rtol=0, atol=1e-7):
            raise ValueError("Protein-cold seed predictions disagree on y_true")
        grouped["error"] = grouped["y_pred"] - grouped["y_true"]
        grouped["abs_error"] = grouped["error"].abs()
        grouped.to_csv(root / "seed_ensemble_predictions.csv", index=False)
        aggregate_frame = grouped
        aggregation = "prediction_mean_across_5_seeds"
    else:
        all_predictions.to_csv(root / "oof_predictions.csv", index=False)
        aggregate_frame = all_predictions
        aggregation = "pooled_out_of_fold_predictions"
    aggregate = {
        "model": model,
        "dataset": dataset,
        "split": split,
        "tuning_config_id": config_id,
        "aggregation": aggregation,
        "n": int(len(aggregate_frame)),
        "test_metrics": regression_metrics(aggregate_frame["y_true"], aggregate_frame["y_pred"]),
    }
    atomic_write_json(root / "aggregate_metrics.json", aggregate)
    pd.DataFrame([{**{key: aggregate[key] for key in aggregate if key != "test_metrics"}, **aggregate["test_metrics"]}]).to_csv(
        root / "aggregate_metrics.csv", index=False
    )
    print("Summarized %s/%s/%s: %s" % (model, dataset, split, root))


def summarize_config_root(root: Path) -> None:
    per_run_frames, summary_frames, display_frames = [], [], []
    for split in SPLITS:
        split_root = root / split
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        summarize_split(split_root)
        per_run_frames.append(pd.read_csv(split_root / "per_run_metrics.csv"))
        summary_frames.append(pd.read_csv(split_root / "mean_std_metrics.csv"))
        display_frames.append(pd.read_csv(split_root / "mean_std_display.csv"))
    pd.concat(per_run_frames, ignore_index=True).to_csv(
        root / "all_splits_per_run_metrics.csv", index=False
    )
    pd.concat(summary_frames, ignore_index=True).to_csv(
        root / "all_splits_mean_std_metrics.csv", index=False
    )
    pd.concat(display_frames, ignore_index=True).to_csv(
        root / "all_splits_mean_std_display.csv", index=False
    )
    print("Combined three splits: %s" % root)


def summarize_results_root(root: Path) -> None:
    per_run_files = sorted(root.glob("*/*/*/all_splits_per_run_metrics.csv"))
    summary_files = sorted(root.glob("*/*/*/all_splits_mean_std_metrics.csv"))
    display_files = sorted(root.glob("*/*/*/all_splits_mean_std_display.csv"))
    if not per_run_files or len(per_run_files) != len(summary_files) or len(per_run_files) != len(display_files):
        raise FileNotFoundError("No complete config summaries found under %s" % root)
    pd.concat([pd.read_csv(path) for path in per_run_files], ignore_index=True).to_csv(
        root / "combined_per_run_metrics.csv", index=False
    )
    pd.concat([pd.read_csv(path) for path in summary_files], ignore_index=True).to_csv(
        root / "combined_mean_std_metrics.csv", index=False
    )
    pd.concat([pd.read_csv(path) for path in display_files], ignore_index=True).to_csv(
        root / "combined_mean_std_display.csv", index=False
    )
    print("Combined all completed studies: %s" % root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-dir", type=Path)
    group.add_argument("--config-root", type=Path)
    group.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    if args.experiment_dir:
        summarize_split(args.experiment_dir.resolve())
    elif args.config_root:
        summarize_config_root(args.config_root.resolve())
    else:
        summarize_results_root(args.results_root.resolve())


if __name__ == "__main__":
    main()
