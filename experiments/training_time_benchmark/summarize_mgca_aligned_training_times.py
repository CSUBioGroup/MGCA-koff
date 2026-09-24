#!/usr/bin/env python3
"""Validate MGCA timing reruns and compare them with the aligned MoE benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("warm", "drug_cold", "protein_cold")
METRICS = ("mse", "rmse", "mae", "r2", "pearson", "spearman", "ci")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_study(output_root: Path, study: str, expected_model: str) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for run in range(1, 6):
            run_dir = output_root / study / split / ("run%d" % run)
            for required in (".complete", "metrics.json", "process_timing.json"):
                if not (run_dir / required).is_file():
                    raise FileNotFoundError("Incomplete timed run: %s" % run_dir)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            process = json.loads(
                (run_dir / "process_timing.json").read_text(encoding="utf-8")
            )
            if metrics.get("model") != expected_model:
                raise ValueError("Model mismatch at %s" % run_dir)
            if metrics.get("dataset") != "KinetX":
                raise ValueError("Dataset mismatch at %s" % run_dir)
            if metrics.get("split") != split or int(metrics.get("run", -1)) != run:
                raise ValueError("Split/run identity mismatch at %s" % run_dir)
            timing = metrics.get("timing", {})
            if timing.get("protocol") != "synchronized_wall_clock_v1":
                raise ValueError("Wrong timing protocol at %s" % run_dir)
            hyper = metrics["hyperparameters"]
            coverage = metrics["coverage"]
            row = {
                "study": study,
                "model": "MGCA-Morgan" if expected_model == "mgca_morgan" else "MoE",
                "dataset": "KinetX",
                "configuration": "tuned",
                "split": split,
                "run": run,
                "seed": int(metrics["seed"]),
                "train_n": int(coverage.get("train_total", 0)),
                "val_n": int(coverage.get("val_total", 0)),
                "test_n_total": int(coverage.get("test_total", 0)),
                "test_n_valid": int(coverage.get("test_valid", 0)),
                "best_epoch": int(metrics["best_epoch"]),
                "epochs_ran": int(timing["epochs_ran"]),
                "epochs_max": int(hyper["epochs"]),
                "patience": int(hyper["patience"]),
                "batch_size": int(hyper["batch_size"]),
                "lr": float(hyper["lr"]),
                "weight_decay": float(hyper["weight_decay"]),
                "dropout": float(hyper["dropout"]),
                "parameter_count": int(metrics["parameter_count"]),
                "trainable_parameter_count": int(
                    metrics.get("trainable_parameter_count", metrics["parameter_count"])
                ),
                "training_duration_sec": float(timing["training_duration_sec"]),
                "mean_epoch_duration_sec": float(timing["mean_epoch_duration_sec"]),
                "setup_duration_sec": float(timing["setup_duration_sec"]),
                "final_evaluation_duration_sec": float(
                    timing["final_evaluation_duration_sec"]
                ),
                "artifact_write_duration_sec": float(
                    timing.get(
                        "artifact_write_duration_sec",
                        timing.get("artifact_and_other_duration_sec", 0.0),
                    )
                ),
                "run_to_metrics_duration_sec": float(timing["run_to_metrics_duration_sec"]),
                "subprocess_wall_clock_duration_sec": float(
                    process["subprocess_wall_clock_duration_sec"]
                ),
                "training_peak_gpu_memory_mb": timing.get("training_peak_gpu_memory_mb"),
                "timing_protocol": timing["protocol"],
                "feature_precomputation_included": bool(
                    timing["feature_precomputation_included"]
                ),
            }
            for phase in ("train", "val", "test"):
                phase_metrics = metrics.get(phase + "_metrics", {})
                for metric in METRICS:
                    row[phase + "_" + metric] = phase_metrics.get(metric, np.nan)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 15:
        raise ValueError("Expected 15 %s rows, found %d" % (study, len(frame)))
    return frame


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = (
        "epochs_ran",
        "best_epoch",
        "training_duration_sec",
        "mean_epoch_duration_sec",
        "setup_duration_sec",
        "final_evaluation_duration_sec",
        "artifact_write_duration_sec",
        "run_to_metrics_duration_sec",
        "subprocess_wall_clock_duration_sec",
        "training_peak_gpu_memory_mb",
    )
    for (study, split), subset in frame.groupby(["study", "split"], sort=False):
        first = subset.iloc[0]
        row = {
            "study": study,
            "model": first["model"],
            "dataset": first["dataset"],
            "configuration": first["configuration"],
            "split": split,
            "runs": len(subset),
            "batch_size": int(first["batch_size"]),
            "parameter_count": int(first["parameter_count"]),
            "trainable_parameter_count": int(first["trainable_parameter_count"]),
        }
        for column in columns:
            values = pd.to_numeric(subset[column], errors="coerce")
            row[column + "_mean"] = float(values.mean())
            row[column + "_std"] = float(values.std(ddof=1))
        for metric in METRICS:
            values = pd.to_numeric(subset["test_" + metric], errors="coerce")
            row["test_" + metric + "_mean"] = float(values.mean())
            row["test_" + metric + "_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ratios(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for run in range(1, 6):
            current = combined[(combined["split"] == split) & (combined["run"] == run)]
            mgca = current[current["study"] == "mgca_kinetx_tuned"].iloc[0]
            moe = current[current["study"] == "moe_kinetx_tuned"].iloc[0]
            rows.append(
                {
                    "split": split,
                    "run": run,
                    "mgca_seed": int(mgca["seed"]),
                    "moe_seed": int(moe["seed"]),
                    "mgca_epochs_ran": int(mgca["epochs_ran"]),
                    "moe_epochs_ran": int(moe["epochs_ran"]),
                    "mgca_training_sec": float(mgca["training_duration_sec"]),
                    "moe_training_sec": float(moe["training_duration_sec"]),
                    "training_speedup_moe_over_mgca": float(
                        moe["training_duration_sec"] / mgca["training_duration_sec"]
                    ),
                    "mgca_sec_per_epoch": float(mgca["mean_epoch_duration_sec"]),
                    "moe_sec_per_epoch": float(moe["mean_epoch_duration_sec"]),
                    "per_epoch_speedup_moe_over_mgca": float(
                        moe["mean_epoch_duration_sec"] / mgca["mean_epoch_duration_sec"]
                    ),
                    "mgca_subprocess_sec": float(
                        mgca["subprocess_wall_clock_duration_sec"]
                    ),
                    "moe_subprocess_sec": float(moe["subprocess_wall_clock_duration_sec"]),
                    "subprocess_speedup_moe_over_mgca": float(
                        moe["subprocess_wall_clock_duration_sec"]
                        / mgca["subprocess_wall_clock_duration_sec"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def ratio_summary(by_split: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        current = by_split[by_split["split"] == split]
        mgca = current[current["study"] == "mgca_kinetx_tuned"].iloc[0]
        moe = current[current["study"] == "moe_kinetx_tuned"].iloc[0]
        pairs = paired[paired["split"] == split]
        rows.append(
            {
                "scope": split,
                "runs_per_model": 5,
                "mgca_batch_size": int(mgca["batch_size"]),
                "moe_batch_size": int(moe["batch_size"]),
                "mgca_epochs_mean": mgca["epochs_ran_mean"],
                "moe_epochs_mean": moe["epochs_ran_mean"],
                "mgca_training_sec_mean": mgca["training_duration_sec_mean"],
                "mgca_training_sec_std": mgca["training_duration_sec_std"],
                "moe_training_sec_mean": moe["training_duration_sec_mean"],
                "moe_training_sec_std": moe["training_duration_sec_std"],
                "ratio_of_mean_training_time_moe_over_mgca": (
                    moe["training_duration_sec_mean"] / mgca["training_duration_sec_mean"]
                ),
                "paired_training_speedup_mean": pairs[
                    "training_speedup_moe_over_mgca"
                ].mean(),
                "paired_training_speedup_std": pairs[
                    "training_speedup_moe_over_mgca"
                ].std(ddof=1),
                "mgca_sec_per_epoch_mean": mgca["mean_epoch_duration_sec_mean"],
                "moe_sec_per_epoch_mean": moe["mean_epoch_duration_sec_mean"],
                "per_epoch_ratio_of_means_moe_over_mgca": (
                    moe["mean_epoch_duration_sec_mean"]
                    / mgca["mean_epoch_duration_sec_mean"]
                ),
                "subprocess_ratio_of_means_moe_over_mgca": (
                    moe["subprocess_wall_clock_duration_sec_mean"]
                    / mgca["subprocess_wall_clock_duration_sec_mean"]
                ),
                "mgca_peak_gpu_memory_mb_mean": mgca[
                    "training_peak_gpu_memory_mb_mean"
                ],
                "moe_peak_gpu_memory_mb_mean": moe[
                    "training_peak_gpu_memory_mb_mean"
                ],
                "status": "controlled aligned timing; native tuned batch sizes and stopping schedules",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: ("%.6g" % value)
            if isinstance(value, (float, np.floating))
            else str(value)
        )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display.iterrows():
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(output_root: Path, ratios: pd.DataFrame, by_split: pd.DataFrame) -> None:
    compact = ratios[
        [
            "scope",
            "mgca_training_sec_mean",
            "mgca_training_sec_std",
            "moe_training_sec_mean",
            "moe_training_sec_std",
            "ratio_of_mean_training_time_moe_over_mgca",
            "mgca_sec_per_epoch_mean",
            "moe_sec_per_epoch_mean",
            "per_epoch_ratio_of_means_moe_over_mgca",
            "subprocess_ratio_of_means_moe_over_mgca",
        ]
    ]
    lines = [
        "# Controlled aligned MGCA-Morgan versus MoE timing",
        "",
        "All runs used the same aligned KinetX split files and were launched serially (concurrency=1). GPU work was synchronized at the timing boundaries. The primary endpoint includes all training epochs, per-epoch validation, early stopping, and best-state copying; feature precomputation, setup, final evaluation, and artifact writes are excluded.",
        "",
        "## Controlled speed ratios",
        "",
        markdown_table(compact),
        "",
        "A ratio greater than 1 means MGCA-Morgan is faster. The total-training ratio reflects each model's selected stopping schedule. The per-epoch ratio compares synchronized mean epoch time. Batch size and actual epochs are reported rather than forced to match.",
        "",
        "The subprocess ratio is a secondary sensitivity analysis, not the primary speed claim: the MGCA timed process performs final train/validation/test evaluation, whereas the existing MoE process performs final validation/test evaluation. The synchronized training-loop and per-epoch endpoints have the aligned boundary.",
        "",
        "MoE traverses every data-loader batch but masks rows that its legacy feature conversion marks invalid. Coverage is retained in the per-run table and should be disclosed when interpreting model efficiency.",
        "",
        "## Full mean and SD table",
        "",
        markdown_table(by_split),
        "",
        "Feature-cache construction remains a separate preprocessing cost and is not included in the primary training endpoint.",
        "",
    ]
    (output_root / "mgca_vs_moe_aligned_training_time_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    mgca = read_study(output_root, "mgca_kinetx_tuned", "mgca_morgan")
    moe = read_study(output_root, "moe_kinetx_tuned", "moe")
    combined = pd.concat([mgca, moe], ignore_index=True)
    by_split = summary(combined)
    paired = paired_ratios(combined)
    ratios = ratio_summary(by_split, paired)

    mgca.to_csv(output_root / "mgca_aligned_training_time_per_run.csv", index=False)
    combined.to_csv(output_root / "mgca_moe_aligned_training_time_per_run.csv", index=False)
    by_split.to_csv(output_root / "mgca_moe_aligned_training_time_summary_by_split.csv", index=False)
    paired.to_csv(output_root / "mgca_vs_moe_aligned_paired_ratios.csv", index=False)
    ratios.to_csv(output_root / "mgca_vs_moe_aligned_speed_ratios.csv", index=False)
    write_report(output_root, ratios, by_split)
    print("Validated and summarized 15 MGCA + 15 existing MoE aligned timing runs")
    print("MGCA per-run table: %s" % (output_root / "mgca_aligned_training_time_per_run.csv"))
    print("Controlled ratios: %s" % (output_root / "mgca_vs_moe_aligned_speed_ratios.csv"))
    print("Report: %s" % (output_root / "mgca_vs_moe_aligned_training_time_report.md"))


if __name__ == "__main__":
    main()
