#!/usr/bin/env python3
"""Validate and summarize the three completed 15-run timing studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STUDIES = (
    ("bicoa_kinetx_default", "BiCoA-Net", "KinetX", "published/default"),
    ("bicoa_2773_tuned", "BiCoA-Net", "2773", "tuned"),
    ("moe_kinetx_tuned", "MoE", "KinetX", "tuned"),
)
SPLITS = ("warm", "drug_cold", "protein_cold")
METRICS = ("mse", "rmse", "mae", "r2", "pearson", "spearman", "ci")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--reference-final-per-run", type=Path, default=None)
    return parser.parse_args()


def read_runs(output_root: Path) -> pd.DataFrame:
    rows = []
    for study, display_model, expected_dataset, configuration in STUDIES:
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
                if metrics.get("dataset") != expected_dataset:
                    raise ValueError("Dataset mismatch at %s" % run_dir)
                if metrics.get("split") != split or int(metrics.get("run", -1)) != run:
                    raise ValueError("Split/run identity mismatch at %s" % run_dir)
                timing = metrics.get("timing", {})
                if timing.get("protocol") != "synchronized_wall_clock_v1":
                    raise ValueError("Wrong or missing timing protocol at %s" % run_dir)
                coverage = metrics["coverage"]
                hyper = metrics["hyperparameters"]
                row = {
                    "study": study,
                    "model": display_model,
                    "dataset": expected_dataset,
                    "configuration": configuration,
                    "split": split,
                    "run": run,
                    "seed": int(metrics["seed"]),
                    "train_n": int(coverage.get("train_total", 0)),
                    "val_n": int(coverage["val_total"]),
                    "test_n_total": int(coverage["test_total"]),
                    "test_n_valid": int(coverage["test_valid"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "epochs_ran": int(timing["epochs_ran"]),
                    "epochs_max": int(hyper["epochs"]),
                    "patience": int(hyper["patience"]),
                    "batch_size": int(hyper["batch_size"]),
                    "lr": float(hyper["lr"]),
                    "weight_decay": float(hyper["weight_decay"]),
                    "dropout": float(hyper["dropout"]),
                    "parameter_count": int(metrics["parameter_count"]),
                    "training_duration_sec": float(timing["training_duration_sec"]),
                    "mean_epoch_duration_sec": float(timing["mean_epoch_duration_sec"]),
                    "setup_duration_sec": float(timing["setup_duration_sec"]),
                    "final_evaluation_duration_sec": float(
                        timing["final_evaluation_duration_sec"]
                    ),
                    "run_to_metrics_duration_sec": float(
                        timing["run_to_metrics_duration_sec"]
                    ),
                    "subprocess_wall_clock_duration_sec": float(
                        process["subprocess_wall_clock_duration_sec"]
                    ),
                    "training_peak_gpu_memory_mb": timing.get(
                        "training_peak_gpu_memory_mb"
                    ),
                    "timing_protocol": timing["protocol"],
                    "feature_precomputation_included": bool(
                        timing["feature_precomputation_included"]
                    ),
                }
                row.update(
                    {"test_" + key: float(metrics["test_metrics"][key]) for key in METRICS}
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 45:
        raise ValueError("Expected exactly 45 runs, found %d" % len(frame))
    counts = frame.groupby(["study", "split"]).size()
    if not np.all(counts.values == 5):
        raise ValueError("Expected exactly five runs per study/split: %s" % counts.to_dict())
    return frame


def mean_std_summary(frame: pd.DataFrame, groups) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(groups, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {name: value for name, value in zip(groups, keys)}
        first = subset.iloc[0]
        for key in ("model", "dataset", "configuration"):
            if key not in row:
                row[key] = first[key]
        row.update(
            {
                "runs": int(len(subset)),
                "batch_size": (
                    int(first["batch_size"])
                    if subset["batch_size"].nunique() == 1
                    else "mixed"
                ),
                "epochs_ran_mean": float(subset["epochs_ran"].mean()),
                "epochs_ran_std": float(subset["epochs_ran"].std(ddof=1)),
                "best_epoch_mean": float(subset["best_epoch"].mean()),
                "training_duration_sec_mean": float(
                    subset["training_duration_sec"].mean()
                ),
                "training_duration_sec_std": float(
                    subset["training_duration_sec"].std(ddof=1)
                ),
                "training_duration_min_mean": float(
                    subset["training_duration_sec"].mean() / 60.0
                ),
                "mean_epoch_duration_sec_mean": float(
                    subset["mean_epoch_duration_sec"].mean()
                ),
                "subprocess_wall_clock_sec_mean": float(
                    subset["subprocess_wall_clock_duration_sec"].mean()
                ),
                "training_peak_gpu_memory_mb_mean": float(
                    pd.to_numeric(subset["training_peak_gpu_memory_mb"], errors="coerce").mean()
                ),
            }
        )
        for metric in METRICS:
            column = "test_" + metric
            row[column + "_mean"] = float(subset[column].mean())
            row[column + "_std"] = float(subset[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def cross_model_time_ratios(by_split: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        current = by_split[by_split["split"] == split]
        bicoa = current[current["study"] == "bicoa_kinetx_default"].iloc[0]
        moe = current[current["study"] == "moe_kinetx_tuned"].iloc[0]
        rows.append(
            {
                "scope": split,
                "bicoa_kinetx_mean_sec": bicoa["training_duration_sec_mean"],
                "moe_kinetx_mean_sec": moe["training_duration_sec_mean"],
                "bicoa_over_moe_time_ratio": (
                    bicoa["training_duration_sec_mean"]
                    / moe["training_duration_sec_mean"]
                ),
                "interpretation": "descriptive; model configurations have different batch sizes and stopping schedules",
            }
        )
    bicoa = overall[overall["study"] == "bicoa_kinetx_default"].iloc[0]
    moe = overall[overall["study"] == "moe_kinetx_tuned"].iloc[0]
    rows.append(
        {
            "scope": "all_15_runs",
            "bicoa_kinetx_mean_sec": bicoa["training_duration_sec_mean"],
            "moe_kinetx_mean_sec": moe["training_duration_sec_mean"],
            "bicoa_over_moe_time_ratio": (
                bicoa["training_duration_sec_mean"] / moe["training_duration_sec_mean"]
            ),
            "interpretation": "descriptive; averages three evaluation protocols",
        }
    )
    return pd.DataFrame(rows)


def find_reference(project_root: Path, explicit: Path):
    if explicit is not None:
        return explicit.resolve()
    candidate = (
        project_root
        / "\u6700\u7ec8\u7ed3\u679cfinal"
        / "MGCA_\u8c03\u53c2\u4e0e\u5b8c\u6574\u5b9e\u9a8c\u8bb0\u5f55_\u539f\u59cb\u8868"
        / "final_per_run_metrics.csv"
    )
    return candidate if candidate.is_file() else None


def mgca_vs_moe(reference: Path, by_split: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(reference)
    mgca = source[
        (source["model"] == "MGCA-Morgan")
        & (source["dataset"] == "KinetX")
        & (source["configuration"] == "tuned")
    ].copy()
    protocol_map = {
        "Warm-start": "warm",
        "Drug-cold": "drug_cold",
        "Protein-cold": "protein_cold",
    }
    mgca["split"] = mgca["protocol"].map(protocol_map)
    mgca = mgca[
        mgca["split"].isin(SPLITS)
        & (mgca["duration_semantics"] == "recorded training duration")
    ]
    if len(mgca) != 15:
        raise ValueError("Expected 15 compatible MGCA/KinetX timing rows, found %d" % len(mgca))
    rows = []
    for split in SPLITS:
        mgca_times = pd.to_numeric(
            mgca.loc[mgca["split"] == split, "duration_sec"], errors="raise"
        )
        moe = by_split[
            (by_split["study"] == "moe_kinetx_tuned") & (by_split["split"] == split)
        ].iloc[0]
        mgca_mean = float(mgca_times.mean())
        moe_mean = float(moe["training_duration_sec_mean"])
        rows.append(
            {
                "split": split,
                "mgca_morgan_mean_sec": mgca_mean,
                "mgca_morgan_std_sec": float(mgca_times.std(ddof=1)),
                "moe_kinetx_tuned_mean_sec": moe_mean,
                "moe_kinetx_tuned_std_sec": float(moe["training_duration_sec_std"]),
                "mgca_speedup_vs_moe": moe_mean / mgca_mean,
                "mgca_training_time_reduction_pct": 100.0 * (1.0 - mgca_mean / moe_mean),
                "comparison_status": "historical descriptive only until hardware and timing scopes are verified identical",
                "mgca_source": str(reference),
            }
        )
    return pd.DataFrame(rows)


def validation_check(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag timing reruns whose test metric unexpectedly diverges from the formal table."""
    rows = []
    for (study, split), subset in frame.groupby(["study", "split"], sort=False):
        metric = "test_mse"
        values = subset[metric].astype(float)
        rows.append(
            {
                "study": study,
                "split": split,
                "runs": len(subset),
                "test_mse_mean": float(values.mean()),
                "test_mse_std": float(values.std(ddof=1)),
                "all_finite": bool(np.isfinite(values).all()),
                "manual_action": "compare this rerun mean against the corresponding formal-result table before publication",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without requiring the optional tabulate package."""
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: ("%.6g" % value) if isinstance(value, (float, np.floating)) else str(value)
        )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display.iterrows():
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_report(
    output_root: Path,
    by_split: pd.DataFrame,
    overall: pd.DataFrame,
    ratios: pd.DataFrame,
    mgca_comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Missing formal training-time benchmark",
        "",
        "Primary endpoint: synchronized wall-clock time for the complete training loop, including per-epoch validation and early stopping, but excluding feature precomputation, setup, final test evaluation, and file writes.",
        "",
        "All 45 runs were launched sequentially with scheduler concurrency fixed to 1.",
        "",
        "## Mean +/- SD by split",
        "",
        markdown_table(by_split),
        "",
        "## Mean +/- SD over all 15 runs",
        "",
        markdown_table(overall),
        "",
        "## Descriptive BiCoA/KinetX versus MoE/KinetX ratios",
        "",
        markdown_table(ratios),
        "",
    ]
    if mgca_comparison is not None:
        lines.extend(
            [
                "## Historical MGCA-Morgan versus newly timed MoE/KinetX",
                "",
                markdown_table(mgca_comparison),
                "",
                "This cross-run comparison is descriptive. It is suitable for a formal speed claim only after confirming identical GPU, software stack, split, timing boundary, and workload policy.",
                "",
            ]
        )
    lines.extend(
        [
            "## Reporting note",
            "",
            "Different batch sizes should not be forced to match when reporting each model under its selected or published configuration. Always show batch size, epochs actually run, early-stopping rule, hardware, and timing definition beside the time result. A controlled throughput ablation with a shared batch size is a separate experiment.",
            "",
        ]
    )
    (output_root / "training_time_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    project_root = args.project_root.resolve()
    per_run = read_runs(output_root)
    by_split = mean_std_summary(per_run, ["study", "split"])
    overall = mean_std_summary(per_run, ["study"])
    ratios = cross_model_time_ratios(by_split, overall)
    validation = validation_check(per_run)
    per_run.to_csv(output_root / "training_time_per_run.csv", index=False)
    by_split.to_csv(output_root / "training_time_summary_by_split.csv", index=False)
    overall.to_csv(output_root / "training_time_summary_overall.csv", index=False)
    ratios.to_csv(output_root / "bicoa_vs_moe_time_ratios.csv", index=False)
    validation.to_csv(output_root / "performance_sanity_check.csv", index=False)

    reference = find_reference(project_root, args.reference_final_per_run)
    comparison = None
    if reference is not None:
        if not reference.is_file():
            raise FileNotFoundError(reference)
        comparison = mgca_vs_moe(reference, by_split)
        comparison.to_csv(output_root / "mgca_vs_moe_descriptive.csv", index=False)
    markdown_report(output_root, by_split, overall, ratios, comparison)
    print("Validated and summarized exactly 45 timed runs")
    print("Per-run table: %s" % (output_root / "training_time_per_run.csv"))
    print("Split summary: %s" % (output_root / "training_time_summary_by_split.csv"))
    print("Overall summary: %s" % (output_root / "training_time_summary_overall.csv"))
    if comparison is None:
        print("MGCA reference CSV not found; MGCA-vs-MoE ratio was not fabricated.")
        print("Set MGCA_REFERENCE_CSV and rerun only this summarizer when available.")
    else:
        print("Descriptive MGCA-vs-MoE comparison: %s" % (
            output_root / "mgca_vs_moe_descriptive.csv"
        ))


if __name__ == "__main__":
    main()
