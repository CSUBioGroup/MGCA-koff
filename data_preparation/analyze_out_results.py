from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_ORDER = ["4090_emb_4090_run", "4090_emb_L20_run", "L20_emb_L20_run"]
METHOD_MAP = {
    "outputs_kinetx_random": "Morgan",
    "outputs_kinetx_random_fcfp": "FCFP",
    "outputs_moe_kinetx_random": "MOE",
    "outputs_bicoa_kinetx_random": "BiCoA",
}
METRICS = ["mse", "rmse", "mae", "r2", "pearson", "spearman"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_number(path: Path) -> int:
    for part in path.parts:
        m = re.fullmatch(r"run(\d+)", part)
        if m:
            return int(m.group(1))
    raise ValueError(f"Run folder not found: {path}")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    mse = float(np.mean(err * err))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - np.sum(err * err) / denom) if denom else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(err))),
        "r2": r2,
        "pearson": safe_corr(y, pred),
        "spearman": safe_corr(average_ranks(y), average_ranks(pred)),
    }


def parse_mgca_metrics(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    total = float(next(x.split(":", 1)[1].rstrip("s") for x in lines if x.startswith("Total Duration:")))
    train = float(next(x.split(":", 1)[1].rstrip("s") for x in lines if x.startswith("Train Duration:")))
    header_idx = next(i for i, x in enumerate(lines) if x.startswith("Split\t"))
    header = lines[header_idx].split("\t")
    row = lines[header_idx + 1].split("\t")
    values = dict(zip(header, row))
    result = {m: float(values[f"test_{m}"]) for m in METRICS}
    result.update({"duration_total_s": total, "duration_train_s": train})
    return result


def parse_moe_metrics(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    duration = float(next(x.split(":", 1)[1].rstrip("s") for x in lines if x.startswith("Duration:")))
    header_idx = next(i for i, x in enumerate(lines) if x.startswith("phase\t"))
    header = lines[header_idx].split("\t")
    test = next(x.split("\t") for x in lines[header_idx + 1 :] if x.startswith("test\t"))
    values = dict(zip(header, test))
    result = {m: float(values[m]) for m in METRICS}
    result.update({"duration_total_s": duration, "duration_train_s": None})
    return result


def parse_bicoa_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    result = {m: float(row[m]) for m in METRICS}
    result.update(
        {
            "duration_total_s": None,
            "duration_train_s": None,
            "concordance_index": float(row["concordance_index"]),
            "best_epoch": int(row["best_epoch"]),
            "embedded_run_field": int(row["run"]),
        }
    )
    return result


def load_predictions(path: Path, method: str) -> tuple[np.ndarray, np.ndarray, int | None, int, dict[str, float] | None]:
    if method in {"Morgan", "FCFP"}:
        data = np.loadtxt(path, comments="#")
        return data[:, 0].astype(float), data[:, 1].astype(float), None, len(data), None
    df = pd.read_csv(path)
    if method == "MOE":
        test_rows = df[df["valid"] == 1]
        full_metrics = metric_values(df["pkoff"].to_numpy(float), df["pred"].to_numpy(float))
        return (
            test_rows["pkoff"].to_numpy(float),
            test_rows["pred"].to_numpy(float),
            int(df["valid"].sum()),
            len(df),
            full_metrics,
        )
    return (
        df["true_pkoff"].to_numpy(float),
        df["predicted_pkoff"].to_numpy(float),
        None,
        len(df),
        None,
    )


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def exact_sign_test(diffs: list[float], tol: float = 1e-12) -> dict:
    nz = [d for d in diffs if abs(d) > tol]
    n = len(nz)
    if n == 0:
        return {"n_nonzero": 0, "positive": 0, "negative": 0, "p_two_sided": 1.0}
    positive = sum(d > 0 for d in nz)
    tail = sum(math.comb(n, k) for k in range(0, min(positive, n - positive) + 1)) / (2**n)
    return {
        "n_nonzero": n,
        "positive": positive,
        "negative": n - positive,
        "p_two_sided": min(1.0, 2 * tail),
    }


def clean_float(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def main(out_dir: Path, output_json: Path) -> None:
    records = []
    file_catalog = []
    prediction_arrays: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}

    for config in CONFIG_ORDER:
        config_dir = out_dir / config
        if not config_dir.exists():
            continue
        for method_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
            if method_dir.name not in METHOD_MAP:
                continue
            method = METHOD_MAP[method_dir.name]
            for run_dir in sorted((method_dir / "fixed_split").glob("run*"), key=run_number):
                run = run_number(run_dir)
                metric_files = [p for p in run_dir.iterdir() if p.is_file() and re.match(r"^(metrics_|results_)", p.name)]
                pred_files = [p for p in run_dir.iterdir() if p.is_file() and "prediction" in p.name]
                if len(metric_files) != 1 or len(pred_files) != 1:
                    raise RuntimeError(f"Expected one metric and prediction file in {run_dir}")
                metric_path, pred_path = metric_files[0], pred_files[0]
                if method in {"Morgan", "FCFP"}:
                    parsed = parse_mgca_metrics(metric_path)
                elif method == "MOE":
                    parsed = parse_moe_metrics(metric_path)
                else:
                    parsed = parse_bicoa_metrics(metric_path)
                y, pred, valid_sum, total_rows, full_metrics = load_predictions(pred_path, method)
                recomputed = metric_values(y, pred)
                deltas = {m: recomputed[m] - parsed[m] for m in METRICS}
                record = {
                    "config": config,
                    "method": method,
                    "run": run,
                    **parsed,
                    "n_test": int(len(y)),
                    "n_prediction_rows_total": int(total_rows),
                    "evaluation_coverage": float(len(y) / total_rows),
                    "valid_sum": valid_sum,
                    "metric_file": str(metric_path.relative_to(out_dir.parent)),
                    "prediction_file": str(pred_path.relative_to(out_dir.parent)),
                    "metric_sha256": sha256(metric_path),
                    "prediction_sha256": sha256(pred_path),
                    "max_recompute_abs_delta": max(abs(v) for v in deltas.values()),
                }
                if full_metrics is not None:
                    record.update({f"full_{m}": full_metrics[m] for m in METRICS})
                records.append(record)
                prediction_arrays[(config, method, run)] = (y, pred)
                for kind, path in (("metric", metric_path), ("prediction", pred_path)):
                    file_catalog.append(
                        {
                            "config": config,
                            "method": method,
                            "run": run,
                            "kind": kind,
                            "path": str(path.relative_to(out_dir.parent)),
                            "sha256": sha256(path),
                            "bytes": path.stat().st_size,
                        }
                    )

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["config"], record["method"])].append(record)

    summaries = []
    for (config, method), rows in grouped.items():
        summary = {
            "config": config,
            "method": method,
            "n_runs": len(rows),
            "n_test_min": min(r["n_test"] for r in rows),
            "n_test_max": max(r["n_test"] for r in rows),
            "evaluation_coverage_mean": float(np.mean([r["evaluation_coverage"] for r in rows])),
        }
        for metric in METRICS:
            vals = [r[metric] for r in rows]
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_sd"] = sample_std(vals)
            summary[f"{metric}_min"] = float(np.min(vals))
            summary[f"{metric}_max"] = float(np.max(vals))
        for duration in ["duration_total_s", "duration_train_s"]:
            vals = [r[duration] for r in rows if r.get(duration) is not None]
            summary[f"{duration}_mean"] = float(np.mean(vals)) if vals else None
            summary[f"{duration}_sd"] = sample_std(vals) if vals else None
        if method == "BiCoA":
            summary["concordance_index_mean"] = float(np.mean([r["concordance_index"] for r in rows]))
            summary["best_epoch_mean"] = float(np.mean([r["best_epoch"] for r in rows]))
        if method == "MOE":
            for metric in METRICS:
                values = [r[f"full_{metric}"] for r in rows]
                summary[f"full_{metric}_mean"] = float(np.mean(values))
                summary[f"full_{metric}_sd"] = sample_std(values)
        summaries.append(summary)

    pairwise = []
    for method in ["Morgan", "FCFP", "MOE", "BiCoA"]:
        available = [c for c in CONFIG_ORDER if (c, method) in grouped]
        for i, config_a in enumerate(available):
            for config_b in available[i + 1 :]:
                rows_a = {r["run"]: r for r in grouped[(config_a, method)]}
                rows_b = {r["run"]: r for r in grouped[(config_b, method)]}
                common = sorted(set(rows_a) & set(rows_b))
                for metric in METRICS:
                    diffs = [rows_b[r][metric] - rows_a[r][metric] for r in common]
                    sign = exact_sign_test(diffs)
                    pairwise.append(
                        {
                            "method": method,
                            "config_a": config_a,
                            "config_b": config_b,
                            "metric": metric,
                            "n_pairs": len(diffs),
                            "mean_diff_b_minus_a": float(np.mean(diffs)),
                            "mean_abs_diff": float(np.mean(np.abs(diffs))),
                            "max_abs_diff": float(np.max(np.abs(diffs))),
                            **sign,
                        }
                    )

    prediction_comparisons = []
    for method in ["Morgan", "FCFP", "MOE", "BiCoA"]:
        available = [c for c in CONFIG_ORDER if (c, method) in grouped]
        for i, config_a in enumerate(available):
            for config_b in available[i + 1 :]:
                for run in range(1, 6):
                    if (config_a, method, run) not in prediction_arrays or (config_b, method, run) not in prediction_arrays:
                        continue
                    y_a, pred_a = prediction_arrays[(config_a, method, run)]
                    y_b, pred_b = prediction_arrays[(config_b, method, run)]
                    if len(y_a) != len(y_b):
                        true_diff = float("nan")
                        pred_mae = pred_rmse = pred_corr = float("nan")
                    else:
                        true_diff = float(np.max(np.abs(y_a - y_b)))
                        pred_delta = pred_b - pred_a
                        pred_mae = float(np.mean(np.abs(pred_delta)))
                        pred_rmse = float(np.sqrt(np.mean(pred_delta * pred_delta)))
                        pred_corr = safe_corr(pred_a, pred_b)
                    prediction_comparisons.append(
                        {
                            "method": method,
                            "config_a": config_a,
                            "config_b": config_b,
                            "run": run,
                            "n": int(min(len(y_a), len(y_b))),
                            "max_true_abs_diff": true_diff,
                            "prediction_mae_between": pred_mae,
                            "prediction_rmse_between": pred_rmse,
                            "prediction_pearson_between": pred_corr,
                            "byte_identical": grouped[(config_a, method)][run - 1]["prediction_sha256"]
                            == grouped[(config_b, method)][run - 1]["prediction_sha256"],
                        }
                    )

    duplicate_groups = []
    by_hash = defaultdict(list)
    for item in file_catalog:
        by_hash[(item["kind"], item["sha256"])].append(item)
    for (kind, digest), items in by_hash.items():
        if len(items) > 1:
            duplicate_groups.append(
                {
                    "kind": kind,
                    "sha256": digest,
                    "count": len(items),
                    "paths": [x["path"] for x in items],
                    "configs": sorted({x["config"] for x in items}),
                    "method": items[0]["method"],
                    "run": items[0]["run"],
                }
            )

    # Within-configuration ranking: average rank across the six performance metrics.
    rankings = []
    for config in CONFIG_ORDER:
        available_summaries = [s for s in summaries if s["config"] == config]
        if not available_summaries:
            continue
        rank_lists = defaultdict(list)
        for metric in METRICS:
            reverse = metric in {"r2", "pearson", "spearman"}
            ordered = sorted(available_summaries, key=lambda s: s[f"{metric}_mean"], reverse=reverse)
            for idx, s in enumerate(ordered, start=1):
                rank_lists[s["method"]].append(idx)
        for s in available_summaries:
            rankings.append(
                {
                    "config": config,
                    "method": s["method"],
                    "average_rank_6_metrics": float(np.mean(rank_lists[s["method"]])),
                    "rmse_mean": s["rmse_mean"],
                    "r2_mean": s["r2_mean"],
                    "pearson_mean": s["pearson_mean"],
                    "spearman_mean": s["spearman_mean"],
                }
            )

    output = {
        "metadata": {
            "source": str(out_dir),
            "config_order": CONFIG_ORDER,
            "methods": list(METHOD_MAP.values()),
            "record_count": len(records),
            "metric_file_count": sum(x["kind"] == "metric" for x in file_catalog),
            "prediction_file_count": sum(x["kind"] == "prediction" for x in file_catalog),
            "duplicate_group_count": len(duplicate_groups),
            "notes": [
                "All summary standard deviations use sample SD (ddof=1) across five fixed-split runs.",
                "Pairwise device comparisons are matched by run number; sign-test p-values are exact and low-powered at n=5.",
                "BiCoA result CSVs store run=1 and predictions_run1.csv in every run folder; folder run number is used as the authoritative run identifier.",
            ],
        },
        "records": records,
        "summaries": summaries,
        "rankings": rankings,
        "pairwise": pairwise,
        "prediction_comparisons": prediction_comparisons,
        "duplicates": duplicate_groups,
        "file_catalog": file_catalog,
    }
    output = json.loads(json.dumps(output, default=clean_float, allow_nan=False))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: analyze_out_results.py OUT_DIR OUTPUT_JSON")
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
