#!/usr/bin/env python3
"""Precompute all BiCoA feature caches needed by the 30 timed BiCoA runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = HERE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def kinetx_paths(project_root: Path):
    paths = []
    warm = project_root / "KinetX" / "random_split_mgca_input"
    for run in range(1, 6):
        paths.extend(
            warm / ("%s_run%d.csv" % (role, run))
            for role in ("train", "val", "test")
        )
    for run in range(1, 6):
        fold = project_root / "KinetX" / "drug_cold_start_canonical_5fold" / ("fold%d" % run)
        paths.extend(fold / (role + ".csv") for role in ("train", "val", "test"))
    protein = project_root / "KinetX" / "cold_start"
    paths.extend(protein / (role + ".csv") for role in ("train", "val", "test"))
    return paths


def dataset_2773_paths(project_root: Path):
    paths = []
    root = project_root / "2773" / "new_folds"
    for protocol in ("warm", "drug-cold"):
        for run in range(1, 6):
            paths.extend(
                root / protocol / ("%s_run%d.csv" % (role, run))
                for role in ("train", "val", "test")
            )
    protein = root / "target-cold"
    paths.extend(protein / (role + ".csv") for role in ("train", "val", "test"))
    return paths


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    cross_root = project_root / "cross_dataset_bayesian_tuning"
    sys.path.insert(0, str(cross_root))
    from bicoa_train_selection import (
        build_feature_cache,
        cache_complete,
        feature_dir,
        load_source,
        resolve_device,
    )

    cache_root = (
        args.cache_root
        or Path(os.environ.get("BICOA_FEATURE_CACHE", project_root / "bicoa_cross_tuning_cache"))
    ).resolve()
    output_dir = args.output_dir.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    labelled = [("KinetX", path) for path in kinetx_paths(project_root)]
    labelled.extend(("2773", path) for path in dataset_2773_paths(project_root))
    unique = []
    seen = set()
    for dataset, path in labelled:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        if resolved not in seen:
            unique.append((dataset, resolved))
            seen.add(resolved)

    before = {path: cache_complete(cache_root, path) for _, path in unique}
    missing = [(dataset, path) for dataset, path in unique if not before[path]]
    print(
        "BiCoA cache preflight: unique_csvs=%d cached=%d missing=%d"
        % (len(unique), len(unique) - len(missing), len(missing))
    )
    source = None
    device = None
    embedders = None
    rows = []
    total_started = time.perf_counter()
    if missing:
        source = load_source()
        device = resolve_device(args.device)
        embedders = (source.MolFormerEmbedder(device=device), source.ESM2Embedder(device=device))
    for index, (dataset, path) in enumerate(unique, start=1):
        cached_before = before[path]
        started = time.perf_counter()
        if not cached_before:
            print("[%d/%d] caching %s" % (index, len(unique), path))
            build_feature_cache(source, cache_root, path, device, embedders)
        duration = time.perf_counter() - started
        rows.append(
            {
                "dataset": dataset,
                "csv": str(path),
                "cache_key": feature_dir(cache_root, path).name,
                "cache_hit_before_run": cached_before,
                "precompute_duration_sec": duration,
            }
        )
    total_duration = time.perf_counter() - total_started

    import pandas as pd

    pd.DataFrame(rows).to_csv(output_dir / "feature_precompute_times.csv", index=False)
    payload = {
        "timing_protocol": "wall_clock_feature_precomputation_v1",
        "cache_root": str(cache_root),
        "device": str(device) if device is not None else args.device,
        "unique_csv_count": len(unique),
        "cache_hits": len(unique) - len(missing),
        "cache_misses": len(missing),
        "total_precompute_duration_sec": total_duration,
        "included_in_training_duration": False,
        "rows": rows,
    }
    (output_dir / "feature_precompute_timing.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BiCoA feature cache complete: %s" % cache_root)
    print("Feature precomputation wall time: %.3f s" % total_duration)


if __name__ == "__main__":
    main()
