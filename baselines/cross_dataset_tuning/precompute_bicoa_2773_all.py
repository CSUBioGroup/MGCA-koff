#!/usr/bin/env python3
"""Precompute hash-keyed BiCoA features for all final 2773 split CSVs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bicoa_train_selection import (
    PROJECT_ROOT,
    build_feature_cache,
    cache_complete,
    load_source,
    resolve_device,
)


def all_split_paths(project_root: Path):
    fold_root = project_root / "2773" / "new_folds"
    paths = []
    for protocol in ("warm", "drug-cold"):
        for run in range(1, 6):
            for role in ("train", "val", "test"):
                paths.append(fold_root / protocol / ("%s_run%d.csv" % (role, run)))
    for role in ("train", "val", "test"):
        paths.append(fold_root / "target-cold" / (role + ".csv"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    paths = all_split_paths(project_root)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_root = (
        args.cache_root
        or Path(os.environ.get("BICOA_FEATURE_CACHE", project_root / "bicoa_cross_tuning_cache"))
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    missing = [path for path in paths if not cache_complete(cache_root, path)]
    print(
        "BiCoA final cache preflight: total_csvs=%d cached=%d missing=%d"
        % (len(paths), len(paths) - len(missing), len(missing))
    )
    if not missing:
        print("BiCoA final feature cache already complete: %s" % cache_root)
        return

    source = load_source()
    device = resolve_device(args.device)
    embedders = (source.MolFormerEmbedder(device=device), source.ESM2Embedder(device=device))
    for index, path in enumerate(missing, start=1):
        print("[%d/%d] caching %s" % (index, len(missing), path))
        build_feature_cache(source, cache_root, path, device, embedders)
    print("BiCoA final feature cache complete: %s" % cache_root)


if __name__ == "__main__":
    main()
