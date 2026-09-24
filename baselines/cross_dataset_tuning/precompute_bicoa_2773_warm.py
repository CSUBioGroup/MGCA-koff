#!/usr/bin/env python3
"""Precompute hash-keyed BiCoA-Net features for all 2773 warm train/val CSVs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from bicoa_train_selection import (
    HERE,
    PROJECT_ROOT,
    build_feature_cache,
    cache_complete,
    load_source,
    resolve_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args()
    data_root = PROJECT_ROOT / "2773" / "new_folds" / "warm"
    paths = [
        data_root / f"{split}_run{run}.csv"
        for run in range(1, 6)
        for split in ("train", "val")
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_root = (
        args.cache_root
        or Path(
            os.environ.get(
                "BICOA_FEATURE_CACHE", PROJECT_ROOT / "bicoa_cross_tuning_cache"
            )
        )
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    missing = [path for path in paths if not cache_complete(cache_root, path)]
    if not missing:
        print(f"BiCoA feature cache already complete: {cache_root}")
        return

    source = load_source()
    device = resolve_device(args.device)
    embedders = (source.MolFormerEmbedder(device=device), source.ESM2Embedder(device=device))
    for index, path in enumerate(missing, start=1):
        print(f"[{index}/{len(missing)}] caching {path}")
        build_feature_cache(source, cache_root, path, device, embedders)
    print(f"BiCoA feature cache complete: {cache_root}")


if __name__ == "__main__":
    main()
