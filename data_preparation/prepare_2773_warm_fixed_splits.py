#!/usr/bin/env python3
"""Build explicit 7:1:2 warm splits for the 2773-sample dataset."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.source_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    shuffled_indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(shuffled_indices)
    fold_size, remainder = divmod(len(rows), 5)
    test_folds = []
    offset = 0
    for fold_index in range(5):
        size = fold_size + int(fold_index < remainder)
        test_folds.append(shuffled_indices[offset:offset + size])
        offset += size

    flattened = [index for fold in test_folds for index in fold]
    if len(flattened) != len(rows) or set(flattened) != set(range(len(rows))):
        raise ValueError("The five test folds do not partition the source CSV exactly once")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_indices = set(range(len(rows)))
    for run_index in range(5):
        test_indices = set(test_folds[run_index])
        train_val_indices = sorted(all_indices - test_indices)
        val_count = round(len(rows) * 0.10)
        rng = random.Random(args.seed + run_index + 1)
        val_indices = set(rng.sample(train_val_indices, val_count))
        train_indices = set(train_val_indices) - val_indices

        if train_indices & val_indices or train_indices & test_indices or val_indices & test_indices:
            raise RuntimeError(f"Overlap detected while preparing warm run{run_index + 1}")

        split_indices = {
            "train": sorted(train_indices),
            "val": sorted(val_indices),
            "test": sorted(test_indices),
        }
        for split, indices in split_indices.items():
            output_path = args.output_dir / f"{split}_run{run_index + 1}.csv"
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows[index] for index in indices)

        print(
            f"run{run_index + 1}: train={len(train_indices)} "
            f"val={len(val_indices)} test={len(test_indices)}"
        )


if __name__ == "__main__":
    main()
