"""Export the 2773 pickle fold indices as train/test CSV files."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "2773"
SOURCE_CSV = DATASET_DIR / "koff.csv"
EXPECTED_HEADER = ["FASTA", "SMILES", "pkoff"]
MODES = ("pair", "drug", "target")

ALLOWED_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class RestrictedUnpickler(pickle.Unpickler):
    """Only permit the NumPy globals used by these index-array pickles."""

    def find_class(self, module: str, name: str):
        try:
            return ALLOWED_PICKLE_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Forbidden pickle global: {module}.{name}"
            ) from exc


def load_folds(path: Path):
    with path.open("rb") as handle:
        folds = RestrictedUnpickler(handle).load()
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError(f"Expected five folds in {path}, got {type(folds)!r}")
    return folds


def write_and_verify(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    with path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.reader(handle))
    if written != [header, *rows]:
        raise RuntimeError(f"Round-trip verification failed for {path}")


def main() -> None:
    with SOURCE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.reader(handle))
    header, rows = source[0], source[1:]
    if header != EXPECTED_HEADER:
        raise ValueError(f"Unexpected source header: {header}")

    all_indices = set(range(len(rows)))
    manifest: list[tuple[str, int, int, int]] = []

    for mode in MODES:
        mode_dir = DATASET_DIR / "folds" / mode
        folds = load_folds(mode_dir / "unified_folds.pkl")
        test_occurrences: list[int] = []

        for run, fold in enumerate(folds, start=1):
            if not isinstance(fold, tuple) or len(fold) != 2:
                raise ValueError(f"{mode} run {run} is not a (train, test) tuple")

            train_idx = [int(value) for value in fold[0]]
            test_idx = [int(value) for value in fold[1]]
            train_set, test_set = set(train_idx), set(test_idx)

            if len(train_set) != len(train_idx) or len(test_set) != len(test_idx):
                raise ValueError(f"Duplicate index within {mode} run {run}")
            if train_set & test_set:
                raise ValueError(f"Train/test overlap in {mode} run {run}")
            if train_set | test_set != all_indices:
                raise ValueError(f"Incomplete index coverage in {mode} run {run}")

            train_rows = [rows[index] for index in train_idx]
            test_rows = [rows[index] for index in test_idx]
            write_and_verify(mode_dir / f"train_run{run}.csv", header, train_rows)
            write_and_verify(mode_dir / f"test_run{run}.csv", header, test_rows)

            test_occurrences.extend(test_idx)
            manifest.append((mode, run, len(train_rows), len(test_rows)))

        if len(test_occurrences) != len(rows) or set(test_occurrences) != all_indices:
            raise ValueError(f"Five test folds do not partition the dataset for {mode}")

    print("mode,run,train_rows,test_rows")
    for mode, run, train_rows, test_rows in manifest:
        print(f"{mode},{run},{train_rows},{test_rows}")


if __name__ == "__main__":
    main()
