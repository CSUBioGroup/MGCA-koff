#!/usr/bin/env python3
"""Fail-fast checks for dependencies and all aligned benchmark inputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_ROOT.parent


def split_paths(dataset: str, split: str):
    for run in range(1, 6):
        if dataset == "KinetX" and split == "warm":
            root = PROJECT_ROOT / "KinetX" / "random_split_mgca_input"
            yield run, root / f"train_run{run}.csv", root / f"val_run{run}.csv", root / f"test_run{run}.csv"
        elif dataset == "KinetX" and split == "drug_cold":
            root = PROJECT_ROOT / "KinetX" / "drug_cold_start_canonical_5fold" / f"fold{run}"
            yield run, root / "train.csv", root / "val.csv", root / "test.csv"
        elif dataset == "KinetX":
            root = PROJECT_ROOT / "KinetX" / "cold_start"
            yield run, root / "train.csv", root / "val.csv", root / "test.csv"
        elif split == "warm":
            root = PROJECT_ROOT / "2773" / "new_folds" / "warm"
            yield run, root / f"train_run{run}.csv", root / f"val_run{run}.csv", root / f"test_run{run}.csv"
        elif split == "drug_cold":
            root = PROJECT_ROOT / "2773" / "new_folds" / "drug-cold"
            yield run, root / f"train_run{run}.csv", root / f"val_run{run}.csv", root / f"test_run{run}.csv"
        else:
            root = Path(os.environ.get(
                "PROTEIN_COLD_2773_DIR",
                PROJECT_ROOT / "2773" / "new_folds" / "target-cold",
            ))
            yield run, root / "train.csv", root / "val.csv", root / "test.csv"


def validate_header(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    lower = {column.lower() for column in header}
    missing = {"fasta", "smiles", "pkoff"} - lower
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["deepdta", "graphdta", "attentiondta"])
    parser.add_argument("--datasets", nargs="+", default=["KinetX", "2773"])
    parser.add_argument("--phase", choices=("tuning", "formal"), default="formal")
    parser.add_argument("--skip-dependencies", action="store_true")
    args = parser.parse_args()

    allowed_models = {"deepdta", "graphdta", "attentiondta"}
    allowed_datasets = {"KinetX", "2773"}
    if not set(args.models) <= allowed_models or not set(args.datasets) <= allowed_datasets:
        raise ValueError("Unsupported model or dataset")

    if not args.skip_dependencies:
        packages = {"numpy", "pandas", "torch", "optuna"}
        if "graphdta" in args.models:
            packages.update({"rdkit", "torch_geometric"})
        missing = sorted(package for package in packages if importlib.util.find_spec(package) is None)
        if missing:
            raise RuntimeError(f"Missing Python dependencies: {', '.join(missing)}")

    checked = set()
    for dataset in args.datasets:
        splits = ("warm",) if args.phase == "tuning" else ("warm", "drug_cold", "protein_cold")
        for split in splits:
            for _, train_csv, val_csv, test_csv in split_paths(dataset, split):
                paths = (train_csv, val_csv) if args.phase == "tuning" else (train_csv, val_csv, test_csv)
                for path in paths:
                    resolved = path.resolve()
                    if resolved in checked:
                        continue
                    if not resolved.is_file():
                        raise FileNotFoundError(resolved)
                    validate_header(resolved)
                    checked.add(resolved)

    if args.phase == "formal" and "2773" in args.datasets:
        protein_root = next(split_paths("2773", "protein_cold"))[1].parent
        provenance = ["clusters.tsv", "split_summary.json", "cluster_assignments.csv"]
        missing_provenance = [name for name in provenance if not (protein_root / name).is_file()]
        if missing_provenance:
            print(
                "WARNING: 2773 protein-cold CSVs are runnable, but similarity-split provenance is incomplete: "
                + ", ".join(missing_provenance),
                file=sys.stderr,
            )
    print(f"Preflight OK: phase={args.phase}, models={args.models}, datasets={args.datasets}, unique CSVs={len(checked)}")


if __name__ == "__main__":
    main()
