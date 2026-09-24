#!/usr/bin/env python3
"""Read-only preflight for the two cross-dataset Bayesian tuning studies."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def split_root(model: str) -> Path:
    if model == "moe":
        return PROJECT_ROOT / "KinetX" / "random_split_mgca_input"
    return PROJECT_ROOT / "2773" / "new_folds" / "warm"


def check_csv(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = {str(name).strip().lower() for name in next(csv.reader(handle))}
    missing = {"fasta", "smiles", "pkoff"} - header
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("moe", "bicoa", "both"), default="both")
    parser.add_argument("--skip-dependencies", action="store_true")
    args = parser.parse_args()
    models = ("moe", "bicoa") if args.model == "both" else (args.model,)

    checked = []
    for model in models:
        root = split_root(model)
        for run in range(1, 6):
            for split in ("train", "val"):
                path = root / f"{split}_run{run}.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                check_csv(path)
                checked.append(path)

    required = [HERE / "tune_warm.py"]
    if "moe" in models:
        source_root = Path(os.environ.get("MOE_SOURCE_ROOT", PROJECT_ROOT / "moe_base"))
        necessary = Path(
            os.environ.get("MOE_NECESSARY_FILES", source_root / "necessary_files")
        )
        required.extend(
            [
                HERE / "moe_train_selection.py",
                HERE / "models" / "moe" / "model_bimodal_regression_moe.py",
                source_root / "cold_start_framework.py",
                necessary / "model_300dim.pkl",
                necessary / "res_list3.txt",
            ]
        )
    if "bicoa" in models:
        required.extend(
            [
                HERE / "bicoa_train_selection.py",
                HERE / "precompute_bicoa_2773_warm.py",
                HERE / "models" / "bicoa" / "train_random.py",
            ]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    selection_sources = [HERE / f"{model}_train_selection.py" for model in models]
    for path in selection_sources:
        text = path.read_text(encoding="utf-8")
        if '"test_csv": None' not in text or '"test_sha256": None' not in text:
            raise RuntimeError(f"Selection-only test guard missing: {path}")

    if not args.skip_dependencies:
        packages = {"numpy", "pandas", "torch", "optuna"}
        if "moe" in models:
            packages.update({"gensim", "rdkit"})
        if "bicoa" in models:
            packages.update({"esm", "transformers", "rdkit", "scipy", "sklearn"})
        missing = sorted(name for name in packages if importlib.util.find_spec(name) is None)
        if missing:
            raise RuntimeError("Missing Python packages: " + ", ".join(missing))

    pairs = ["moe/KinetX" if model == "moe" else "bicoa/2773" for model in models]
    print("Cross-dataset tuning preflight OK")
    print("  studies:", pairs)
    print("  warm train/validation CSVs:", len(checked))
    print("  objective: validation MSE; no test CSV is accepted")


if __name__ == "__main__":
    main()
