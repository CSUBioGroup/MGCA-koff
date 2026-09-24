#!/usr/bin/env python3
"""Create balanced KinetX drug cold-start train/val/test folds.

The default input reconstructs the 5,446-row cleaned KinetX dataset from the
three run1 random-split CSVs. Rows are grouped by drug before splitting, so a
drug can occur in only one of train, validation, or test within a fold. Across
all folds, every drug is assigned to the test set exactly once.

By default, drugs are grouped by RDKit canonical isomeric SMILES so equivalent
SMILES spellings cannot leak across train, validation, and test. Use
``--drug_key exact`` only to reproduce the legacy exact-string behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_INPUT_CSVS = [
    "KinetX/random_split/train_run1.csv",
    "KinetX/random_split/val_run1.csv",
    "KinetX/random_split/test_run1.csv",
]


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    lower_map = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(
        f"Missing required column. Expected one of {list(candidates)}, got {list(fieldnames)}"
    )


def read_input_csvs(paths: Sequence[Path]) -> Tuple[List[str], List[dict], str, str, str]:
    fieldnames: List[str] | None = None
    rows: List[dict] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            current_fields = list(reader.fieldnames)
            if fieldnames is None:
                fieldnames = current_fields
            elif current_fields != fieldnames:
                raise ValueError(
                    f"CSV headers differ. Expected {fieldnames}, got {current_fields} in {path}"
                )
            rows.extend(dict(row) for row in reader)

    if not fieldnames:
        raise ValueError("No input CSV rows were loaded")

    fasta_col = pick_column(fieldnames, ["FASTA", "fasta", "sequence", "protein_sequence"])
    smiles_col = pick_column(fieldnames, ["smiles", "SMILES"])
    label_col = pick_column(fieldnames, ["pkoff", "pKoff", "label", "y"])

    for row_idx, row in enumerate(rows, start=1):
        fasta = "".join(str(row.get(fasta_col, "")).split()).upper()
        smiles = str(row.get(smiles_col, "")).strip()
        label = str(row.get(label_col, "")).strip()
        if not fasta:
            raise ValueError(f"Empty FASTA in combined input row {row_idx}")
        if not smiles:
            raise ValueError(f"Empty SMILES in combined input row {row_idx}")
        try:
            float(label)
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric {label_col} in combined input row {row_idx}: {label}"
            ) from exc
        row[fasta_col] = fasta
        row[smiles_col] = smiles
        row[label_col] = label

    return fieldnames, rows, fasta_col, smiles_col, label_col


def make_drug_key_function(mode: str):
    if mode == "exact":
        return lambda smiles: smiles.strip()

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "--drug_key canonical requires RDKit. Install it with: "
            "pip install rdkit"
        ) from exc

    def canonicalize(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    return canonicalize


def build_drug_records(
    rows: Sequence[dict],
    fasta_col: str,
    smiles_col: str,
    label_col: str,
    drug_key_mode: str,
) -> List[dict]:
    make_key = make_drug_key_function(drug_key_mode)
    key_to_rows: Dict[str, List[int]] = defaultdict(list)
    key_to_smiles: Dict[str, set] = defaultdict(set)
    key_to_proteins: Dict[str, set] = defaultdict(set)
    key_to_labels: Dict[str, List[float]] = defaultdict(list)

    for row_idx, row in enumerate(rows):
        smiles = row[smiles_col]
        key = make_key(smiles)
        key_to_rows[key].append(row_idx)
        key_to_smiles[key].add(smiles)
        key_to_proteins[key].add(row[fasta_col])
        key_to_labels[key].append(float(row[label_col]))

    records = []
    for key in sorted(key_to_rows):
        labels = key_to_labels[key]
        records.append(
            {
                "drug_key": key,
                "row_indices": key_to_rows[key],
                "sample_count": len(key_to_rows[key]),
                "protein_count": len(key_to_proteins[key]),
                "smiles_variants": len(key_to_smiles[key]),
                "representative_smiles": sorted(key_to_smiles[key])[0],
                "label_mean": statistics.fmean(labels),
                "label_min": min(labels),
                "label_max": max(labels),
            }
        )
    return records


def assign_test_folds(records: Sequence[dict], n_splits: int, seed: int) -> List[List[dict]]:
    if len(records) < n_splits:
        raise ValueError(f"Unique drugs ({len(records)}) must be >= n_splits ({n_splits})")

    rng = random.Random(seed)
    ordered = list(records)
    rng.shuffle(ordered)
    ordered.sort(key=lambda rec: rec["sample_count"], reverse=True)

    folds: List[List[dict]] = [[] for _ in range(n_splits)]
    sample_counts = [0] * n_splits
    for record in ordered:
        fold_idx = min(
            range(n_splits),
            key=lambda idx: (sample_counts[idx], len(folds[idx]), idx),
        )
        folds[fold_idx].append(record)
        sample_counts[fold_idx] += record["sample_count"]
    return folds


def choose_validation_drugs(
    candidates: Sequence[dict], target_samples: int, seed: int
) -> set:
    """Choose whole drug groups with sample count as close as possible to target."""
    rng = random.Random(seed)
    ordered = list(candidates)
    rng.shuffle(ordered)

    reachable = [False] * (target_samples + 1)
    previous_sum = [-1] * (target_samples + 1)
    previous_record = [-1] * (target_samples + 1)
    reachable[0] = True

    for record_idx, record in enumerate(ordered):
        count = record["sample_count"]
        if count > target_samples:
            continue
        for total in range(target_samples, count - 1, -1):
            if not reachable[total] and reachable[total - count]:
                reachable[total] = True
                previous_sum[total] = total - count
                previous_record[total] = record_idx
        if reachable[target_samples]:
            break

    best_total = max(total for total, is_reachable in enumerate(reachable) if is_reachable)
    selected = set()
    total = best_total
    while total > 0:
        record_idx = previous_record[total]
        if record_idx < 0:
            raise RuntimeError("Failed to reconstruct validation drug subset")
        selected.add(ordered[record_idx]["drug_key"])
        total = previous_sum[total]
    return selected


def flatten_indices(records: Sequence[dict]) -> List[int]:
    return sorted(index for record in records for index in record["row_indices"])


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict], indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in indices:
            writer.writerow(rows[index])


def label_stats(rows: Sequence[dict], indices: Sequence[int], label_col: str) -> Tuple[float, float]:
    values = [float(rows[index][label_col]) for index in indices]
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def verify_fold(
    fold_name: str,
    total_rows: int,
    split_indices: Dict[str, Sequence[int]],
    split_drugs: Dict[str, set],
) -> None:
    names = ("train", "val", "test")
    index_sets = {name: set(split_indices[name]) for name in names}
    for name in names:
        if len(index_sets[name]) != len(split_indices[name]):
            raise RuntimeError(f"{fold_name}: duplicate row index within {name}")

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if index_sets[left] & index_sets[right]:
            raise RuntimeError(f"{fold_name}: row overlap between {left} and {right}")
        if split_drugs[left] & split_drugs[right]:
            raise RuntimeError(f"{fold_name}: drug leakage between {left} and {right}")

    expected = set(range(total_rows))
    observed = index_sets["train"] | index_sets["val"] | index_sets["test"]
    if observed != expected:
        raise RuntimeError(
            f"{fold_name}: incomplete row coverage; expected {total_rows}, got {len(observed)}"
        )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create balanced, drug-disjoint KinetX cold-start folds."
    )
    parser.add_argument("--input_csvs", nargs="+", default=DEFAULT_INPUT_CSVS)
    parser.add_argument(
        "--output_dir", default="KinetX/drug_cold_start_canonical_5fold"
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.10,
        help="Validation fraction of the complete dataset in every fold.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Data split seed only.")
    parser.add_argument(
        "--drug_key",
        choices=["exact", "canonical"],
        default="canonical",
        help="Drug grouping key. canonical prevents equivalent-SMILES leakage.",
    )
    parser.add_argument(
        "--flat_output",
        action="store_true",
        help="Write train_runN.csv/val_runN.csv/test_runN.csv directly in output_dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n_splits < 2:
        raise ValueError("--n_splits must be at least 2")
    if not 0.0 < args.val_fraction < 1.0 - 1.0 / args.n_splits:
        raise ValueError("--val_fraction leaves no room for a non-empty training set")

    input_paths = [Path(path) for path in args.input_csvs]
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)

    fieldnames, rows, fasta_col, smiles_col, label_col = read_input_csvs(input_paths)
    records = build_drug_records(
        rows, fasta_col, smiles_col, label_col, args.drug_key
    )
    record_by_key = {record["drug_key"]: record for record in records}
    all_drug_keys = set(record_by_key)
    test_folds = assign_test_folds(records, args.n_splits, args.seed)
    val_target = round(len(rows) * args.val_fraction)

    fold_summaries = []
    test_occurrences = Counter()
    test_fold_by_drug = {}

    for fold_idx, test_records in enumerate(test_folds, start=1):
        test_keys = {record["drug_key"] for record in test_records}
        remaining = [record for record in records if record["drug_key"] not in test_keys]
        val_keys = choose_validation_drugs(
            remaining,
            target_samples=val_target,
            seed=args.seed + fold_idx * 1009,
        )
        train_keys = all_drug_keys - test_keys - val_keys

        split_records = {
            "train": [record_by_key[key] for key in train_keys],
            "val": [record_by_key[key] for key in val_keys],
            "test": test_records,
        }
        split_indices = {
            name: flatten_indices(group_records)
            for name, group_records in split_records.items()
        }
        split_drugs = {
            "train": train_keys,
            "val": val_keys,
            "test": test_keys,
        }
        fold_name = f"fold{fold_idx}"
        verify_fold(fold_name, len(rows), split_indices, split_drugs)

        fold_dir = output_dir if args.flat_output else output_dir / fold_name
        for split_name in ("train", "val", "test"):
            csv_name = (
                f"{split_name}_run{fold_idx}.csv"
                if args.flat_output
                else f"{split_name}.csv"
            )
            write_csv(
                fold_dir / csv_name,
                fieldnames,
                rows,
                split_indices[split_name],
            )

        for key in test_keys:
            test_occurrences[key] += 1
            test_fold_by_drug[key] = fold_idx

        summary_row = {"fold": fold_idx}
        for split_name in ("train", "val", "test"):
            indices = split_indices[split_name]
            mean, sd = label_stats(rows, indices, label_col)
            summary_row[f"{split_name}_samples"] = len(indices)
            summary_row[f"{split_name}_drugs"] = len(split_drugs[split_name])
            summary_row[f"{split_name}_proteins"] = len(
                {rows[index][fasta_col] for index in indices}
            )
            summary_row[f"{split_name}_pkoff_mean"] = f"{mean:.6f}"
            summary_row[f"{split_name}_pkoff_sd"] = f"{sd:.6f}"
        summary_row["drug_overlap_count"] = 0
        fold_summaries.append(summary_row)

    if set(test_occurrences) != all_drug_keys or any(
        count != 1 for count in test_occurrences.values()
    ):
        raise RuntimeError("Test folds do not partition the complete drug set exactly once")

    fold_summary_path = output_dir / "fold_summary.csv"
    with fold_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fold_summaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(fold_summaries)

    drug_summary_path = output_dir / "drug_summary.csv"
    with drug_summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "drug_id",
            "drug_key",
            "representative_smiles",
            "smiles_variants",
            "sample_count",
            "protein_count",
            "pkoff_mean",
            "pkoff_min",
            "pkoff_max",
            "test_fold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for drug_idx, record in enumerate(
            sorted(records, key=lambda rec: rec["drug_key"]), start=1
        ):
            writer.writerow(
                {
                    "drug_id": f"drug_{drug_idx:05d}",
                    "drug_key": record["drug_key"],
                    "representative_smiles": record["representative_smiles"],
                    "smiles_variants": record["smiles_variants"],
                    "sample_count": record["sample_count"],
                    "protein_count": record["protein_count"],
                    "pkoff_mean": f"{record['label_mean']:.6f}",
                    "pkoff_min": f"{record['label_min']:.6f}",
                    "pkoff_max": f"{record['label_max']:.6f}",
                    "test_fold": test_fold_by_drug[record["drug_key"]],
                }
            )

    group_sizes = [record["sample_count"] for record in records]
    raw_smiles_count = len({row[smiles_col] for row in rows})
    alias_records = [record for record in records if record["smiles_variants"] > 1]
    alias_raw_smiles_count = sum(record["smiles_variants"] for record in alias_records)
    alias_sample_count = sum(record["sample_count"] for record in alias_records)
    test_sizes = [row["test_samples"] for row in fold_summaries]
    mean_test = statistics.fmean(test_sizes)
    test_cv = statistics.stdev(test_sizes) / mean_test if len(test_sizes) > 1 else 0.0
    summary = {
        "input_csvs": [str(path) for path in input_paths],
        "rows": len(rows),
        "unique_raw_smiles": raw_smiles_count,
        "unique_drugs": len(records),
        "multi_spelling_drug_groups": len(alias_records),
        "raw_smiles_in_multi_spelling_groups": alias_raw_smiles_count,
        "samples_in_multi_spelling_groups": alias_sample_count,
        "unique_proteins": len({row[fasta_col] for row in rows}),
        "drug_key_mode": args.drug_key,
        "n_splits": args.n_splits,
        "validation_fraction": args.val_fraction,
        "split_seed": args.seed,
        "largest_drug_group_samples": max(group_sizes),
        "largest_drug_group_share": max(group_sizes) / len(rows),
        "singleton_drugs": sum(count == 1 for count in group_sizes),
        "test_sample_cv": test_cv,
        "drug_overlap_count_each_fold": 0,
        "each_drug_tested_exactly_once": True,
        "folds": fold_summaries,
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    summary_path = output_dir / "split_summary.txt"
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("KinetX Drug Cold-Start 5-Fold Summary\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Input rows: {len(rows)}\n")
        handle.write(f"Unique raw SMILES: {raw_smiles_count}\n")
        handle.write(f"Unique drugs: {len(records)}\n")
        handle.write(f"Multi-spelling drug groups: {len(alias_records)}\n")
        handle.write(f"Raw SMILES in multi-spelling groups: {alias_raw_smiles_count}\n")
        handle.write(f"Samples in multi-spelling groups: {alias_sample_count}\n")
        handle.write(f"Unique proteins: {summary['unique_proteins']}\n")
        handle.write(f"Drug key mode: {args.drug_key}\n")
        handle.write(f"Split seed: {args.seed}\n")
        handle.write(f"Singleton drugs: {summary['singleton_drugs']}\n")
        handle.write(
            f"Largest drug group: {max(group_sizes)} samples "
            f"({summary['largest_drug_group_share']:.2%})\n"
        )
        handle.write(f"Test sample CV: {test_cv:.6f}\n\n")
        handle.write("Fold summary:\n")
        for row in fold_summaries:
            handle.write(
                f"  fold{row['fold']}: "
                f"train={row['train_samples']} ({row['train_drugs']} drugs), "
                f"val={row['val_samples']} ({row['val_drugs']} drugs), "
                f"test={row['test_samples']} ({row['test_drugs']} drugs), "
                f"test_pkoff={row['test_pkoff_mean']}+/-{row['test_pkoff_sd']}\n"
            )
        handle.write("\nLeakage checks:\n")
        handle.write("  Drug overlap within every fold: 0\n")
        handle.write("  Every drug appears in test exactly once: PASS\n")

    print("Done.")
    print(f"Rows: {len(rows)}")
    print(f"Unique drugs: {len(records)}")
    print(f"Largest drug group: {max(group_sizes)} samples")
    print(f"Fold summary: {fold_summary_path}")
    print(f"Split summary: {summary_path}")
    print("Drug leakage check: PASS")
    print("Each drug tested exactly once: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
