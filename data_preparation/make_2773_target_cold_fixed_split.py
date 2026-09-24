#!/usr/bin/env python3
"""Create the fixed 2773 target-cold split used by the revised benchmark.

Protein sequences are clustered with the same MMseqs2 thresholds as KinetX:
30% sequence identity and 80% coverage (cov-mode 0). Whole clusters are
assigned to train, validation, or test. The largest cluster is pinned to train
because it contains 44.4% of the 2773 dataset and makes strict five-fold
cross-validation severely imbalanced.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from make_protein_similarity_folds import (
    build_cluster_records,
    flatten_indices,
    invalid_sequence_stats,
    parse_clusters,
    read_input_csv,
    run_mmseqs,
    unique_sequences,
    write_csv,
    write_fasta,
)


KINETX_TRAIN_FRACTION = 3750 / 5446
KINETX_VAL_FRACTION = 738 / 5446


def choose_fixed_split(
    records: Sequence[dict],
    total_samples: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[set, set, set, dict]:
    """Choose disjoint validation/test cluster subsets with sample-count DP."""
    test_fraction = 1.0 - train_fraction - val_fraction
    target_val = round(total_samples * val_fraction)
    target_test = round(total_samples * test_fraction)
    target_train = total_samples - target_val - target_test

    largest = max(records, key=lambda record: record["sample_count"])
    remaining = [record for record in records if record["cluster_id"] != largest["cluster_id"]]
    random.Random(seed).shuffle(remaining)

    # State values are (validation_mask, test_mask). Clusters omitted from both
    # masks are assigned to train together with the pinned largest cluster.
    states: Dict[Tuple[int, int], Tuple[int, int]] = {(0, 0): (0, 0)}
    for record_idx, record in enumerate(remaining):
        count = record["sample_count"]
        bit = 1 << record_idx
        additions = {}
        for (val_count, test_count), (val_mask, test_mask) in list(states.items()):
            if val_count + count <= target_val:
                key = (val_count + count, test_count)
                additions.setdefault(key, (val_mask | bit, test_mask))
            if test_count + count <= target_test:
                key = (val_count, test_count + count)
                additions.setdefault(key, (val_mask, test_mask | bit))
        for key, masks in additions.items():
            states.setdefault(key, masks)

    def score(state: Tuple[int, int]) -> Tuple[int, int, int]:
        val_count, test_count = state
        train_count = total_samples - val_count - test_count
        squared_error = (
            (train_count - target_train) ** 2
            + (val_count - target_val) ** 2
            + (test_count - target_test) ** 2
        )
        return squared_error, abs(test_count - target_test), abs(val_count - target_val)

    best_state = min(states, key=score)
    val_mask, test_mask = states[best_state]
    val_ids = {
        record["cluster_id"]
        for idx, record in enumerate(remaining)
        if val_mask & (1 << idx)
    }
    test_ids = {
        record["cluster_id"]
        for idx, record in enumerate(remaining)
        if test_mask & (1 << idx)
    }
    all_ids = {record["cluster_id"] for record in records}
    train_ids = all_ids - val_ids - test_ids

    if largest["cluster_id"] not in train_ids:
        raise RuntimeError("Largest protein cluster was not assigned to train")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Protein cluster leakage detected")

    targets = {
        "train": target_train,
        "val": target_val,
        "test": target_test,
        "largest_cluster_id": largest["cluster_id"],
        "largest_cluster_samples": largest["sample_count"],
    }
    return train_ids, val_ids, test_ids, targets


def label_stats(rows: Sequence[dict], indices: Sequence[int], label_col: str) -> Tuple[float, float]:
    values = [float(rows[index][label_col]) for index in indices]
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the fixed MMseqs2-controlled 2773 target-cold split."
    )
    parser.add_argument("--input_csv", default="2773/koff.csv")
    parser.add_argument(
        "--output_dir",
        default="2773/new_folds/target-cold",
    )
    parser.add_argument("--identity", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov_mode", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=KINETX_TRAIN_FRACTION)
    parser.add_argument("--val_fraction", type=float, default=KINETX_VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=42, help="Cluster assignment tie-break seed.")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--mmseqs_bin", default="mmseqs")
    parser.add_argument(
        "--clusters_tsv",
        default=None,
        help="Reuse clusters.tsv produced for this exact 2773/koff.csv sequence order.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.val_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError("Train + validation fractions must be less than 1")

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows, fasta_col, smiles_col, label_col = read_input_csv(input_csv)
    sequences, sequence_to_id = unique_sequences(rows, fasta_col)
    id_to_sequence = {protein_id: sequence for sequence, protein_id in sequence_to_id.items()}
    invalid_count, invalid_characters = invalid_sequence_stats(sequences)
    if invalid_count:
        raise ValueError(
            f"Found {invalid_count} invalid protein sequences: {invalid_characters}"
        )

    fasta_path = output_dir / "proteins.fasta"
    write_fasta(sequences, sequence_to_id, fasta_path)
    if args.clusters_tsv:
        clusters_tsv = Path(args.clusters_tsv)
        if not clusters_tsv.exists():
            raise FileNotFoundError(clusters_tsv)
        shutil.copyfile(clusters_tsv, output_dir / "clusters.tsv")
        clusters_tsv = output_dir / "clusters.tsv"
    else:
        clusters_tsv = run_mmseqs(
            fasta_path=fasta_path,
            output_dir=output_dir,
            mmseqs_bin=args.mmseqs_bin,
            identity=args.identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            threads=args.threads,
            overwrite=args.overwrite,
        )

    member_to_cluster = parse_clusters(clusters_tsv, id_to_sequence.keys())
    records = build_cluster_records(
        rows, fasta_col, label_col, sequence_to_id, member_to_cluster
    )
    train_ids, val_ids, test_ids, targets = choose_fixed_split(
        records,
        total_samples=len(rows),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    record_by_id = {record["cluster_id"]: record for record in records}
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    split_indices = {
        name: flatten_indices([record_by_id[cluster_id] for cluster_id in cluster_ids])
        for name, cluster_ids in split_ids.items()
    }
    all_indices = set().union(*(set(indices) for indices in split_indices.values()))
    if all_indices != set(range(len(rows))):
        raise RuntimeError("Split does not cover every source row exactly once")

    split_sequences = {
        name: {rows[index][fasta_col] for index in indices}
        for name, indices in split_indices.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(split_indices[left]) & set(split_indices[right]):
            raise RuntimeError(f"Row overlap between {left} and {right}")
        if split_sequences[left] & split_sequences[right]:
            raise RuntimeError(f"Protein overlap between {left} and {right}")

    for name in ("train", "val", "test"):
        write_csv(output_dir / f"{name}.csv", fieldnames, rows, split_indices[name])

    assignment_path = output_dir / "cluster_assignments.csv"
    with assignment_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cluster_id",
            "split",
            "sample_count",
            "protein_count",
            "label_mean",
            "label_min",
            "label_max",
            "is_largest_cluster",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["sample_count"], reverse=True):
            split = next(name for name, ids in split_ids.items() if record["cluster_id"] in ids)
            writer.writerow(
                {
                    "cluster_id": record["cluster_id"],
                    "split": split,
                    "sample_count": record["sample_count"],
                    "protein_count": record["protein_count"],
                    "label_mean": f"{record['label_mean']:.6f}",
                    "label_min": f"{record['label_min']:.6f}",
                    "label_max": f"{record['label_max']:.6f}",
                    "is_largest_cluster": int(
                        record["cluster_id"] == targets["largest_cluster_id"]
                    ),
                }
            )

    split_summary = {}
    for name in ("train", "val", "test"):
        mean, sd = label_stats(rows, split_indices[name], label_col)
        split_summary[name] = {
            "samples": len(split_indices[name]),
            "clusters": len(split_ids[name]),
            "proteins": len(split_sequences[name]),
            "pkoff_mean": mean,
            "pkoff_sd": sd,
        }

    summary = {
        "input_csv": str(input_csv),
        "rows": len(rows),
        "unique_proteins": len(sequences),
        "identity_threshold": args.identity,
        "coverage_threshold": args.coverage,
        "cov_mode": args.cov_mode,
        "clusters": len(records),
        "split_seed": args.seed,
        "target_samples": {
            "train": targets["train"],
            "val": targets["val"],
            "test": targets["test"],
        },
        "largest_cluster_id": targets["largest_cluster_id"],
        "largest_cluster_samples": targets["largest_cluster_samples"],
        "largest_cluster_pinned_to_train": True,
        "protein_overlap_count_each_pair": 0,
        "cluster_overlap_count_each_pair": 0,
        "splits": split_summary,
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / "split_summary.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("2773 Target-Cold Fixed Split Summary\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Unique proteins: {len(sequences)}\n")
        handle.write(f"Protein clusters: {len(records)}\n")
        handle.write(f"Sequence identity threshold: {args.identity:.2f}\n")
        handle.write(f"Coverage threshold: {args.coverage:.2f}\n")
        handle.write(f"MMseqs cov-mode: {args.cov_mode}\n")
        handle.write(f"Split seed: {args.seed}\n")
        handle.write(
            f"Largest cluster: {targets['largest_cluster_samples']} samples, pinned to train\n\n"
        )
        for name in ("train", "val", "test"):
            item = split_summary[name]
            handle.write(
                f"{name}: samples={item['samples']}, clusters={item['clusters']}, "
                f"proteins={item['proteins']}, pkoff={item['pkoff_mean']:.6f}+/-{item['pkoff_sd']:.6f}\n"
            )
        handle.write("\nLeakage checks:\n")
        handle.write("  Protein overlap between splits: 0\n")
        handle.write("  Protein-cluster overlap between splits: 0\n")

    print("Done.")
    print(f"Clusters: {len(records)}")
    print(f"Train/val/test: {split_summary['train']['samples']}/"
          f"{split_summary['val']['samples']}/{split_summary['test']['samples']}")
    print(f"Summary: {output_dir / 'split_summary.txt'}")
    print("Protein and cluster leakage checks: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
