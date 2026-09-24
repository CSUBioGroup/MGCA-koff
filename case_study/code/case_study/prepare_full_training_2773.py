#!/usr/bin/env python3
"""Prepare the complete 2773 cohort for the four-target MGCA case study."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import prepare_kinetx_full_training as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--exclusions-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-rows", type=int, default=2773)
    parser.add_argument("--expected-output-rows", type=int, default=2773)
    parser.add_argument("--expected-case-targets", type=int, default=4)
    return parser.parse_args()


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["FASTA"], row["SMILES"], row["pkoff"]


def overlap_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    source_paths = [args.train_csv, args.val_csv, args.test_csv]
    for path in [*source_paths, args.reference_csv, args.case_csv]:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_roles = ("train", "validation", "test")
    source_rows = [common.read_training_rows(path) for path in source_paths]
    tagged_rows = [
        (role, row)
        for role, split_rows in zip(source_roles, source_rows)
        for row in split_rows
    ]
    if len(tagged_rows) != args.expected_source_rows:
        raise RuntimeError(
            f"Expected {args.expected_source_rows} split rows, found {len(tagged_rows)}"
        )

    reference_rows = common.read_training_rows(args.reference_csv)
    if len(reference_rows) != args.expected_source_rows:
        raise RuntimeError(
            f"Expected {args.expected_source_rows} reference rows, "
            f"found {len(reference_rows)}"
        )
    split_counter = Counter(row_key(row) for _, row in tagged_rows)
    reference_counter = Counter(row_key(row) for row in reference_rows)
    if split_counter != reference_counter:
        missing = sum((reference_counter - split_counter).values())
        extra = sum((split_counter - reference_counter).values())
        raise RuntimeError(
            "Warm-run-1 split union does not reproduce 2773/koff.csv exactly: "
            f"missing={missing}, extra={extra}"
        )

    case_targets, case_pairs = common.read_case_targets(args.case_csv)
    if len(case_targets) != args.expected_case_targets:
        raise RuntimeError(
            f"Expected {args.expected_case_targets} case FASTA sequences, "
            f"found {len(case_targets)}"
        )

    excluded_rows = [
        {"source_split": role, **row}
        for role, row in tagged_rows
        if row["FASTA"] in case_targets
    ]
    rows = [row for _, row in tagged_rows if row["FASTA"] not in case_targets]
    if len(rows) != args.expected_output_rows:
        raise RuntimeError(
            f"Expected {args.expected_output_rows} rows after case-target exclusion, "
            f"found {len(rows)} (excluded {len(excluded_rows)})"
        )

    source_targets = {row["FASTA"] for _, row in tagged_rows}
    source_pairs = {(row["FASTA"], row["SMILES"]) for _, row in tagged_rows}
    source_target_overlap = sorted(source_targets & case_targets)
    source_pair_overlap = sorted(source_pairs & case_pairs)
    final_targets = {row["FASTA"] for row in rows}
    final_pairs = {(row["FASTA"], row["SMILES"]) for row in rows}
    final_target_overlap = sorted(final_targets & case_targets)
    final_pair_overlap = sorted(final_pairs & case_pairs)
    if final_target_overlap or final_pair_overlap:
        raise RuntimeError("Case-target exclusion failed")

    common.atomic_write_csv(args.output_csv, rows)
    common.atomic_write_exclusions(args.exclusions_csv, excluded_rows)
    exact_rows = [row_key(row) for row in rows]
    pair_before = [f"{fasta}|{smiles}" for fasta, smiles in source_pair_overlap]
    pair_after = [f"{fasta}|{smiles}" for fasta, smiles in final_pair_overlap]
    manifest = {
        "protocol": "2773_full_refit_from_warm_run1_union_four_case_targets_excluded",
        "dataset": "2773",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_row_count": len(tagged_rows),
        "row_count": len(rows),
        "excluded_case_target_row_count": len(excluded_rows),
        "excluded_case_target_rows_by_source": {
            role: sum(row["source_split"] == role for row in excluded_rows)
            for role in source_roles
        },
        "unique_exact_rows": len(set(exact_rows)),
        "unique_target_ligand_pairs_exact_smiles": len(final_pairs),
        "source_files": [
            {
                "role": role,
                "path": str(path.resolve()),
                "rows": len(split_rows),
                "sha256": common.sha256_file(path),
            }
            for role, path, split_rows in zip(source_roles, source_paths, source_rows)
        ],
        "reference_file": {
            "path": str(args.reference_csv.resolve()),
            "rows": len(reference_rows),
            "sha256": common.sha256_file(args.reference_csv),
            "split_union_multiset_match": True,
        },
        "case_file": {
            "path": str(args.case_csv.resolve()),
            "sha256": common.sha256_file(args.case_csv),
            "unique_fasta_sequences": len(case_targets),
        },
        "case_exact_fasta_overlap_before_exclusion_count": len(source_target_overlap),
        "case_exact_pair_overlap_before_exclusion_count": len(source_pair_overlap),
        "case_exact_fasta_overlap_count": len(final_target_overlap),
        "case_exact_pair_overlap_count": len(final_pair_overlap),
        "case_exact_fasta_overlap_before_exclusion_sha256": overlap_hash(
            source_target_overlap
        ),
        "case_exact_pair_overlap_before_exclusion_sha256": overlap_hash(pair_before),
        "case_exact_fasta_overlap_sha256": overlap_hash(final_target_overlap),
        "case_exact_pair_overlap_sha256": overlap_hash(pair_after),
        "output_csv": str(args.output_csv.resolve()),
        "output_sha256": common.sha256_file(args.output_csv),
        "exclusions_csv": str(args.exclusions_csv.resolve()),
        "exclusions_sha256": common.sha256_file(args.exclusions_csv),
        "deduplication_applied": False,
        "case_target_exclusion_applied": True,
    }
    common.atomic_write_json(args.manifest, manifest)

    print(
        f"Prepared full 2773 cohort: {len(rows)} rows; "
        f"excluded case-target rows={len(excluded_rows)}"
    )
    print(f"Split union exactly matches reference: {args.reference_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
