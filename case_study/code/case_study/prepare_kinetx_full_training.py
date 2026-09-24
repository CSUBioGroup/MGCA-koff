#!/usr/bin/env python3
"""Build the case-study final KinetX refit table from one frozen warm split.

The three source CSVs are concatenated without deduplication.  This preserves
the exact warm-benchmark cohort before excluding every exact case-target FASTA.
The remaining rows lose their train/validation/test roles for final refitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


REQUIRED_COLUMNS = ("FASTA", "SMILES", "pkoff")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_fasta(value: str) -> str:
    return "".join(str(value).split()).upper()


def column_map(fieldnames: Iterable[str], path: Path) -> Dict[str, str]:
    lookup = {name.lower(): name for name in fieldnames}
    mapping = {}
    for required in REQUIRED_COLUMNS:
        source = lookup.get(required.lower())
        if source is None:
            raise ValueError(
                f"{path} must contain {REQUIRED_COLUMNS}; found {tuple(fieldnames)}"
            )
        mapping[required] = source
    return mapping


def read_training_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        mapping = column_map(reader.fieldnames, path)
        rows = []
        for line_number, row in enumerate(reader, start=2):
            normalized = {
                "FASTA": normalize_fasta(row[mapping["FASTA"]]),
                "SMILES": str(row[mapping["SMILES"]]).strip(),
                "pkoff": str(row[mapping["pkoff"]]).strip(),
            }
            if not all(normalized.values()):
                raise ValueError(f"Empty required value in {path}:{line_number}")
            try:
                float(normalized["pkoff"])
            except ValueError as exc:
                raise ValueError(f"Invalid pkoff in {path}:{line_number}") from exc
            rows.append(normalized)
    return rows


def read_case_targets(path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Case CSV has no header: {path}")
        lookup = {name.lower(): name for name in reader.fieldnames}
        fasta_col = lookup.get("fasta")
        smiles_col = lookup.get("smiles")
        if fasta_col is None or smiles_col is None:
            raise ValueError(f"Case CSV must contain FASTA and smiles: {path}")
        targets: set[str] = set()
        pairs: set[tuple[str, str]] = set()
        for row in reader:
            fasta = normalize_fasta(row[fasta_col])
            smiles = str(row[smiles_col]).strip()
            if fasta:
                targets.add(fasta)
                pairs.add((fasta, smiles))
    return targets, pairs


def atomic_write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_write_exclusions(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = ("source_split", "FASTA", "SMILES", "pkoff")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a frozen KinetX warm split into the full refit cohort."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--exclusions-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-rows", type=int, default=5446)
    parser.add_argument("--expected-output-rows", type=int, default=5438)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = [args.train_csv, args.val_csv, args.test_csv]
    for path in [*source_paths, args.case_csv]:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_rows = [read_training_rows(path) for path in source_paths]
    source_roles = ("train", "validation", "test")
    tagged_rows = [
        (role, row)
        for role, split_rows in zip(source_roles, source_rows)
        for row in split_rows
    ]
    if len(tagged_rows) != args.expected_source_rows:
        raise RuntimeError(
            f"Expected {args.expected_source_rows} source rows, found {len(tagged_rows)}"
        )

    case_targets, case_pairs = read_case_targets(args.case_csv)
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
    full_targets = {row["FASTA"] for row in rows}
    full_pairs = {(row["FASTA"], row["SMILES"]) for row in rows}
    target_overlap = sorted(full_targets & case_targets)
    pair_overlap = sorted(full_pairs & case_pairs)
    if target_overlap or pair_overlap:
        raise RuntimeError("Case-target exclusion failed")

    exact_row_keys = [
        (row["FASTA"], row["SMILES"], row["pkoff"]) for row in rows
    ]
    atomic_write_csv(args.output_csv, rows)
    atomic_write_exclusions(args.exclusions_csv, excluded_rows)

    manifest = {
        "protocol": "kinetx_full_refit_from_warm_run1_union_case_targets_excluded",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_row_count": len(tagged_rows),
        "row_count": len(rows),
        "excluded_case_target_row_count": len(excluded_rows),
        "excluded_case_target_rows_by_source": {
            role: sum(row["source_split"] == role for row in excluded_rows)
            for role in source_roles
        },
        "unique_exact_rows": len(set(exact_row_keys)),
        "unique_target_ligand_pairs_exact_smiles": len(full_pairs),
        "source_files": [
            {
                "role": role,
                "path": str(path.resolve()),
                "rows": len(split_rows),
                "sha256": sha256_file(path),
            }
            for role, path, split_rows in zip(source_roles, source_paths, source_rows)
        ],
        "case_file": {
            "path": str(args.case_csv.resolve()),
            "sha256": sha256_file(args.case_csv),
            "unique_fasta_sequences": len(case_targets),
        },
        "case_exact_fasta_overlap_before_exclusion_count": len(source_target_overlap),
        "case_exact_pair_overlap_before_exclusion_count": len(source_pair_overlap),
        "case_exact_fasta_overlap_count": len(target_overlap),
        "case_exact_pair_overlap_count": len(pair_overlap),
        "case_exact_fasta_overlap_before_exclusion_sha256": hashlib.sha256(
            "\n".join(source_target_overlap).encode("utf-8")
        ).hexdigest(),
        "case_exact_pair_overlap_before_exclusion_sha256": hashlib.sha256(
            "\n".join(
                f"{fasta}|{smiles}" for fasta, smiles in source_pair_overlap
            ).encode("utf-8")
        ).hexdigest(),
        "case_exact_fasta_overlap_sha256": hashlib.sha256(
            "\n".join(target_overlap).encode("utf-8")
        ).hexdigest(),
        "case_exact_pair_overlap_sha256": hashlib.sha256(
            "\n".join(f"{fasta}|{smiles}" for fasta, smiles in pair_overlap).encode(
                "utf-8"
            )
        ).hexdigest(),
        "output_csv": str(args.output_csv.resolve()),
        "output_sha256": sha256_file(args.output_csv),
        "exclusions_csv": str(args.exclusions_csv.resolve()),
        "exclusions_sha256": sha256_file(args.exclusions_csv),
        "deduplication_applied": False,
        "case_target_exclusion_applied": True,
    }
    atomic_write_json(args.manifest, manifest)

    print(
        f"Prepared {len(rows)} rows after excluding {len(excluded_rows)} "
        f"case-target rows: {args.output_csv}"
    )
    print(f"Manifest: {args.manifest}")
    print(f"Output SHA256: {manifest['output_sha256']}")
    print(
        "Case overlap: "
        f"before={len(source_target_overlap)} targets/{len(source_pair_overlap)} pairs; "
        f"after={len(target_overlap)} targets/{len(pair_overlap)} pairs"
    )


if __name__ == "__main__":
    main()
