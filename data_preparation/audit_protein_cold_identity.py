#!/usr/bin/env python3
"""Audit sequence leakage in the fixed protein cold-start splits.

The script uses only the Python standard library plus an external MMseqs2
binary. It reads the current KinetX and 2773 train/validation/test CSV files,
deduplicates protein sequences within each split, runs MMseqs2 easy-search,
and writes both detailed alignments and publication-ready summary tables.

The primary leakage definition is:

    sequence identity >= 30% AND query coverage >= 80%
    AND target coverage >= 80%.

By default, the script exits with status 2 after writing all reports if an
exact cross-split duplicate or a threshold-violating aligned pair is found.
Use --allow-violations only when inspecting a known failing split.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FASTA_COLUMN_CANDIDATES = (
    "fasta",
    "sequence",
    "protein_sequence",
    "protein",
    "target_sequence",
)

MMSEQS_FIELDS = (
    "query",
    "target",
    "fident",
    "alnlen",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "qlen",
    "tstart",
    "tend",
    "tlen",
    "evalue",
    "bits",
    "qcov",
    "tcov",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_csv: Path
    val_csv: Path
    test_csv: Path


@dataclass(frozen=True)
class ProteinRecord:
    seq_id: str
    sequence: str
    sha256: str
    labels: tuple[str, ...]
    source_rows: int


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Audit fixed protein cold-start splits with MMseqs2."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help=f"Project root (default: {default_root})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project-root>/protein_cold_identity_audit)",
    )
    parser.add_argument(
        "--mmseqs",
        default="mmseqs",
        help="MMseqs2 executable or absolute path (default: mmseqs)",
    )
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=30.0,
        help="Violation identity threshold in percent (default: 30)",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Required query AND target coverage in percent (default: 80)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=7.5,
        help="MMseqs2 sensitivity (-s; default: 7.5)",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=1_000_000,
        help="Maximum target hits retained per query (default: 1000000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run searches and replace only this script's existing result/tmp files.",
    )
    parser.add_argument(
        "--allow-violations",
        action="store_true",
        help="Return status 0 even if leakage violations are found.",
    )
    return parser.parse_args()


def normalize_sequence(value: str) -> str:
    sequence = re.sub(r"\s+", "", value or "").upper().rstrip("*")
    if not sequence:
        raise ValueError("Encountered an empty protein sequence")
    if not re.fullmatch(r"[A-Z]+", sequence):
        bad = "".join(sorted(set(re.sub(r"[A-Z]", "", sequence))))
        raise ValueError(f"Protein sequence contains unsupported characters: {bad!r}")
    return sequence


def find_column(fieldnames: Sequence[str] | None, candidates: Sequence[str]) -> str:
    if not fieldnames:
        raise ValueError("CSV has no header")
    by_lower = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        f"Could not find a protein-sequence column. Available columns: {fieldnames}"
    )


def read_unique_proteins(csv_path: Path, dataset: str, split: str) -> list[ProteinRecord]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing split file: {csv_path}")

    grouped: dict[str, dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        sequence_column = find_column(reader.fieldnames, FASTA_COLUMN_CANDIDATES)
        label_column = None
        if reader.fieldnames:
            lower = {name.strip().lower(): name for name in reader.fieldnames}
            label_column = lower.get("target_name") or lower.get("target")

        for row_number, row in enumerate(reader, start=2):
            try:
                sequence = normalize_sequence(row.get(sequence_column, ""))
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{row_number}: {exc}") from exc
            digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            label = (row.get(label_column, "").strip() if label_column else "") or "NA"
            entry = grouped.setdefault(
                digest,
                {"sequence": sequence, "labels": set(), "source_rows": 0},
            )
            entry["labels"].add(label)  # type: ignore[union-attr]
            entry["source_rows"] = int(entry["source_rows"]) + 1

    records: list[ProteinRecord] = []
    for index, digest in enumerate(sorted(grouped), start=1):
        entry = grouped[digest]
        records.append(
            ProteinRecord(
                seq_id=f"{dataset}_{split}_{index:05d}_{digest[:12]}",
                sequence=str(entry["sequence"]),
                sha256=digest,
                labels=tuple(sorted(entry["labels"])),  # type: ignore[arg-type]
                source_rows=int(entry["source_rows"]),
            )
        )
    if not records:
        raise ValueError(f"No protein sequences found in {csv_path}")
    return records


def combine_records(
    first: Sequence[ProteinRecord],
    second: Sequence[ProteinRecord],
    dataset: str,
    split: str,
) -> list[ProteinRecord]:
    grouped: dict[str, dict[str, object]] = {}
    for record in list(first) + list(second):
        entry = grouped.setdefault(
            record.sha256,
            {"sequence": record.sequence, "labels": set(), "source_rows": 0},
        )
        entry["labels"].update(record.labels)  # type: ignore[union-attr]
        entry["source_rows"] = int(entry["source_rows"]) + record.source_rows

    combined: list[ProteinRecord] = []
    for index, digest in enumerate(sorted(grouped), start=1):
        entry = grouped[digest]
        combined.append(
            ProteinRecord(
                seq_id=f"{dataset}_{split}_{index:05d}_{digest[:12]}",
                sequence=str(entry["sequence"]),
                sha256=digest,
                labels=tuple(sorted(entry["labels"])),  # type: ignore[arg-type]
                source_rows=int(entry["source_rows"]),
            )
        )
    return combined


def write_fasta(records: Sequence[ProteinRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.seq_id}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(record.sequence[start : start + 80] + "\n")


def write_manifest(records: Sequence[ProteinRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seq_id", "sha256", "length", "source_rows", "target_labels"]
        )
        for record in records:
            writer.writerow(
                [
                    record.seq_id,
                    record.sha256,
                    len(record.sequence),
                    record.source_rows,
                    " | ".join(record.labels),
                ]
            )


def require_mmseqs(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    candidate = Path(executable)
    if candidate.is_file():
        return str(candidate.resolve())
    raise FileNotFoundError(
        f"MMseqs2 executable not found: {executable!r}. Install MMseqs2 or pass "
        "--mmseqs /absolute/path/to/mmseqs."
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mmseqs_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def run_mmseqs(
    mmseqs: str,
    query_fasta: Path,
    target_fasta: Path,
    result_tsv: Path,
    tmp_dir: Path,
    args: argparse.Namespace,
) -> None:
    if result_tsv.exists() and not args.force:
        print(f"[reuse] {result_tsv}")
        return

    if args.force:
        if result_tsv.exists():
            result_tsv.unlink()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        mmseqs,
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(result_tsv),
        str(tmp_dir),
        "--format-mode",
        "0",
        "--format-output",
        ",".join(MMSEQS_FIELDS),
        "--min-seq-id",
        "0.0",
        "-c",
        "0.0",
        "--cov-mode",
        "0",
        "-s",
        str(args.sensitivity),
        "--max-seqs",
        str(args.max_seqs),
        "--threads",
        str(args.threads),
    ]
    print("[run] " + " ".join(command))
    subprocess.run(command, check=True)


def fraction_to_percent(value: str) -> float:
    number = float(value)
    return number * 100.0 if number <= 1.000001 else number


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def parse_hits(result_tsv: Path) -> list[dict[str, object]]:
    pair_best: dict[tuple[str, str], dict[str, object]] = {}
    if not result_tsv.exists() or result_tsv.stat().st_size == 0:
        return []

    with result_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=MMSEQS_FIELDS, delimiter="\t")
        for row in reader:
            identity = fraction_to_percent(row["fident"])
            qcov = fraction_to_percent(row["qcov"])
            tcov = fraction_to_percent(row["tcov"])
            bits = float(row["bits"])
            parsed: dict[str, object] = {
                **row,
                "identity_pct": identity,
                "qcov_pct": qcov,
                "tcov_pct": tcov,
                "min_cov_pct": min(qcov, tcov),
                "bits_float": bits,
            }
            key = (row["query"], row["target"])
            previous = pair_best.get(key)
            score = (min(qcov, tcov), identity, bits)
            previous_score = (
                (
                    float(previous["min_cov_pct"]),
                    float(previous["identity_pct"]),
                    float(previous["bits_float"]),
                )
                if previous
                else None
            )
            if previous_score is None or score > previous_score:
                pair_best[key] = parsed
    return list(pair_best.values())


def write_hits_csv(hits: Sequence[dict[str, object]], path: Path) -> None:
    columns = [
        "query",
        "target",
        "identity_pct",
        "qcov_pct",
        "tcov_pct",
        "min_cov_pct",
        "alnlen",
        "qlen",
        "tlen",
        "evalue",
        "bits",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for hit in sorted(
            hits,
            key=lambda item: (
                -float(item["identity_pct"]),
                -float(item["min_cov_pct"]),
                str(item["query"]),
                str(item["target"]),
            ),
        ):
            writer.writerow({column: hit[column] for column in columns})


def exact_overlaps(
    query_records: Sequence[ProteinRecord], target_records: Sequence[ProteinRecord]
) -> list[tuple[ProteinRecord, ProteinRecord]]:
    target_by_hash = {record.sha256: record for record in target_records}
    return [
        (record, target_by_hash[record.sha256])
        for record in query_records
        if record.sha256 in target_by_hash
    ]


def write_exact_overlap_csv(
    overlaps: Sequence[tuple[ProteinRecord, ProteinRecord]], path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_id",
                "target_id",
                "sha256",
                "length",
                "query_labels",
                "target_labels",
            ]
        )
        for query, target in overlaps:
            writer.writerow(
                [
                    query.seq_id,
                    target.seq_id,
                    query.sha256,
                    len(query.sequence),
                    " | ".join(query.labels),
                    " | ".join(target.labels),
                ]
            )


def format_optional(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def make_dataset_specs(project_root: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="KinetX",
            train_csv=project_root / "KinetX" / "cold_start" / "train.csv",
            val_csv=project_root / "KinetX" / "cold_start" / "val.csv",
            test_csv=project_root / "KinetX" / "cold_start" / "test.csv",
        ),
        DatasetSpec(
            name="2773",
            train_csv=project_root
            / "2773"
            / "new_folds"
            / "target-cold"
            / "train.csv",
            val_csv=project_root
            / "2773"
            / "new_folds"
            / "target-cold"
            / "val.csv",
            test_csv=project_root
            / "2773"
            / "new_folds"
            / "target-cold"
            / "test.csv",
        ),
    ]


def audit_comparison(
    dataset: str,
    comparison: str,
    query_records: Sequence[ProteinRecord],
    target_records: Sequence[ProteinRecord],
    dataset_dir: Path,
    mmseqs: str,
    args: argparse.Namespace,
) -> tuple[dict[str, object], int]:
    fasta_dir = dataset_dir / "fasta"
    result_dir = dataset_dir / "comparisons" / comparison
    result_dir.mkdir(parents=True, exist_ok=True)

    query_fasta = fasta_dir / f"{comparison}__query.fasta"
    target_fasta = fasta_dir / f"{comparison}__target.fasta"
    write_fasta(query_records, query_fasta)
    write_fasta(target_records, target_fasta)

    raw_tsv = result_dir / "mmseqs_raw.tsv"
    run_mmseqs(
        mmseqs=mmseqs,
        query_fasta=query_fasta,
        target_fasta=target_fasta,
        result_tsv=raw_tsv,
        tmp_dir=dataset_dir / "mmseqs_tmp" / comparison,
        args=args,
    )

    hits = parse_hits(raw_tsv)
    coverage_hits = [
        hit
        for hit in hits
        if float(hit["qcov_pct"]) >= args.coverage_threshold
        and float(hit["tcov_pct"]) >= args.coverage_threshold
    ]
    violations = [
        hit
        for hit in coverage_hits
        if float(hit["identity_pct"]) >= args.identity_threshold
    ]
    overlaps = exact_overlaps(query_records, target_records)

    write_hits_csv(hits, result_dir / "all_pair_best_hits.csv")
    write_hits_csv(coverage_hits, result_dir / "coverage_qualified_hits.csv")
    write_hits_csv(violations, result_dir / "identity_violations.csv")
    write_exact_overlap_csv(overlaps, result_dir / "exact_sequence_overlaps.csv")

    identities = [float(hit["identity_pct"]) for hit in coverage_hits]
    max_hit = (
        max(
            coverage_hits,
            key=lambda item: (
                float(item["identity_pct"]),
                float(item["min_cov_pct"]),
                float(item["bits_float"]),
            ),
        )
        if coverage_hits
        else None
    )
    summary: dict[str, object] = {
        "dataset": dataset,
        "comparison": comparison,
        "query_unique_proteins": len(query_records),
        "target_unique_proteins": len(target_records),
        "exact_sequence_overlaps": len(overlaps),
        "mmseqs_pair_hits": len(hits),
        "coverage_qualified_hits": len(coverage_hits),
        "coverage_threshold_pct_each_side": args.coverage_threshold,
        "identity_threshold_pct": args.identity_threshold,
        "identity_violations": len(violations),
        "max_identity_pct": (
            float(max_hit["identity_pct"]) if max_hit is not None else None
        ),
        "max_identity_query_coverage_pct": (
            float(max_hit["qcov_pct"]) if max_hit is not None else None
        ),
        "max_identity_target_coverage_pct": (
            float(max_hit["tcov_pct"]) if max_hit is not None else None
        ),
        "max_identity_query": str(max_hit["query"]) if max_hit else "NA",
        "max_identity_target": str(max_hit["target"]) if max_hit else "NA",
        "p95_identity_pct": percentile(identities, 0.95),
        "mean_identity_pct": statistics.fmean(identities) if identities else None,
        "status": "PASS" if not overlaps and not violations else "FAIL",
    }
    issue_count = len(overlaps) + len(violations)
    return summary, issue_count


def write_summary_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    rows: Sequence[dict[str, object]], path: Path, args: argparse.Namespace
) -> None:
    lines = [
        "# Protein cold-start sequence-identity leakage audit",
        "",
        "The protein-cold protocol uses one fixed train/validation/test split per "
        "dataset. The five reported model runs are random seeds on the same split, "
        "so sequence leakage is audited once per dataset rather than once per seed.",
        "",
        f"Violation rule: identity >= {args.identity_threshold:g}% with both query "
        f"and target coverage >= {args.coverage_threshold:g}%.",
        "",
        "| Dataset | Comparison | Query proteins | Target proteins | Exact overlap | "
        "Max identity | qcov | tcov | P95 identity | Violations | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {comparison} | {query} | {target} | {overlap} | "
            "{max_identity} | {qcov} | {tcov} | {p95} | {violations} | {status} |".format(
                dataset=row["dataset"],
                comparison=str(row["comparison"]).replace("_", "\\_"),
                query=row["query_unique_proteins"],
                target=row["target_unique_proteins"],
                overlap=row["exact_sequence_overlaps"],
                max_identity=format_optional(row["max_identity_pct"]),  # type: ignore[arg-type]
                qcov=format_optional(
                    row["max_identity_query_coverage_pct"]  # type: ignore[arg-type]
                ),
                tcov=format_optional(
                    row["max_identity_target_coverage_pct"]  # type: ignore[arg-type]
                ),
                p95=format_optional(row["p95_identity_pct"]),  # type: ignore[arg-type]
                violations=row["identity_violations"],
                status=row["status"],
            )
        )

    lines.extend(
        [
            "",
            "## Reporting note",
            "",
            "`Max identity`, `P95 identity`, and violation counts are calculated "
            "only from alignments satisfying the coverage threshold on both "
            "sequences. Detailed alignments and exact-overlap records are retained "
            "under each comparison directory.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if not 0 <= args.identity_threshold <= 100:
        raise ValueError("--identity-threshold must be between 0 and 100")
    if not 0 <= args.coverage_threshold <= 100:
        raise ValueError("--coverage-threshold must be between 0 and 100")

    project_root = args.project_root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else project_root / "protein_cold_identity_audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = require_mmseqs(args.mmseqs)

    specs = make_dataset_specs(project_root)
    metadata = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "project_root": str(project_root),
        "mmseqs_executable": mmseqs,
        "mmseqs_version": mmseqs_version(mmseqs),
        "identity_threshold_pct": args.identity_threshold,
        "coverage_threshold_pct_each_side": args.coverage_threshold,
        "mmseqs_sensitivity": args.sensitivity,
        "mmseqs_max_seqs": args.max_seqs,
        "threads": args.threads,
        "input_files": {
            spec.name: {
                "train": {
                    "path": str(spec.train_csv),
                    "sha256": file_sha256(spec.train_csv),
                },
                "validation": {
                    "path": str(spec.val_csv),
                    "sha256": file_sha256(spec.val_csv),
                },
                "test": {
                    "path": str(spec.test_csv),
                    "sha256": file_sha256(spec.test_csv),
                },
            }
            for spec in specs
        },
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    all_rows: list[dict[str, object]] = []
    total_issues = 0
    for spec in specs:
        print(f"\n=== {spec.name} ===")
        dataset_dir = out_dir / spec.name
        train = read_unique_proteins(spec.train_csv, spec.name, "train")
        val = read_unique_proteins(spec.val_csv, spec.name, "validation")
        test = read_unique_proteins(spec.test_csv, spec.name, "test")
        train_val = combine_records(train, val, spec.name, "train_validation")

        manifest_dir = dataset_dir / "manifests"
        write_manifest(train, manifest_dir / "train_unique_proteins.csv")
        write_manifest(val, manifest_dir / "validation_unique_proteins.csv")
        write_manifest(test, manifest_dir / "test_unique_proteins.csv")
        write_manifest(
            train_val, manifest_dir / "train_validation_unique_proteins.csv"
        )

        comparisons = [
            ("train_vs_validation", train, val),
            ("train_vs_test", train, test),
            ("validation_vs_test", val, test),
            ("train_validation_vs_test", train_val, test),
        ]
        for comparison, query_records, target_records in comparisons:
            row, issues = audit_comparison(
                dataset=spec.name,
                comparison=comparison,
                query_records=query_records,
                target_records=target_records,
                dataset_dir=dataset_dir,
                mmseqs=mmseqs,
                args=args,
            )
            all_rows.append(row)
            total_issues += issues
            print(
                f"[{row['status']}] {comparison}: max_identity="
                f"{format_optional(row['max_identity_pct'])}% "
                f"violations={row['identity_violations']} "
                f"exact_overlap={row['exact_sequence_overlaps']}"
            )

    summary_csv = out_dir / "protein_cold_identity_summary.csv"
    summary_md = out_dir / "protein_cold_identity_summary.md"
    write_summary_csv(all_rows, summary_csv)
    write_summary_markdown(all_rows, summary_md, args)

    print(f"\nSummary CSV: {summary_csv}")
    print(f"Summary Markdown: {summary_md}")
    if total_issues and not args.allow_violations:
        print(
            f"FAIL: found {total_issues} exact-overlap/identity-violation records "
            "across the comparison reports (a pair may also occur in the combined "
            "train+validation comparison). Reports were still written.",
            file=sys.stderr,
        )
        return 2
    print("PASS: no threshold violations or exact cross-split duplicates found.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"MMseqs2 failed with exit status {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
