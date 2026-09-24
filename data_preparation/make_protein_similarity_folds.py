#!/usr/bin/env python3
"""Create protein-similarity-controlled train/val/test folds.

The intended protocol is:
1. read a CSV with FASTA, SMILES/smiles, pkoff;
2. cluster unique protein sequences with MMseqs2;
3. split whole protein clusters into folds;
4. write foldN/train.csv, foldN/val.csv, foldN/test.csv and summaries.

Default clustering matches the KinetX cold-start setting:
30% sequence identity and 80% coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
COMMON_EXTRA_AA = set("BXZUOJ")


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str:
    lower_map = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found:
            return found
    raise ValueError(f"Missing required column. Expected one of {candidates}, got {fieldnames}")


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).split()).upper()


def read_input_csv(path: Path) -> Tuple[List[str], List[dict], str, str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        fieldnames = list(reader.fieldnames)
        fasta_col = pick_column(fieldnames, ["FASTA", "fasta", "sequence", "protein_sequence"])
        smiles_col = pick_column(fieldnames, ["SMILES", "smiles"])
        label_col = pick_column(fieldnames, ["pkoff", "pKoff", "label", "y"])

        rows = []
        for row_idx, row in enumerate(reader, start=2):
            fasta = clean_sequence(row.get(fasta_col, ""))
            smiles = str(row.get(smiles_col, "")).strip()
            label_raw = str(row.get(label_col, "")).strip()
            if not fasta:
                raise ValueError(f"Empty FASTA at CSV line {row_idx}")
            if not smiles:
                raise ValueError(f"Empty SMILES at CSV line {row_idx}")
            try:
                float(label_raw)
            except ValueError as exc:
                raise ValueError(f"Non-numeric {label_col} at CSV line {row_idx}: {label_raw}") from exc

            row[fasta_col] = fasta
            row[smiles_col] = smiles
            row[label_col] = label_raw
            rows.append(row)

    return fieldnames, rows, fasta_col, smiles_col, label_col


def unique_sequences(rows: Sequence[dict], fasta_col: str) -> Tuple[List[str], Dict[str, str]]:
    seqs = []
    seen = set()
    for row in rows:
        seq = row[fasta_col]
        if seq not in seen:
            seen.add(seq)
            seqs.append(seq)
    seq_to_id = {seq: f"prot_{idx}" for idx, seq in enumerate(seqs)}
    return seqs, seq_to_id


def write_fasta(seqs: Sequence[str], seq_to_id: Dict[str, str], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for seq in seqs:
            f.write(f">{seq_to_id[seq]}\n")
            for start in range(0, len(seq), 80):
                f.write(seq[start:start + 80] + "\n")


def invalid_sequence_stats(seqs: Sequence[str]) -> Tuple[int, Dict[str, int]]:
    invalid = Counter()
    bad_count = 0
    allowed = STANDARD_AA | COMMON_EXTRA_AA
    for seq in seqs:
        bad_chars = set(seq) - allowed
        if bad_chars:
            bad_count += 1
            for ch in bad_chars:
                invalid[ch] += 1
    return bad_count, dict(sorted(invalid.items()))


def run_mmseqs(
    fasta_path: Path,
    output_dir: Path,
    mmseqs_bin: str,
    identity: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    overwrite: bool,
) -> Path:
    clusters_tsv = output_dir / "clusters.tsv"
    if clusters_tsv.exists() and not overwrite:
        print(f"Using existing clusters: {clusters_tsv}")
        return clusters_tsv

    if shutil.which(mmseqs_bin) is None:
        raise RuntimeError(
            f"MMseqs2 executable not found: {mmseqs_bin}\n"
            "Install it in the Linux environment, for example:\n"
            "  conda install -c bioconda mmseqs2"
        )

    tmp_dir = output_dir / "mmseqs_tmp"
    prefix = output_dir / "mmseqs_cluster"
    if overwrite:
        for path in [clusters_tsv, Path(str(prefix) + "_cluster.tsv")]:
            if path.exists():
                path.unlink()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]

    print("Running MMseqs2:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    produced = Path(str(prefix) + "_cluster.tsv")
    if not produced.exists():
        candidates = sorted(output_dir.glob("*cluster.tsv"))
        if not candidates:
            raise FileNotFoundError(f"Could not find MMseqs2 cluster TSV under {output_dir}")
        produced = candidates[0]

    shutil.copyfile(produced, clusters_tsv)
    return clusters_tsv


def parse_clusters(path: Path, protein_ids: Iterable[str]) -> Dict[str, str]:
    protein_ids = set(protein_ids)
    member_to_cluster = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            member_to_cluster[member] = rep

    missing = sorted(protein_ids - set(member_to_cluster))
    for protein_id in missing:
        member_to_cluster[protein_id] = protein_id

    if missing:
        print(f"WARNING: {len(missing)} proteins were absent from clusters.tsv; treated as singleton clusters.")
    return member_to_cluster


def build_cluster_records(
    rows: Sequence[dict],
    fasta_col: str,
    label_col: str,
    seq_to_id: Dict[str, str],
    member_to_cluster: Dict[str, str],
) -> List[dict]:
    cluster_to_rows = defaultdict(list)
    cluster_to_proteins = defaultdict(set)
    cluster_to_labels = defaultdict(list)

    for row_idx, row in enumerate(rows):
        seq = row[fasta_col]
        protein_id = seq_to_id[seq]
        cluster_id = member_to_cluster[protein_id]
        cluster_to_rows[cluster_id].append(row_idx)
        cluster_to_proteins[cluster_id].add(protein_id)
        cluster_to_labels[cluster_id].append(float(row[label_col]))

    records = []
    for cluster_id in sorted(cluster_to_rows):
        labels = cluster_to_labels[cluster_id]
        records.append(
            {
                "cluster_id": cluster_id,
                "row_indices": cluster_to_rows[cluster_id],
                "sample_count": len(cluster_to_rows[cluster_id]),
                "protein_count": len(cluster_to_proteins[cluster_id]),
                "label_mean": sum(labels) / len(labels),
                "label_min": min(labels),
                "label_max": max(labels),
            }
        )
    return records


def assign_test_folds(cluster_records: Sequence[dict], n_splits: int, seed: int) -> List[List[dict]]:
    rng = random.Random(seed)
    records = list(cluster_records)
    rng.shuffle(records)
    records.sort(key=lambda rec: (rec["sample_count"], rec["protein_count"]), reverse=True)

    folds = [[] for _ in range(n_splits)]
    fold_samples = [0 for _ in range(n_splits)]
    fold_proteins = [0 for _ in range(n_splits)]
    for rec in records:
        fold_idx = min(range(n_splits), key=lambda idx: (fold_samples[idx], fold_proteins[idx], len(folds[idx])))
        folds[fold_idx].append(rec)
        fold_samples[fold_idx] += rec["sample_count"]
        fold_proteins[fold_idx] += rec["protein_count"]
    return folds


def choose_validation_clusters(train_val_records: Sequence[dict], val_fraction: float, seed: int) -> set:
    rng = random.Random(seed)
    records = list(train_val_records)
    rng.shuffle(records)

    target = max(1, int(round(sum(rec["sample_count"] for rec in records) * val_fraction)))
    if not records:
        return set()

    # Small subset-sum DP over sample counts. The datasets here are small enough
    # that exact balancing is cheap and avoids validation folds dominated by one
    # large cluster when a better combination exists.
    dp = {0: ()}
    total_samples = sum(rec["sample_count"] for rec in records)
    for rec_idx, rec in enumerate(records):
        count = rec["sample_count"]
        for current_sum, chosen in list(dp.items()):
            new_sum = current_sum + count
            if new_sum <= total_samples and new_sum not in dp:
                dp[new_sum] = chosen + (rec_idx,)

    non_empty_sums = [value for value in dp if value > 0]
    best_sum = min(non_empty_sums, key=lambda value: (abs(value - target), value > target, value))
    return {records[idx]["cluster_id"] for idx in dp[best_sum]}


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict], indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in indices:
            writer.writerow(rows[idx])


def flatten_indices(records: Sequence[dict]) -> List[int]:
    indices = []
    for rec in records:
        indices.extend(rec["row_indices"])
    return indices


def median(values: Sequence[int]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def summarize_counts(values: Sequence[int]) -> dict:
    if not values:
        return {"min": 0, "median": 0, "max": 0, "mean": 0}
    return {
        "min": min(values),
        "median": median(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def write_cluster_summary(path: Path, records: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "sample_count",
                "protein_count",
                "label_mean",
                "label_min",
                "label_max",
            ],
        )
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r["sample_count"], reverse=True):
            writer.writerow(
                {
                    "cluster_id": rec["cluster_id"],
                    "sample_count": rec["sample_count"],
                    "protein_count": rec["protein_count"],
                    "label_mean": f"{rec['label_mean']:.6f}",
                    "label_min": f"{rec['label_min']:.6f}",
                    "label_max": f"{rec['label_max']:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Protein similarity cluster-level split generator.")
    parser.add_argument("--input_csv", default="2773/koff.csv")
    parser.add_argument("--output_dir", default="2773/protein_similarity_30id_80cov_5fold")
    parser.add_argument("--identity", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--cov_mode", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--mmseqs_bin", default="mmseqs")
    parser.add_argument("--clusters_tsv", default=None, help="Reuse an existing MMseqs cluster TSV instead of clustering.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min_test_samples", type=int, default=50)
    parser.add_argument("--max_largest_cluster_share", type=float, default=None,
                        help="Largest cluster sample share allowed for a balanced k-fold assessment. "
                             "Default is 1.5 / n_splits.")
    parser.add_argument("--max_test_cv", type=float, default=0.35,
                        help="Maximum allowed coefficient of variation for test fold sizes.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows, fasta_col, smiles_col, label_col = read_input_csv(input_csv)
    seqs, seq_to_id = unique_sequences(rows, fasta_col)
    id_to_seq = {protein_id: seq for seq, protein_id in seq_to_id.items()}
    bad_seq_count, bad_chars = invalid_sequence_stats(seqs)

    fasta_path = output_dir / "proteins.fasta"
    write_fasta(seqs, seq_to_id, fasta_path)

    if args.clusters_tsv:
        clusters_tsv = Path(args.clusters_tsv)
        if not clusters_tsv.exists():
            raise FileNotFoundError(clusters_tsv)
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

    member_to_cluster = parse_clusters(clusters_tsv, id_to_seq.keys())
    cluster_records = build_cluster_records(rows, fasta_col, label_col, seq_to_id, member_to_cluster)

    cluster_count = len(cluster_records)
    protein_count = len(seqs)
    sample_count = len(rows)
    sample_counts = [rec["sample_count"] for rec in cluster_records]
    protein_counts = [rec["protein_count"] for rec in cluster_records]
    singleton_clusters = sum(1 for rec in cluster_records if rec["protein_count"] == 1)

    folds = assign_test_folds(cluster_records, args.n_splits, args.seed)
    fold_rows = []

    all_cluster_ids = {rec["cluster_id"] for rec in cluster_records}
    record_by_cluster = {rec["cluster_id"]: rec for rec in cluster_records}

    for fold_idx, test_records in enumerate(folds, start=1):
        test_cluster_ids = {rec["cluster_id"] for rec in test_records}
        train_val_records = [rec for rec in cluster_records if rec["cluster_id"] not in test_cluster_ids]
        val_cluster_ids = choose_validation_clusters(
            train_val_records,
            val_fraction=args.val_fraction,
            seed=args.seed + fold_idx * 1009,
        )
        train_cluster_ids = all_cluster_ids - test_cluster_ids - val_cluster_ids

        train_records = [record_by_cluster[cid] for cid in sorted(train_cluster_ids)]
        val_records = [record_by_cluster[cid] for cid in sorted(val_cluster_ids)]

        train_indices = flatten_indices(train_records)
        val_indices = flatten_indices(val_records)
        test_indices = flatten_indices(test_records)

        fold_dir = output_dir / f"fold{fold_idx}"
        write_csv(fold_dir / "train.csv", fieldnames, rows, train_indices)
        write_csv(fold_dir / "val.csv", fieldnames, rows, val_indices)
        write_csv(fold_dir / "test.csv", fieldnames, rows, test_indices)

        if train_cluster_ids & val_cluster_ids or train_cluster_ids & test_cluster_ids or val_cluster_ids & test_cluster_ids:
            raise RuntimeError(f"Cluster leakage detected in fold{fold_idx}")

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
                "train_clusters": len(train_cluster_ids),
                "val_clusters": len(val_cluster_ids),
                "test_clusters": len(test_cluster_ids),
                "train_proteins": sum(record_by_cluster[cid]["protein_count"] for cid in train_cluster_ids),
                "val_proteins": sum(record_by_cluster[cid]["protein_count"] for cid in val_cluster_ids),
                "test_proteins": sum(record_by_cluster[cid]["protein_count"] for cid in test_cluster_ids),
            }
        )

    write_cluster_summary(output_dir / "cluster_summary.csv", cluster_records)

    with (output_dir / "fold_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    test_samples = [row["test_samples"] for row in fold_rows]
    mean_test = sum(test_samples) / len(test_samples)
    sd_test = statistics.stdev(test_samples) if len(test_samples) > 1 else 0.0
    max_cluster_share = max(sample_counts) / sample_count if sample_count else 0.0
    max_largest_cluster_share = (
        args.max_largest_cluster_share
        if args.max_largest_cluster_share is not None
        else 1.5 / args.n_splits
    )
    fold_test_cv = sd_test / mean_test if mean_test else float("nan")
    can_make_5fold = (
        cluster_count >= args.n_splits
        and min(test_samples) >= args.min_test_samples
        and max_cluster_share <= max_largest_cluster_share
        and fold_test_cv <= args.max_test_cv
    )

    summary = {
        "input_csv": str(input_csv),
        "rows": sample_count,
        "columns": {"fasta": fasta_col, "smiles": smiles_col, "label": label_col},
        "unique_proteins": protein_count,
        "invalid_unique_FASTA_count": bad_seq_count,
        "invalid_FASTA_characters": bad_chars,
        "identity_threshold": args.identity,
        "coverage_threshold": args.coverage,
        "cov_mode": args.cov_mode,
        "clusters": cluster_count,
        "singleton_clusters": singleton_clusters,
        "cluster_sample_count": summarize_counts(sample_counts),
        "cluster_protein_count": summarize_counts(protein_counts),
        "largest_cluster_sample_share": max_cluster_share,
        "fold_test_samples_mean": mean_test,
        "fold_test_samples_sd": sd_test,
        "fold_test_samples_cv": fold_test_cv,
        "max_largest_cluster_share_threshold": max_largest_cluster_share,
        "max_test_cv_threshold": args.max_test_cv,
        "can_make_5fold_by_basic_checks": can_make_5fold,
        "folds": fold_rows,
    }

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "split_summary.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("Protein Similarity Split Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input CSV: {input_csv}\n")
        f.write(f"Rows: {sample_count}\n")
        f.write(f"Unique proteins: {protein_count}\n")
        f.write(f"Similarity threshold: {args.identity:.2f}\n")
        f.write(f"Coverage threshold: {args.coverage:.2f}\n")
        f.write(f"MMseqs cov-mode: {args.cov_mode}\n")
        f.write(f"Clusters: {cluster_count}\n")
        f.write(f"Singleton clusters: {singleton_clusters}\n")
        f.write(f"Largest cluster sample share: {max_cluster_share:.3f}\n")
        f.write(f"Fold test sample CV: {fold_test_cv:.3f}\n")
        f.write(f"Invalid unique FASTA count: {bad_seq_count}\n")
        if bad_chars:
            f.write(f"Invalid FASTA characters: {bad_chars}\n")
        f.write("\nCluster sample count summary:\n")
        for key, value in summary["cluster_sample_count"].items():
            f.write(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
            f.write("\n")
        f.write("\nFold summary:\n")
        for row in fold_rows:
            f.write(
                f"  fold{row['fold']}: train={row['train_samples']} "
                f"val={row['val_samples']} test={row['test_samples']} "
                f"test_clusters={row['test_clusters']} test_proteins={row['test_proteins']}\n"
            )
        f.write("\n5-fold ability assessment:\n")
        if can_make_5fold:
            f.write("  PASS basic checks: cluster-level 5-fold split is feasible.\n")
        else:
            f.write("  WARNING: cluster-level 5-fold split may be unstable or infeasible.\n")
            if cluster_count < args.n_splits:
                f.write(f"  Reason: clusters ({cluster_count}) < n_splits ({args.n_splits}).\n")
            if min(test_samples) < args.min_test_samples:
                f.write(f"  Reason: minimum test samples ({min(test_samples)}) < {args.min_test_samples}.\n")
            if max_cluster_share > max_largest_cluster_share:
                f.write(
                    f"  Reason: largest cluster contains {max_cluster_share:.1%} of all samples, "
                    f"above the balance threshold {max_largest_cluster_share:.1%}.\n"
                )
            if fold_test_cv > args.max_test_cv:
                f.write(
                    f"  Reason: test fold size CV is {fold_test_cv:.3f}, "
                    f"above the threshold {args.max_test_cv:.3f}.\n"
                )

    print("\nDone.")
    print(f"Clusters: {cluster_count}")
    print(f"Unique proteins: {protein_count}")
    print(f"Fold summary: {output_dir / 'fold_summary.csv'}")
    print(f"Split summary: {output_dir / 'split_summary.txt'}")
    print(f"5-fold basic feasibility: {'PASS' if can_make_5fold else 'CHECK MANUALLY'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
