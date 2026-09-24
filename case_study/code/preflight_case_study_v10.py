#!/usr/bin/env python3
"""Audit the inputs, frozen v10 implementation, and optional five checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

import common_v10 as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, default=common.V10_MODEL)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--four-target-csv", type=Path, required=True)
    parser.add_argument("--k4dd-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--train-split", type=Path)
    parser.add_argument("--val-split", type=Path)
    parser.add_argument("--test-split", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--check-checkpoints", action="store_true")
    parser.add_argument("--epochs", type=int, default=common.DEFAULT_EPOCHS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    required = (
        args.data_csv, args.data_manifest, args.config_json, args.model_file,
        args.esm2_path, args.four_target_csv, args.k4dd_csv,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    config = common.read_json(args.config_json)
    common.validate_frozen_config(config, args.model_file)
    data_manifest = common.read_json(args.data_manifest)
    if data_manifest.get("row_count") != 2773:
        raise RuntimeError("Full-refit manifest row_count must be 2773")
    if data_manifest.get("output_sha256") != common.sha256_file(args.data_csv):
        raise RuntimeError("Full-refit CSV/manifest hash mismatch")
    if data_manifest.get("case_exact_fasta_overlap_count") != 0:
        raise RuntimeError("Four-target exact FASTA overlap is not zero")
    case_record = data_manifest.get("case_file", {})
    if case_record.get("sha256") != common.sha256_file(args.four_target_csv):
        raise RuntimeError("Training manifest was prepared against a different case CSV")
    if args.reference_csv is not None:
        if not args.reference_csv.is_file():
            raise FileNotFoundError(args.reference_csv)
        if data_manifest.get("reference_file", {}).get("sha256") != common.sha256_file(args.reference_csv):
            raise RuntimeError("Training manifest/reference 2773 CSV hash mismatch")
    supplied_splits = {
        "train": args.train_split,
        "validation": args.val_split,
        "test": args.test_split,
    }
    source_records = {item.get("role"): item for item in data_manifest.get("source_files", [])}
    for role, path in supplied_splits.items():
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        if source_records.get(role, {}).get("sha256") != common.sha256_file(path):
            raise RuntimeError(f"Training manifest/{role} split hash mismatch")
    four_rows = common.read_case_rows(args.four_target_csv)
    k4dd_rows = common.read_case_rows(args.k4dd_csv)
    if len({row["uniprot_id"] for row in four_rows}) != 4:
        raise RuntimeError("Four-target CSV does not contain exactly four target IDs")
    if len({row["uniprot_id"] for row in k4dd_rows}) != 6:
        raise RuntimeError("K4DD CSV does not contain exactly six target IDs")
    mgca = common.load_v10_module(args.model_file)
    kwargs = common.model_kwargs_from_frozen_config(config)
    model = mgca.FullRegressionTransformer(**kwargs)
    parameter_count = mgca.parameter_count(model)
    checkpoint_audit = []
    if args.check_checkpoints:
        if args.checkpoint_root is None:
            raise ValueError("--checkpoint-root is required with --check-checkpoints")
        records = common.checkpoint_records(args.checkpoint_root, epochs=args.epochs)
        data_hash = None
        for record in records:
            payload = common.torch_load(torch, record["path"], map_location="cpu")
            data_hash = common.validate_checkpoint(payload, record, data_hash, args.epochs)
            checkpoint_audit.append(
                {
                    "seed": record["seed"],
                    "path": str(record["path"].resolve()),
                    "sha256": record["sha256"],
                    "parameter_count": payload.get("parameter_count"),
                }
            )
    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": config["config_id"],
        "model_sha256": common.sha256_file(args.model_file),
        "config_sha256": common.sha256_file(args.config_json),
        "data_sha256": common.sha256_file(args.data_csv),
        "training_rows": data_manifest["row_count"],
        "case_exact_fasta_overlap_count": data_manifest["case_exact_fasta_overlap_count"],
        "four_target_rows": len(four_rows),
        "four_target_ids": sorted({row["uniprot_id"] for row in four_rows}),
        "k4dd_rows": len(k4dd_rows),
        "k4dd_target_ids": sorted({row["uniprot_id"] for row in k4dd_rows}),
        "parameter_count": parameter_count,
        "checkpoint_audit": checkpoint_audit,
    }
    common.atomic_json(args.output, report)
    print(f"Preflight passed: {args.output}")


if __name__ == "__main__":
    main()
