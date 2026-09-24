#!/usr/bin/env python3
"""Serially create and audit the full-2773 ESM2 cache before parallel training."""

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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.data_csv, args.data_manifest, args.config_json, args.model_file, args.esm2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    config = common.read_json(args.config_json)
    common.validate_frozen_config(config, args.model_file)
    manifest = common.read_json(args.data_manifest)
    data_hash = common.sha256_file(args.data_csv)
    if manifest.get("output_sha256") != data_hash or manifest.get("row_count") != 2773:
        raise RuntimeError("Full-refit data/manifest mismatch")
    if args.output.exists():
        previous = common.read_json(args.output)
        if (previous.get("data_sha256") != data_hash
                or previous.get("window_size") != int(config["params"]["window_size"])
                or previous.get("window_layout") != config["params"]["window_layout"]):
            raise RuntimeError("Previous cache audit belongs to a different cohort/configuration")
        old_cache = Path(previous["esm_cache"])
        if not old_cache.is_file() or common.sha256_file(old_cache) != previous["esm_cache_sha256"]:
            raise RuntimeError("Previously audited ESM cache is missing/corrupt; do not silently reuse it")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    mgca = common.load_v10_module(args.model_file)
    legacy = mgca.legacy
    rows = legacy.read_labeled_rows(str(args.data_csv))
    cache_input = args.data_csv.with_name(
        f"{args.data_csv.stem}__sha256_{data_hash[:12]}{args.data_csv.suffix}"
    )
    cache_base = legacy.get_default_esm_cache_path(
        str(cache_input),
        window_size=int(config["params"]["window_size"]),
        window_layout=config["params"]["window_layout"],
    )
    protein, ligand, labels, _ = legacy.preprocess_rows(
        rows,
        str(args.esm2_path),
        device,
        esm_cache=cache_base,
        cache_label="MGCA final 2773 full-refit cache precomputation",
        fingerprint_type="morgan",
        window_size=int(config["params"]["window_size"]),
        window_layout=config["params"]["window_layout"],
    )
    if tuple(protein.shape) != (2773, 4, 2560):
        raise RuntimeError(f"Unexpected ESM feature shape: {tuple(protein.shape)}")
    if tuple(ligand.shape) != (2773, 4, 2048) or len(labels) != 2773:
        raise RuntimeError("Unexpected ligand/label feature shape")
    cache_path = Path(
        legacy._esm_cache_path_for_window(
            cache_base,
            int(config["params"]["window_size"]),
            config["params"]["window_layout"],
        )
    )
    if not cache_path.is_file():
        raise RuntimeError(f"ESM cache was not created: {cache_path}")
    common.atomic_json(
        args.output,
        {
            "status": "passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_sha256": data_hash,
            "row_count": 2773,
            "protein_shape": list(protein.shape),
            "ligand_shape": list(ligand.shape),
            "esm_cache": str(cache_path.resolve()),
            "esm_cache_sha256": common.sha256_file(cache_path),
            "window_size": int(config["params"]["window_size"]),
            "window_layout": config["params"]["window_layout"],
        },
    )
    print(f"Full-2773 ESM2 cache verified: {cache_path}")


if __name__ == "__main__":
    main()
