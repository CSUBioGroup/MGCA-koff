#!/usr/bin/env python3
"""Infer the prior four-target case set with five full-data 2773 MGCA models."""

from __future__ import annotations

import argparse
from pathlib import Path

import infer_mgca_case_ensemble as base


EXPECTED_CONFIG_ID_2773 = "f3a43680dda9"
DEFAULT_EPOCHS_2773 = 7


def parse_args_2773() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242, 342, 442])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS_2773)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--esm-batch-size", type=int, default=1)
    parser.add_argument("--batch-invariance-tolerance", type=float, default=1e-5)
    parser.add_argument("--expected-config-id", default=EXPECTED_CONFIG_ID_2773)
    parser.add_argument(
        "--ranking-target-pattern",
        default="dipeptidyl peptidase 4",
        help="Target-name substring or exact UniProt ID for the stability ranking.",
    )
    parser.add_argument(
        "--ranking-output-name",
        default="dpp4_unique_compound_stability_ranking_2773.csv",
    )
    parser.add_argument(
        "--report-title", default="MGCA-2773 four-target case-study ensemble"
    )
    parser.add_argument(
        "--protocol",
        default=(
            "mgca_2773_full7_five_seed_four_target_exact_fasta_unseen_case_inference"
        ),
    )
    parser.add_argument("--expected-heldout-uniprot", default="")
    parser.add_argument("--prior-selection-exposure", action="store_true")
    return parser.parse_args()


def checkpoint_paths_2773(args: argparse.Namespace) -> list[dict]:
    records = []
    for seed in args.seeds:
        run_dir = args.checkpoint_root / f"seed_{seed}"
        checkpoint = run_dir / f"mgca_full_seed_{seed}_epoch{args.epochs}.pt"
        complete = run_dir / ".complete"
        manifest = run_dir / "run_manifest.json"
        for path in (checkpoint, complete, manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoint_hash = base.sha256_file(checkpoint)
        marker_hash = complete.read_text(encoding="utf-8").strip()
        if marker_hash != checkpoint_hash:
            raise RuntimeError(f"Completion hash mismatch for seed {seed}: {checkpoint}")
        run_manifest = base.read_json(manifest)
        identity = run_manifest.get("run_identity", {})
        if identity.get("dataset") != "2773":
            raise RuntimeError(f"Run manifest is not a 2773 refit: {manifest}")
        if int(identity.get("epochs", -1)) != args.epochs:
            raise RuntimeError(f"Run-manifest epoch mismatch: {manifest}")
        records.append(
            {
                "seed": seed,
                "epochs": args.epochs,
                "path": checkpoint,
                "sha256": checkpoint_hash,
                "run_manifest": manifest,
                "run_manifest_sha256": base.sha256_file(manifest),
            }
        )
    return records


def validate_checkpoint_2773(
    checkpoint: dict,
    record: dict,
    expected_config_id: str,
    reference_config: dict | None,
    reference_data_hash: str | None,
    expected_heldout_uniprot: str = "",
) -> tuple[dict, str]:
    config = checkpoint.get("config", {})
    data = checkpoint.get("data", {})
    code = checkpoint.get("code", {})
    if config.get("config_id") != expected_config_id:
        raise RuntimeError(
            f"Seed {record['seed']} config mismatch: {config.get('config_id')!r}"
        )
    if int(config.get("seed", -1)) != record["seed"]:
        raise RuntimeError(f"Seed metadata mismatch in {record['path']}")
    if int(config.get("epochs", -1)) != record["epochs"]:
        raise RuntimeError(f"Checkpoint epoch mismatch: {record['path']}")
    if data.get("case_exact_fasta_overlap_count") != 0:
        raise RuntimeError(f"Checkpoint training data contains a case target: {record['path']}")
    if code.get("corrected_entry_sha256") != base.sha256_file(base.CORRECTED_ENTRY):
        raise RuntimeError(f"Corrected model code mismatch: {record['path']}")
    if code.get("legacy_entry_sha256") != base.sha256_file(base.LEGACY_ENTRY):
        raise RuntimeError(f"Legacy model code mismatch: {record['path']}")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError(f"Missing model state: {record['path']}")

    if expected_heldout_uniprot:
        manifest_value = data.get("manifest_path")
        if not manifest_value:
            raise RuntimeError(f"Checkpoint has no training manifest: {record['path']}")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        training_manifest = base.read_json(manifest_path)
        heldout = training_manifest.get("heldout_target", {})
        actual_uniprot = str(heldout.get("uniprot_id", "")).upper()
        if actual_uniprot != expected_heldout_uniprot.upper():
            raise RuntimeError(
                "Training-manifest held-out target mismatch: "
                f"expected={expected_heldout_uniprot}, actual={actual_uniprot!r}"
            )
        if training_manifest.get("heldout_exact_fasta_overlap_count") != 0:
            raise RuntimeError(f"Training data contain held-out target: {manifest_path}")
        if training_manifest.get("output_sha256") != data.get("csv_sha256"):
            raise RuntimeError(f"Training-manifest hash mismatch: {record['path']}")

    identity_keys = [
        "proj_dim1",
        "proj_dim2",
        "hidden_dim",
        "dropout",
        "nums_of_experts",
        "num_heads",
        "moe_num_experts",
        "ablation",
        "config_id",
        "model_variant",
        "fingerprint_type",
        "window_size",
        "window_layout",
        "lr",
        "weight_decay",
        "batch_size",
        "epochs",
    ]
    comparable = {key: config.get(key) for key in identity_keys}
    if reference_config is not None and comparable != reference_config:
        raise RuntimeError(f"Checkpoint configuration differs across seeds: {record['path']}")
    data_hash = data.get("csv_sha256")
    if reference_data_hash is not None and data_hash != reference_data_hash:
        raise RuntimeError(f"Checkpoint training data differs across seeds: {record['path']}")
    return comparable, data_hash


def main() -> None:
    base.__file__ = __file__
    base.EXPECTED_CONFIG_ID = EXPECTED_CONFIG_ID_2773
    base.parse_args = parse_args_2773
    base.checkpoint_paths = checkpoint_paths_2773
    base.validate_checkpoint = validate_checkpoint_2773
    base.main()


if __name__ == "__main__":
    main()
