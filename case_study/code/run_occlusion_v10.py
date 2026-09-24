#!/usr/bin/env python3
"""Run the established post-ESM occlusion workflow with v10 checkpoints.

This adapter deliberately reuses the audited perturbation implementation in
case_study/factor_xa_stage2_occlusion.py while replacing every v4-specific
checkpoint/model assumption with MGCA-CGRS v10 equivalents.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

import common_v10 as common


def adapter_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-file", type=Path, default=common.V10_MODEL)
    parser.add_argument("--compound-id-prefix", default="factor_xa")
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


def main():
    adapter, remaining = adapter_args(sys.argv[1:])
    case_dir = common.PROJECT_ROOT / "case_study"
    sys.path.insert(0, str(case_dir))
    import factor_xa_stage2_occlusion as occlusion

    if not adapter.model_file.is_file():
        raise FileNotFoundError(adapter.model_file)
    occlusion.DATASET_DEFAULTS["2773"] = {
        "epochs": common.DEFAULT_EPOCHS,
        "config_id": common.EXPECTED_CONFIG_ID,
    }
    occlusion.base.load_corrected_module = lambda: common.load_v10_module(adapter.model_file)

    def records(args):
        return common.checkpoint_records(args.checkpoint_root, args.seeds, args.epochs)

    def validator(args, payload, record, reference_config, reference_data_hash):
        data_hash = common.validate_checkpoint(
            payload, record, reference_data_hash, args.epochs
        )
        config = payload["config"]
        comparable = {key: value for key, value in config.items() if key != "seed"}
        if reference_config is not None and comparable != reference_config:
            raise RuntimeError(f"Configuration differs across seeds: {record['path']}")
        return comparable, data_hash

    def model_kwargs(config):
        keys = (
            "proj_dim1", "proj_dim2", "hidden_dim", "dropout", "nums_of_experts",
            "ablation", "joint_rank", "protein_aux_weight", "drug_utility_weight",
            "joint_utility_weight", "branch_margin",
            "drug_gate_init", "joint_gate_init",
            "joint_branch_dropout",
        )
        return {key: config[key] for key in keys}

    occlusion.checkpoint_records = records
    occlusion.validate_checkpoint = validator
    occlusion.base.model_kwargs_from_config = model_kwargs

    original_loader = occlusion.load_factor_xa_compounds

    def target_loader(path, legacy, uniprot):
        compounds, fasta = original_loader(path, legacy, uniprot)
        for index, compound in enumerate(compounds, 1):
            compound["compound_id"] = f"{adapter.compound_id_prefix}_{index:02d}"
        return compounds, fasta

    occlusion.load_factor_xa_compounds = target_loader
    sys.argv = [sys.argv[0], *remaining]
    parsed = occlusion.parse_args()
    occlusion.resolve_settings(parsed)
    common.check_case_settings(parsed.epochs)
    if tuple(parsed.seeds) != common.DEFAULT_SEEDS: raise RuntimeError('Frozen seed list changed')

    def generic_reference_audit(path, original_rows, seeds, tolerance):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reference = [
                row for row in csv.DictReader(handle)
                if str(row.get("uniprot_id", "")).upper() == parsed.target_uniprot.upper()
            ]
        lookup = {
            (row["canonical_smiles"], int(row["seed"])): float(row["predicted_pkoff"])
            for row in original_rows
        }
        differences = []
        matched = 0
        for row in reference:
            canonical = row.get("canonical_smiles", "")
            for seed in seeds:
                column = f"predicted_pkoff_seed_{seed}"
                if column in row and (canonical, seed) in lookup:
                    differences.append(abs(lookup[(canonical, seed)] - float(row[column])))
                    matched += 1
        if matched != len(lookup):
            raise RuntimeError(
                f"Stage-1 prediction audit coverage mismatch: matched={matched}, expected={len(lookup)}"
            )
        maximum = max(differences, default=0.0)
        if maximum > tolerance:
            raise RuntimeError(
                f"Stage-1 no-occlusion prediction audit failed: max difference={maximum:.3e}"
            )
        return {
            "reference_path": str(path.resolve()),
            "reference_sha256": common.sha256_file(path),
            "target_uniprot": parsed.target_uniprot,
            "matched_predictions": matched,
            "maximum_absolute_difference": maximum,
            "tolerance": tolerance,
            "passed": True,
        }

    occlusion.audit_original_predictions = generic_reference_audit
    records_for_identity = common.checkpoint_records(
        parsed.checkpoint_root, parsed.seeds, parsed.epochs
    )
    identity = {
        "protocol": common.PROTOCOL+'_occlusion',
        "package_sha256": common.sha256_file(common.PROJECT_ROOT/'release_manifest.json'),
        "case_csv_sha256": common.sha256_file(parsed.case_csv),
        "model_sha256": common.sha256_file(adapter.model_file),
        "adapter_sha256": common.sha256_file(Path(__file__).resolve()),
        "legacy_occlusion_sha256": common.sha256_file(
            common.PROJECT_ROOT / "case_study" / "factor_xa_stage2_occlusion.py"
        ),
        "target_uniprot": parsed.target_uniprot,
        "compound_id_prefix": adapter.compound_id_prefix,
        "representative_compound_id": parsed.representative_compound_id,
        "reference_predictions_sha256": (
            common.sha256_file(parsed.reference_predictions)
            if parsed.reference_predictions is not None else None
        ),
        "seeds": parsed.seeds,
        "epochs": parsed.epochs,
        "checkpoint_sha256": [record["sha256"] for record in records_for_identity],
        "protein_window_size": parsed.protein_window_size,
        "protein_stride": parsed.protein_stride,
        "sensitivity_window_sizes": parsed.sensitivity_window_sizes,
        "protein_baseline": parsed.protein_baseline,
        "top_features": parsed.top_features,
    }
    complete = parsed.output_dir / ".complete"
    audit = parsed.output_dir / "v10_occlusion_audit.json"
    if complete.is_file() and audit.is_file():
        existing = common.read_json(audit)
        if (
            existing.get("run_identity") == identity
            and complete.read_text(encoding="utf-8").strip() == common.sha256_file(audit)
            and all((parsed.output_dir/name).is_file() and common.sha256_file(parsed.output_dir/name)==digest
                    for name,digest in existing.get('files',{}).items())
            and bool(existing.get('files'))
        ):
            print(f"Matching completed occlusion exists; skipping {parsed.output_dir}")
            return
        raise RuntimeError(f"Stale occlusion completion marker: {complete}")
    occlusion.main()
    manifest = parsed.output_dir / "factor_xa_stage2_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"Occlusion manifest was not created: {manifest}")
    manifest_payload = common.read_json(manifest)
    manifest_payload["legacy_core_protocol"] = manifest_payload.get("protocol")
    manifest_payload["protocol"] = common.PROTOCOL+'_occlusion'
    manifest_payload["analysis_label"] = adapter.compound_id_prefix
    manifest_payload["model_variant"] = common.MODEL_VARIANT
    manifest_payload["model_sha256"] = common.sha256_file(adapter.model_file)
    manifest_payload["interpretation_boundary"] = (
        "Post-ESM feature occlusion sensitivity under v10; not native residue-atom "
        "attention, a sample-wise final router, or physical interaction energy."
    )
    common.atomic_json(manifest, manifest_payload)
    common.atomic_json(
        audit,
        {
            "run_identity": identity,
            "files": {p.name:common.sha256_file(p) for p in parsed.output_dir.iterdir()
                      if p.is_file() and p.suffix in {'.json','.csv'} and p != audit},
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "occlusion_manifest": str(manifest.resolve()),
            "occlusion_manifest_sha256": common.sha256_file(manifest),
        },
    )
    common.atomic_text(complete, common.sha256_file(audit) + "\n")


if __name__ == "__main__":
    main()
