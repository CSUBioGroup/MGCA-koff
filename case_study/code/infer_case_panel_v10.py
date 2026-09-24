#!/usr/bin/env python3
"""Infer a labeled target panel with five frozen final MGCA checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import common_v10 as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--training-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, default=common.V10_MODEL)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=common.DEFAULT_EPOCHS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--esm-batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--kmer-size", type=int, default=5)
    parser.add_argument("--sequence-threshold", type=float, default=0.70)
    parser.add_argument("--sequence-margin", type=float, default=0.05)
    parser.add_argument("--check-batch-invariance", action="store_true")
    return parser.parse_args()


def read_training_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        lookup = {name.lower(): name for name in (reader.fieldnames or [])}
        if "fasta" not in lookup or "smiles" not in lookup:
            raise ValueError(f"Training CSV lacks FASTA/SMILES: {path}")
        return [
            {
                "fasta": "".join(row[lookup["fasta"]].split()).upper(),
                "smiles": row[lookup["smiles"]].strip(),
            }
            for row in reader
        ]


def kmer_set(sequence: str, size: int):
    if len(sequence) < size:
        return {sequence}
    return {sequence[index:index + size] for index in range(len(sequence) - size + 1)}


def shorter_coverage(left: str, right: str, size: int) -> float:
    left_set, right_set = kmer_set(left, size), kmer_set(right, size)
    denominator = min(len(left_set), len(right_set))
    return len(left_set & right_set) / denominator if denominator else 0.0


def training_exposure(metadata, training_rows, legacy, args):
    target_sequences = OrderedDict()
    for row in metadata:
        target_sequences.setdefault((row["uniprot_id"], row["target_name"]), set()).add(row["fasta"])
    compound_sets = {key: set() for key in target_sequences}
    audit_rows = []
    grouped = defaultdict(list)
    for row in training_rows:
        grouped[row["fasta"]].append(row)
    for train_fasta, members in grouped.items():
        scores = []
        for key, sequences in target_sequences.items():
            exact = train_fasta in sequences
            substring = any(train_fasta in sequence or sequence in train_fasta for sequence in sequences)
            score = max(shorter_coverage(train_fasta, sequence, args.kmer_size) for sequence in sequences)
            scores.append((key, exact, substring, score))
        scores.sort(key=lambda item: item[3], reverse=True)
        best = scores[0]
        second = scores[1][3] if len(scores) > 1 else 0.0
        assigned = best[1] or best[2] or (
            best[3] >= args.sequence_threshold and best[3] - second >= args.sequence_margin
        )
        if assigned:
            for member in members:
                compound_sets[best[0]].add(common.canonical_smiles(legacy, member["smiles"]))
        if assigned or best[3] >= args.sequence_threshold * 0.75:
            audit_rows.append(
                {
                    "assigned": int(assigned),
                    "assigned_uniprot_id": best[0][0] if assigned else "",
                    "assigned_target_name": best[0][1] if assigned else "",
                    "best_uniprot_id": best[0][0],
                    "best_target_name": best[0][1],
                    "exact_match": int(best[1]),
                    "substring_match": int(best[2]),
                    "best_shorter_kmer_coverage": best[3],
                    "second_best_coverage": second,
                    "score_margin": best[3] - second,
                    "training_rows": len(members),
                    "training_fasta_length": len(train_fasta),
                    "training_fasta_sha256": common.sha256_text(train_fasta),
                }
            )
    return compound_sets, audit_rows


def load_models_and_predict(args, metadata, protein, ligand, mgca, records):
    predictions = {}
    diagnostics = []
    reference_data_hash = None
    reference_config = None
    invariance = []
    device = torch.device(args.device)
    for record in records:
        payload = common.torch_load(torch, record["path"], map_location=device)
        data_hash = common.validate_checkpoint(payload, record, reference_data_hash, args.epochs)
        reference_data_hash = data_hash if reference_data_hash is None else reference_data_hash
        config = payload["config"]
        comparable = {key: value for key, value in config.items() if key != "seed"}
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            raise RuntimeError(f"Checkpoint configuration differs across seeds: {record['path']}")
        model_keys = (
            "proj_dim1", "proj_dim2", "hidden_dim", "dropout", "nums_of_experts",
            "ablation", "joint_rank", "protein_aux_weight", "drug_utility_weight",
            "joint_utility_weight", "branch_margin",
            "drug_gate_init", "joint_gate_init",
            "joint_branch_dropout",
        )
        kwargs = {key: config[key] for key in model_keys}
        model = mgca.FullRegressionTransformer(**kwargs).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        values, aux = common.predict_with_aux(model, protein, ligand, device, args.batch_size)
        if args.check_batch_invariance:
            values_one, _ = common.predict_with_aux(model, protein, ligand, device, 1)
            difference = float(np.max(np.abs(values - values_one)))
            if difference > 1e-5:
                raise RuntimeError(f"Batch invariance failed for seed {record['seed']}: {difference}")
            invariance.append({"seed": record["seed"], "max_abs_difference": difference})
        predictions[record["seed"]] = values
        for index, meta in enumerate(metadata):
            row = {
                "sample_id": meta["sample_id"],
                "target_name": meta["target_name"],
                "uniprot_id": meta["uniprot_id"],
                "seed": record["seed"],
                "protein_prediction": float(aux["protein_prediction"][index].reshape(-1)[0]),
                "drug_candidate_prediction": float(aux["drug_candidate_prediction"][index].reshape(-1)[0]),
                "joint_candidate_prediction": float(aux["joint_candidate_prediction"][index].reshape(-1)[0]),
                "drug_gate": float(aux["drug_gate"][index].reshape(-1)[0]),
                "joint_gate": float(aux["joint_gate"][index].reshape(-1)[0]),
                "effective_drug_gate": float(aux["effective_drug_gate"][index].reshape(-1)[0]),
                "effective_joint_gate": float(aux["effective_joint_gate"][index].reshape(-1)[0]),
                "drug_contribution_rms": float(aux["drug_contribution_rms"][index].reshape(-1)[0]),
                "joint_contribution_rms": float(aux["joint_contribution_rms"][index].reshape(-1)[0]),
                "joint_attention_confidence": float(aux["joint_attention_confidence"][index].reshape(-1)[0]),
                "protein_expert_weights": json.dumps(aux["protein_expert_weights"][index].tolist(), separators=(",", ":")),
                "drug_expert_weights": json.dumps(aux["drug_expert_weights"][index].tolist(), separators=(",", ":")),
                "attention_p2d_4x4": json.dumps(aux["attention_p2d"][index].tolist(), separators=(",", ":")),
                "attention_d2p_4x4": json.dumps(aux["attention_d2p"][index].tolist(), separators=(",", ":")),
                "checkpoint_sha256": record["sha256"],
            }
            row["drug_prediction_shift_signed"] = row["drug_candidate_prediction"] - row["protein_prediction"]
            row["joint_prediction_shift_signed"] = row["joint_candidate_prediction"] - row["drug_candidate_prediction"]
            diagnostics.append(row)
        del payload, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions, diagnostics, reference_data_hash, reference_config, invariance


def build_sample_rows(metadata, canonical_values, predictions, seeds):
    output = []
    for index, (meta, canonical) in enumerate(zip(metadata, canonical_values)):
        values = [float(predictions[seed][index]) for seed in seeds]
        mean, sd, low, high = common.mean_sd_ci(values)
        output.append(
            {
                "sample_id": meta["sample_id"],
                "row_index": meta["row_index"],
                "target_name": meta["target_name"],
                "uniprot_id": meta["uniprot_id"],
                "category": meta["category"],
                "source": meta["source"],
                "fasta_sha256": meta["fasta_sha256"],
                "smiles": meta["smiles"],
                "canonical_smiles": canonical,
                "observed_pkoff": meta["observed_pkoff"],
                **{f"predicted_pkoff_seed_{seed}": value for seed, value in zip(seeds, values)},
                "predicted_pkoff_mean": mean,
                "predicted_pkoff_sd_across_seeds": sd,
                "seed_mean_ci95_low": low,
                "seed_mean_ci95_high": high,
                "error": mean - meta["observed_pkoff"],
                "absolute_error": abs(mean - meta["observed_pkoff"]),
            }
        )
    return output


def aggregate_unique(sample_rows, seeds):
    groups = OrderedDict()
    for row in sample_rows:
        key = (row["target_name"], row["uniprot_id"], row["canonical_smiles"])
        groups.setdefault(key, []).append(row)
    output = []
    for index, ((target, uniprot, canonical), members) in enumerate(groups.items(), 1):
        observed = np.asarray([row["observed_pkoff"] for row in members], dtype=float)
        seed_values = [
            float(np.mean([row[f"predicted_pkoff_seed_{seed}"] for row in members]))
            for seed in seeds
        ]
        mean, sd, low, high = common.mean_sd_ci(seed_values)
        output.append(
            {
                "compound_id": f"compound_{index:04d}",
                "target_name": target,
                "uniprot_id": uniprot,
                "category": members[0]["category"],
                "canonical_smiles": canonical,
                "representative_smiles": members[0]["smiles"],
                "n_measurements": len(members),
                "observed_pkoff": float(observed.mean()),
                "observed_pkoff_sd": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
                "observed_pkoff_min": float(observed.min()),
                "observed_pkoff_max": float(observed.max()),
                **{f"predicted_pkoff_seed_{seed}": value for seed, value in zip(seeds, seed_values)},
                "predicted_pkoff_mean": mean,
                "predicted_pkoff_sd_across_seeds": sd,
                "seed_mean_ci95_low": low,
                "seed_mean_ci95_high": high,
                "error": mean - float(observed.mean()),
                "absolute_error": abs(mean - float(observed.mean())),
            }
        )
    return output


def build_metrics(rows, seeds):
    output = []
    groups = [("ALL", rows)] + [
        (target, [row for row in rows if row["target_name"] == target])
        for target in sorted({row["target_name"] for row in rows})
    ]
    for target, members in groups:
        observed = [row["observed_pkoff"] for row in members]
        for seed in seeds:
            metrics = common.regression_metrics(observed, [row[f"predicted_pkoff_seed_{seed}"] for row in members])
            output.append({"target_name": target, "evaluation": f"seed_{seed}", **metrics})
        metrics = common.regression_metrics(observed, [row["predicted_pkoff_mean"] for row in members])
        output.append({"target_name": target, "evaluation": "five_seed_ensemble", **metrics})
    return output


def build_rankings(unique_rows, seeds, exposure, legacy, top_k):
    ranking_rows, summary_rows = [], []
    for target_key in OrderedDict(((row["uniprot_id"], row["target_name"]), None) for row in unique_rows):
        uniprot, target = target_key
        members = [row for row in unique_rows if row["uniprot_id"] == uniprot and row["target_name"] == target]
        observed_ranked = sorted(members, key=lambda row: (-row["observed_pkoff"], row["canonical_smiles"]))
        observed_top = {k: {row["canonical_smiles"] for row in observed_ranked[:k]} for k in top_k}
        train_set = exposure.get(target_key, set())
        for label, column in [(f"seed_{seed}", f"predicted_pkoff_seed_{seed}") for seed in seeds] + [("ensemble", "predicted_pkoff_mean")]:
            ranked = sorted(members, key=lambda row: (-row[column], row["canonical_smiles"]))
            predicted_top = {k: {row["canonical_smiles"] for row in ranked[:k]} for k in top_k}
            cindex = common.concordance_index(
                [row["observed_pkoff"] for row in members], [row[column] for row in members]
            )
            summary_rows.append(
                {
                    "target_name": target,
                    "uniprot_id": uniprot,
                    "evaluation": label,
                    "All": len(members),
                    "Train": len({row["canonical_smiles"] for row in members} & train_set),
                    **{f"Top_{k}_overlap": len(predicted_top[k] & observed_top[k]) for k in top_k},
                    "c_index": cindex,
                }
            )
            for rank, row in enumerate(ranked, 1):
                ranking_rows.append(
                    {
                        "target_name": target,
                        "uniprot_id": uniprot,
                        "evaluation": label,
                        "predicted_rank": rank,
                        "predicted_pkoff": row[column],
                        "observed_pkoff": row["observed_pkoff"],
                        "canonical_smiles": row["canonical_smiles"],
                        "representative_smiles": row["representative_smiles"],
                        "is_Train": int(row["canonical_smiles"] in train_set),
                        **{f"is_predicted_Top_{k}": int(row["canonical_smiles"] in predicted_top[k]) for k in top_k},
                        **{f"is_experimental_Top_{k}": int(row["canonical_smiles"] in observed_top[k]) for k in top_k},
                    }
                )
    return ranking_rows, summary_rows


def aggregate_seed_metrics(metric_rows):
    output = []
    for target in sorted({row["target_name"] for row in metric_rows}):
        members = [
            row for row in metric_rows
            if row["target_name"] == target and row["evaluation"].startswith("seed_")
        ]
        record = {"target_name": target, "seeds": len(members), "n_per_seed": members[0]["n"]}
        for key in ("mse", "rmse", "mae", "r2", "pearson", "spearman", "kendall_tau", "c_index"):
            values = [float(row[key]) for row in members if row[key] is not None]
            record[f"{key}_mean"] = float(np.mean(values)) if values else None
            record[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
        output.append(record)
    return output


def aggregate_seed_rankings(ranking_summary, top_k):
    output = []
    keys = OrderedDict(
        ((row["uniprot_id"], row["target_name"]), None)
        for row in ranking_summary if row["evaluation"].startswith("seed_")
    )
    for uniprot, target in keys:
        members = [
            row for row in ranking_summary
            if row["uniprot_id"] == uniprot
            and row["target_name"] == target
            and row["evaluation"].startswith("seed_")
        ]
        record = {
            "target_name": target,
            "uniprot_id": uniprot,
            "seeds": len(members),
            "All": members[0]["All"],
            "Train": members[0]["Train"],
            "c_index_mean": float(np.mean([row["c_index"] for row in members])) if members[0]["c_index"] is not None else None,
            "c_index_sd": float(np.std([row["c_index"] for row in members], ddof=1)) if len(members) > 1 and members[0]["c_index"] is not None else None,
        }
        for k in top_k:
            values = [row[f"Top_{k}_overlap"] for row in members]
            record[f"Top_{k}_overlap_mean"] = float(np.mean(values))
            record[f"Top_{k}_overlap_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def main():
    args = parse_args()
    common.check_case_settings(args.epochs)
    if tuple(args.seeds) != common.DEFAULT_SEEDS:
        raise ValueError(f"Seeds must be exactly {common.DEFAULT_SEEDS}")
    for path in (args.case_csv, args.training_csv, args.checkpoint_root, args.model_file, args.esm2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mgca = common.load_v10_module(args.model_file)
    legacy = mgca.legacy
    metadata = common.read_case_rows(args.case_csv)
    records = common.checkpoint_records(args.checkpoint_root, args.seeds, args.epochs)
    run_identity = {
        "protocol": common.PROTOCOL+'_panel',
        "package_sha256": common.sha256_file(common.PROJECT_ROOT/'release_manifest.json'),
        "panel_name": args.panel_name,
        "case_csv_sha256": common.sha256_file(args.case_csv),
        "training_csv_sha256": common.sha256_file(args.training_csv),
        "model_sha256": common.sha256_file(args.model_file),
        "inference_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "epochs": args.epochs,
        "seeds": args.seeds,
        "checkpoint_sha256": [record["sha256"] for record in records],
        "batch_size": args.batch_size,
        "esm_batch_size": args.esm_batch_size,
        "top_k": args.top_k,
        "kmer_size": args.kmer_size,
        "sequence_threshold": args.sequence_threshold,
        "sequence_margin": args.sequence_margin,
    }
    complete_path = args.output_dir / ".complete"
    manifest_path = args.output_dir / "manifest.json"
    if complete_path.is_file() and manifest_path.is_file():
        existing = common.read_json(manifest_path)
        if (
            existing.get("run_identity") == run_identity
            and common.artifact_files_valid(args.output_dir, existing)
            and complete_path.read_text(encoding="utf-8").strip()
            == common.sha256_file(manifest_path)
        ):
            print(f"Matching completed case panel exists; skipping {args.output_dir}")
            return
        raise RuntimeError(f"Stale case-panel completion marker: {complete_path}")
    first = common.torch_load(torch, records[0]["path"], map_location="cpu")
    common.validate_checkpoint(first, records[0], epochs=args.epochs)
    checkpoint_config = first["config"]
    del first
    protein, ligand, esm_cache, unique_fastas = common.prepare_case_features(
        metadata, legacy, args.esm2_path, device, args.output_dir / "cache", checkpoint_config, torch, args.esm_batch_size
    )
    canonical_values = [common.canonical_smiles(legacy, row["smiles"]) for row in metadata]
    predictions, diagnostics, data_hash, model_config, invariance = load_models_and_predict(
        args, metadata, protein, ligand, mgca, records
    )
    if data_hash != common.sha256_file(args.training_csv):
        raise RuntimeError("Checkpoint training cohort does not match the exposure-audit cohort")
    sample_rows = build_sample_rows(metadata, canonical_values, predictions, args.seeds)
    unique_rows = aggregate_unique(sample_rows, args.seeds)
    metrics = build_metrics(unique_rows, args.seeds)
    metrics_mean_sd = aggregate_seed_metrics(metrics)
    training_rows = read_training_rows(args.training_csv)
    exposure, exposure_audit = training_exposure(metadata, training_rows, legacy, args)
    ranking_rows, ranking_summary = build_rankings(unique_rows, args.seeds, exposure, legacy, args.top_k)
    ranking_mean_sd = aggregate_seed_rankings(ranking_summary, args.top_k)
    paths = {
        "sample": args.output_dir / "sample_five_seed_predictions.csv",
        "unique": args.output_dir / "unique_compound_five_seed_predictions.csv",
        "metrics": args.output_dir / "metrics_by_target_and_seed.csv",
        "metrics_mean_sd": args.output_dir / "metrics_five_seed_mean_sd.csv",
        "rankings": args.output_dir / "compound_rankings.csv",
        "ranking_summary": args.output_dir / "ranking_summary.csv",
        "ranking_mean_sd": args.output_dir / "ranking_five_seed_mean_sd.csv",
        "diagnostics": args.output_dir / "v10_internal_diagnostics_per_seed.csv",
        "exposure": args.output_dir / "training_target_sequence_audit.csv",
    }
    common.atomic_csv(paths["sample"], sample_rows)
    common.atomic_csv(paths["unique"], unique_rows)
    common.atomic_csv(paths["metrics"], metrics)
    common.atomic_csv(paths["metrics_mean_sd"], metrics_mean_sd)
    common.atomic_csv(paths["rankings"], ranking_rows)
    common.atomic_csv(paths["ranking_summary"], ranking_summary)
    common.atomic_csv(paths["ranking_mean_sd"], ranking_mean_sd)
    common.atomic_csv(paths["diagnostics"], diagnostics)
    if exposure_audit:
        common.atomic_csv(paths["exposure"], exposure_audit)
    summary_lines = [
        f"# {args.panel_name}: frozen final MGCA trained on full 2773",
        "",
        f"Case rows: {len(metadata)}; unique compounds: {len(unique_rows)}; targets: {len(set(row['uniprot_id'] for row in metadata))}.",
        "Five checkpoints are ranked independently; ensemble rows are additional descriptive summaries.",
        "Train is audited from the actual full-refit CSV using exact/subsequence or guarded 5-mer matching.",
        "",
        "| Target | Evaluation | N | RMSE | MAE | Spearman | C-index |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if row["target_name"] != "ALL" and row["evaluation"] == "five_seed_ensemble":
            summary_lines.append(
                f"| {row['target_name']} | ensemble | {row['n']} | {row['rmse']:.4f} | {row['mae']:.4f} | "
                f"{row['spearman'] if row['spearman'] is not None else 'NA'} | {row['c_index'] if row['c_index'] is not None else 'NA'} |"
            )
    common.atomic_text(args.output_dir / "case_summary.md", "\n".join(summary_lines) + "\n")
    manifest = {
        "run_identity": run_identity,
        "protocol": common.PROTOCOL+'_panel',
        "panel_name": args.panel_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_csv": str(args.case_csv.resolve()),
        "case_csv_sha256": common.sha256_file(args.case_csv),
        "training_csv": str(args.training_csv.resolve()),
        "training_csv_sha256": common.sha256_file(args.training_csv),
        "training_data_sha256_from_checkpoints": data_hash,
        "model_config": model_config,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "unique_fastas": len(unique_fastas),
        "esm_cache": str(esm_cache.resolve()),
        "batch_invariance": invariance,
        "checkpoints": records,
        "outputs": {
            key: {"path": str(path.resolve()), "sha256": common.sha256_file(path)}
            for key, path in paths.items() if path.is_file()
        },
    }
    # Convert Path objects before strict JSON serialization.
    for item in manifest["checkpoints"]:
        for key in ("path", "manifest"):
            item[key] = str(item[key].resolve())
    common.atomic_json(args.output_dir / "manifest.json", manifest)
    complete_hash = common.sha256_file(args.output_dir / "manifest.json")
    common.atomic_text(args.output_dir / ".complete", complete_hash + "\n")
    print(f"Completed case panel: {args.panel_name} -> {args.output_dir}")


if __name__ == "__main__":
    main()
