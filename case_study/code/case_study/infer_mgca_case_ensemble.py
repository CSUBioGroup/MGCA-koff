#!/usr/bin/env python3
"""Run five-checkpoint MGCA case inference and produce audited summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTED_ENTRY = (
    PROJECT_ROOT / "mgca_hyperparameter_tuning" / "ESM_Morgan_Hybrid_Fusion_nonredundant.py"
)
LEGACY_ENTRY = PROJECT_ROOT / "local" / "ESM_Morgan_Hybrid_Fusion.py"
EXPECTED_CONFIG_ID = "69590ecd8276"
T_CRITICAL_95_DF4 = 2.7764451051977987


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def torch_load_full_compat(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)


def load_corrected_module():
    spec = importlib.util.spec_from_file_location("mgca_corrected_case_infer", CORRECTED_ENTRY)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import corrected MGCA entry: {CORRECTED_ENTRY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242, 342, 442])
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--esm-batch-size", type=int, default=1)
    parser.add_argument("--batch-invariance-tolerance", type=float, default=1e-5)
    parser.add_argument("--expected-config-id", default=EXPECTED_CONFIG_ID)
    parser.add_argument(
        "--ranking-target-pattern",
        default="dipeptidyl peptidase 4",
        help="Case-insensitive target-name substring or exact UniProt ID used for the stability ranking.",
    )
    parser.add_argument(
        "--ranking-output-name",
        default="dpp4_unique_compound_stability_ranking.csv",
    )
    parser.add_argument(
        "--report-title", default="MGCA four-target case-study ensemble"
    )
    parser.add_argument(
        "--protocol",
        default="mgca_full36_five_seed_four_target_retrospective_case_inference",
    )
    parser.add_argument(
        "--expected-heldout-uniprot",
        default="",
        help="If set, require both the case rows and training manifest to record this held-out UniProt ID.",
    )
    parser.add_argument(
        "--prior-selection-exposure",
        action="store_true",
        help="Record that case labels occurred in the earlier global hyperparameter-selection cohort.",
    )
    return parser.parse_args()


def read_case_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Case CSV has no header: {path}")
        lookup = {name.lower(): name for name in reader.fieldnames}
        required = ["target_name", "uniprot_id", "fasta", "smiles", "pkoff"]
        missing = [name for name in required if name not in lookup]
        if missing:
            raise ValueError(f"Missing case columns {missing}: {path}")

        rows = []
        for index, source in enumerate(reader):
            fasta = "".join(source[lookup["fasta"]].split()).upper()
            smiles = source[lookup["smiles"]].strip()
            observed = float(source[lookup["pkoff"]])
            rows.append(
                {
                    "sample_id": f"case_{index + 1:04d}",
                    "row_index": index,
                    "target_name": source[lookup["target_name"]].strip(),
                    "uniprot_id": source[lookup["uniprot_id"]].strip(),
                    "category": source.get(lookup.get("category", ""), "").strip(),
                    "source": source.get(lookup.get("source", ""), "").strip(),
                    "fasta": fasta,
                    "fasta_sha256": sha256_text(fasta),
                    "smiles": smiles,
                    "observed_pkoff": observed,
                }
            )
    if not rows:
        raise ValueError(f"Case CSV is empty: {path}")
    return rows


def checkpoint_paths(args: argparse.Namespace) -> list[dict]:
    records = []
    for seed in args.seeds:
        run_dir = args.checkpoint_root / f"seed_{seed}"
        checkpoint = run_dir / f"mgca_full_seed_{seed}_epoch{args.epochs}.pt"
        complete = run_dir / ".complete"
        manifest = run_dir / "run_manifest.json"
        for path in (checkpoint, complete, manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoint_hash = sha256_file(checkpoint)
        marker_hash = complete.read_text(encoding="utf-8").strip()
        if marker_hash != checkpoint_hash:
            raise RuntimeError(f"Completion hash mismatch for seed {seed}: {checkpoint}")
        records.append(
            {
                "seed": seed,
                "path": checkpoint,
                "sha256": checkpoint_hash,
                "run_manifest": manifest,
                "run_manifest_sha256": sha256_file(manifest),
            }
        )
    return records


def validate_checkpoint(
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
    if config.get("epochs") != 36:
        raise RuntimeError(f"Expected a 36-epoch checkpoint: {record['path']}")
    if data.get("case_exact_fasta_overlap_count") != 0:
        raise RuntimeError(f"Checkpoint training data contains a case target: {record['path']}")
    if code.get("corrected_entry_sha256") != sha256_file(CORRECTED_ENTRY):
        raise RuntimeError(f"Corrected model code mismatch: {record['path']}")
    if code.get("legacy_entry_sha256") != sha256_file(LEGACY_ENTRY):
        raise RuntimeError(f"Legacy model code mismatch: {record['path']}")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError(f"Missing model state: {record['path']}")

    if expected_heldout_uniprot:
        manifest_value = data.get("manifest_path")
        if not manifest_value:
            raise RuntimeError(
                f"Checkpoint does not record a training manifest: {record['path']}"
            )
        manifest_path = Path(manifest_value)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        training_manifest = read_json(manifest_path)
        heldout = training_manifest.get("heldout_target", {})
        actual_uniprot = str(heldout.get("uniprot_id", "")).upper()
        if actual_uniprot != expected_heldout_uniprot.upper():
            raise RuntimeError(
                "Training-manifest held-out target mismatch: "
                f"expected={expected_heldout_uniprot}, actual={actual_uniprot!r}"
            )
        if training_manifest.get("heldout_exact_fasta_overlap_count") != 0:
            raise RuntimeError(
                f"Training data contain the held-out target: {manifest_path}"
            )
        if training_manifest.get("output_sha256") != data.get("csv_sha256"):
            raise RuntimeError(
                f"Checkpoint/training-manifest data hash mismatch: {record['path']}"
            )

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


def canonicalize_smiles(legacy, smiles: str) -> str:
    molecule = legacy.Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid case SMILES: {smiles}")
    return legacy.Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def prepare_case_features(
    metadata: list[dict],
    legacy,
    esm2_path: Path,
    device,
    output_dir: Path,
    case_hash: str,
    config: dict,
    esm_batch_size: int,
):
    unique_fastas = list(OrderedDict((row["fasta"], None) for row in metadata))
    fasta_to_index = {fasta: index for index, fasta in enumerate(unique_fastas)}
    window_size = int(config["window_size"])
    window_layout = config["window_layout"]
    cache_path = output_dir / "cache" / (
        f"case_unique_fasta_{case_hash[:12]}__ws{window_size}__wl{window_layout}.pt"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    unique_features = None
    if cache_path.is_file():
        cached = torch_load_full_compat(cache_path, map_location="cpu")
        if tuple(cached.shape) == (len(unique_fastas), 4, int(config["proj_dim1"])):
            unique_features = cached
        else:
            raise RuntimeError(f"Unexpected case ESM cache shape: {tuple(cached.shape)}")

    if unique_features is None:
        print(f"Extracting ESM2 features for {len(unique_fastas)} unique case targets...")
        tokenizer = legacy.AutoTokenizer.from_pretrained(str(esm2_path))
        esm_model = legacy.AutoModelForMaskedLM.from_pretrained(str(esm2_path)).to(device)
        esm_model.eval()
        with torch.no_grad():
            unique_features = legacy.batch_extract_esm2(
                unique_fastas,
                tokenizer,
                esm_model,
                device,
                batch_size=esm_batch_size,
                window_size=window_size,
                window_layout=window_layout,
            ).cpu()
        cache_temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(unique_features, cache_temporary)
        os.replace(cache_temporary, cache_path)
        del esm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    row_indices = torch.tensor(
        [fasta_to_index[row["fasta"]] for row in metadata], dtype=torch.long
    )
    fasta_features = unique_features.index_select(0, row_indices).to(device)

    molecules = [legacy.Chem.MolFromSmiles(row["smiles"]) for row in metadata]
    if any(molecule is None for molecule in molecules):
        raise ValueError("At least one case SMILES could not be parsed by RDKit")
    smiles_features = []
    for radius in range(4):
        fingerprints, valid = legacy.get_fingerprint(
            radius, molecules, device=device, fingerprint_type=config["fingerprint_type"]
        )
        if not bool(valid.all().item()):
            raise RuntimeError(f"Invalid Morgan fingerprint at radius {radius}")
        smiles_features.append(fingerprints.unsqueeze(1))
    smiles_features = torch.cat(smiles_features, dim=1)
    return fasta_features, smiles_features, cache_path, unique_fastas


def predict_model(model, fasta, smiles, batch_size: int, collect_aux: bool):
    predictions = []
    aux_chunks = {
        "protein_gate": [],
        "drug_gate": [],
        "p2d": [],
        "d2p": [],
        "branch_scale": [],
        "moe_router": [],
    }
    hook = None
    if collect_aux:
        def capture_moe(_module, _inputs, outputs):
            aux_chunks["moe_router"].append(outputs[1].detach().cpu().numpy())

        hook = model.moe.register_forward_hook(capture_moe)

    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(fasta), batch_size):
                stop = min(start + batch_size, len(fasta))
                output, aux = model(fasta[start:stop].float(), smiles[start:stop].float())
                predictions.append(output.reshape(-1).detach().cpu().numpy())
                if collect_aux:
                    protein_gate, drug_gate, p2d, d2p = aux
                    aux_chunks["protein_gate"].append(protein_gate.detach().cpu().numpy())
                    aux_chunks["drug_gate"].append(drug_gate.detach().cpu().numpy())
                    aux_chunks["p2d"].append(p2d.detach().cpu().numpy())
                    aux_chunks["d2p"].append(d2p.detach().cpu().numpy())
                    aux_chunks["branch_scale"].append(
                        model.last_branch_scales.detach().cpu().numpy()
                    )
    finally:
        if hook is not None:
            hook.remove()

    prediction_array = np.concatenate(predictions)
    if not collect_aux:
        return prediction_array, None
    return prediction_array, {
        key: np.concatenate(chunks, axis=0) for key, chunks in aux_chunks.items()
    }


def model_kwargs_from_config(config: dict) -> dict:
    keys = [
        "proj_dim1",
        "proj_dim2",
        "hidden_dim",
        "dropout",
        "nums_of_experts",
        "num_heads",
        "moe_num_experts",
        "ablation",
    ]
    return {key: config[key] for key in keys}


def base_output_row(metadata: dict, canonical_smiles: str) -> dict:
    return {
        "sample_id": metadata["sample_id"],
        "row_index": metadata["row_index"],
        "target_name": metadata["target_name"],
        "uniprot_id": metadata["uniprot_id"],
        "category": metadata["category"],
        "source": metadata["source"],
        "fasta_sha256": metadata["fasta_sha256"],
        "smiles": metadata["smiles"],
        "canonical_smiles": canonical_smiles,
        "observed_pkoff": metadata["observed_pkoff"],
    }


def metric_record(legacy, level: str, target: str, observed, predicted) -> dict:
    metrics = legacy.compute_metrics(np.asarray(observed), np.asarray(predicted))
    return {
        "analysis_level": level,
        "target_name": target,
        "n": len(observed),
        **{key: float(value) for key, value in metrics.items()},
    }


def build_metrics(legacy, ensemble_rows: list[dict], unique_rows: list[dict]) -> list[dict]:
    output = []
    for level, rows in (("sample", ensemble_rows), ("unique_compound", unique_rows)):
        output.append(
            metric_record(
                legacy,
                level,
                "ALL",
                [row["observed_pkoff"] for row in rows],
                [row["predicted_pkoff_mean"] for row in rows],
            )
        )
        target_records = []
        for target in sorted({row["target_name"] for row in rows}):
            selected = [row for row in rows if row["target_name"] == target]
            record = metric_record(
                legacy,
                level,
                target,
                [row["observed_pkoff"] for row in selected],
                [row["predicted_pkoff_mean"] for row in selected],
            )
            target_records.append(record)
            output.append(record)
        metric_keys = ["mse", "rmse", "mae", "r2", "pearson", "spearman"]
        output.append(
            {
                "analysis_level": level,
                "target_name": "MACRO_TARGET_MEAN",
                "n": len(target_records),
                **{
                    key: float(np.mean([record[key] for record in target_records]))
                    for key in metric_keys
                },
            }
        )
    return output


def unique_compound_rows(ensemble_rows: list[dict], seeds: list[int]) -> list[dict]:
    groups = OrderedDict()
    for row in ensemble_rows:
        key = (row["target_name"], row["uniprot_id"], row["canonical_smiles"])
        groups.setdefault(key, []).append(row)

    output = []
    for compound_index, ((target, uniprot, canonical), rows) in enumerate(groups.items(), 1):
        observed = np.asarray([row["observed_pkoff"] for row in rows], dtype=float)
        seed_predictions = np.asarray(
            [np.mean([row[f"predicted_pkoff_seed_{seed}"] for row in rows]) for seed in seeds]
        )
        mean = float(seed_predictions.mean())
        sd = float(seed_predictions.std(ddof=1))
        half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(len(seeds))
        output.append(
            {
                "compound_id": f"compound_{compound_index:04d}",
                "target_name": target,
                "uniprot_id": uniprot,
                "category": rows[0]["category"],
                "canonical_smiles": canonical,
                "representative_smiles": rows[0]["smiles"],
                "n_measurements": len(rows),
                "observed_pkoff": float(observed.mean()),
                "observed_pkoff_sd": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
                "observed_pkoff_min": float(observed.min()),
                "observed_pkoff_max": float(observed.max()),
                **{
                    f"predicted_pkoff_seed_{seed}": float(value)
                    for seed, value in zip(seeds, seed_predictions)
                },
                "predicted_pkoff_mean": mean,
                "predicted_pkoff_sd_across_seeds": sd,
                "seed_mean_ci95_low": mean - half_width,
                "seed_mean_ci95_high": mean + half_width,
                "error": mean - float(observed.mean()),
                "absolute_error": abs(mean - float(observed.mean())),
            }
        )
    return output


def attribution_feature_names() -> list[str]:
    names = [f"protein_gate_expert_{index}" for index in range(1, 5)]
    names += [f"drug_gate_radius_{radius}" for radius in range(4)]
    names += ["branch_scale_protein", "branch_scale_drug", "branch_scale_cross"]
    names += [f"moe_router_{index}" for index in range(1, 3)]
    names += [f"p2d_protein_{i}_drug_{j}" for i in range(1, 5) for j in range(4)]
    names += [f"d2p_drug_{i}_protein_{j}" for i in range(4) for j in range(1, 5)]
    return names


def flatten_attribution(aux: dict) -> np.ndarray:
    return np.concatenate(
        [
            aux["protein_gate"],
            aux["drug_gate"],
            aux["branch_scale"],
            aux["moe_router"],
            aux["p2d"].reshape(len(aux["p2d"]), -1),
            aux["d2p"].reshape(len(aux["d2p"]), -1),
        ],
        axis=1,
    )


def attribution_summary_rows(
    metadata: list[dict], canonical_smiles: list[str], feature_matrix: np.ndarray
) -> list[dict]:
    feature_names = attribution_feature_names()
    output = []
    for target in sorted({row["target_name"] for row in metadata}):
        sample_indices = [i for i, row in enumerate(metadata) if row["target_name"] == target]
        sample_matrix = feature_matrix[sample_indices]
        compound_groups = OrderedDict()
        for index in sample_indices:
            compound_groups.setdefault(canonical_smiles[index], []).append(index)
        compound_matrix = np.stack(
            [feature_matrix[indices].mean(axis=0) for indices in compound_groups.values()]
        )
        for level, matrix in (("sample", sample_matrix), ("unique_compound", compound_matrix)):
            record = {
                "target_name": target,
                "aggregation_level": level,
                "n": len(matrix),
            }
            for column, name in enumerate(feature_names):
                record[f"{name}_mean"] = float(matrix[:, column].mean())
                record[f"{name}_sd"] = (
                    float(matrix[:, column].std(ddof=1)) if len(matrix) > 1 else 0.0
                )
            output.append(record)
    return output


def markdown_report(
    metrics: list[dict],
    unique_rows: list[dict],
    seeds: list[int],
    title: str,
    prior_selection_exposure: bool,
) -> str:
    selected = [row for row in metrics if row["target_name"] != "MACRO_TARGET_MEAN"]
    lines = [
        f"# {title}",
        "",
        f"Five-seed ensemble: {', '.join(map(str, seeds))}.",
        (
            "Case labels were excluded from the final refit, but occurred in the earlier "
            "global hyperparameter-selection cohort; this is not fully blind nested selection."
            if prior_selection_exposure
            else "Case labels were used only for retrospective evaluation after inference."
        ),
        "The 95% intervals quantify seed-mean uncertainty (t, df=4); they are not predictive intervals.",
        "",
        "| Level | Target | n | MSE | RMSE | MAE | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['analysis_level']} | {row['target_name']} | {row['n']} | "
            f"{row['mse']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} | "
            f"{row['pearson']:.4f} | {row['spearman']:.4f} |"
        )
    lines += [
        "",
        f"Unique target-compound pairs: {len(unique_rows)}.",
        "",
        "Interpretability outputs are expert-space summaries (ESM2 depth experts, Morgan radii,",
        "bidirectional expert attention, branch scales, and MoE routing). They are not residue- or",
        "atom-level structural contacts and must not be presented as such.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    for path in (args.case_csv, args.checkpoint_root, args.esm2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise ValueError("Exactly five distinct seeds are required")
    if args.esm_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("Batch sizes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    corrected = load_corrected_module()
    legacy = corrected.legacy
    metadata = read_case_metadata(args.case_csv)
    args.expected_heldout_uniprot = args.expected_heldout_uniprot.strip().upper()
    if args.expected_heldout_uniprot:
        case_uniprots = {row["uniprot_id"].upper() for row in metadata}
        if case_uniprots != {args.expected_heldout_uniprot}:
            raise RuntimeError(
                "Case UniProt mismatch: "
                f"expected only {args.expected_heldout_uniprot}, found {sorted(case_uniprots)}"
            )
    ranking_output = Path(args.ranking_output_name)
    if ranking_output.name != args.ranking_output_name or ranking_output.suffix.lower() != ".csv":
        raise ValueError("--ranking-output-name must be a plain .csv file name")
    case_hash = sha256_file(args.case_csv)
    records = checkpoint_paths(args)

    first_checkpoint = torch_load_full_compat(records[0]["path"], map_location="cpu")
    reference_config, reference_data_hash = validate_checkpoint(
        first_checkpoint,
        records[0],
        args.expected_config_id,
        reference_config=None,
        reference_data_hash=None,
        expected_heldout_uniprot=args.expected_heldout_uniprot,
    )
    full_config = first_checkpoint["config"]
    del first_checkpoint

    canonical_smiles = [canonicalize_smiles(legacy, row["smiles"]) for row in metadata]
    fasta_features, smiles_features, esm_cache, unique_fastas = prepare_case_features(
        metadata,
        legacy,
        args.esm2_path,
        device,
        args.output_dir,
        case_hash,
        full_config,
        args.esm_batch_size,
    )

    seed_predictions = []
    seed_attributions = []
    determinism = []
    model_kwargs = model_kwargs_from_config(full_config)
    for record in records:
        print(f"Inferring case set with seed {record['seed']}...")
        checkpoint = torch_load_full_compat(record["path"], map_location=device)
        validate_checkpoint(
            checkpoint,
            record,
            args.expected_config_id,
            reference_config,
            reference_data_hash,
            expected_heldout_uniprot=args.expected_heldout_uniprot,
        )
        model = corrected.FullRegressionTransformer(**model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        predictions, aux = predict_model(
            model, fasta_features, smiles_features, args.batch_size, collect_aux=True
        )
        batch_one_predictions, _ = predict_model(
            model, fasta_features, smiles_features, 1, collect_aux=False
        )
        max_difference = float(np.max(np.abs(predictions - batch_one_predictions)))
        if max_difference > args.batch_invariance_tolerance:
            raise RuntimeError(
                f"Seed {record['seed']} batch-invariance failure: {max_difference:.3e}"
            )
        determinism.append({"seed": record["seed"], "max_abs_difference": max_difference})
        seed_predictions.append(predictions)
        seed_attributions.append(flatten_attribution(aux))
        del checkpoint, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_matrix = np.stack(seed_predictions, axis=0)
    attribution_matrix = np.stack(seed_attributions, axis=0)
    attribution_mean = attribution_matrix.mean(axis=0)
    observed = np.asarray([row["observed_pkoff"] for row in metadata], dtype=float)

    per_seed_rows = []
    ensemble_rows = []
    for index, (meta, canonical) in enumerate(zip(metadata, canonical_smiles)):
        base = base_output_row(meta, canonical)
        values = prediction_matrix[:, index]
        for seed, value, record in zip(args.seeds, values, records):
            per_seed_rows.append(
                {
                    **base,
                    "seed": seed,
                    "predicted_pkoff": float(value),
                    "error": float(value - observed[index]),
                    "absolute_error": float(abs(value - observed[index])),
                    "checkpoint_sha256": record["sha256"],
                }
            )
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = T_CRITICAL_95_DF4 * sd / math.sqrt(len(args.seeds))
        ensemble_rows.append(
            {
                **base,
                **{
                    f"predicted_pkoff_seed_{seed}": float(value)
                    for seed, value in zip(args.seeds, values)
                },
                "predicted_pkoff_mean": mean,
                "predicted_pkoff_sd_across_seeds": sd,
                "seed_mean_ci95_low": mean - half_width,
                "seed_mean_ci95_high": mean + half_width,
                "error": mean - observed[index],
                "absolute_error": abs(mean - observed[index]),
            }
        )

    unique_rows = unique_compound_rows(ensemble_rows, args.seeds)
    ranking_pattern = args.ranking_target_pattern.strip().lower()
    if not ranking_pattern:
        raise ValueError("--ranking-target-pattern must not be empty")
    selected_unique_rows = [
        dict(row)
        for row in unique_rows
        if ranking_pattern in row["target_name"].lower()
        or ranking_pattern == row["uniprot_id"].lower()
    ]
    if not selected_unique_rows:
        raise RuntimeError(
            f"No unique-compound rows matched ranking target {args.ranking_target_pattern!r}"
        )
    selected_unique_rows.sort(
        key=lambda row: (
            row["predicted_pkoff_sd_across_seeds"],
            row["canonical_smiles"],
        )
    )
    for rank, row in enumerate(selected_unique_rows, 1):
        row["stability_rank"] = rank
    metrics = build_metrics(legacy, ensemble_rows, unique_rows)
    feature_names = attribution_feature_names()
    attribution_rows = []
    for index, (meta, canonical) in enumerate(zip(metadata, canonical_smiles)):
        attribution_rows.append(
            {
                **base_output_row(meta, canonical),
                **{
                    name: float(attribution_mean[index, column])
                    for column, name in enumerate(feature_names)
                },
            }
        )
    attribution_summary = attribution_summary_rows(
        metadata, canonical_smiles, attribution_mean
    )

    ranking_output_key = "selected_target_unique_compound_ranking"
    if (
        args.ranking_output_name == "dpp4_unique_compound_stability_ranking.csv"
        and ranking_pattern == "dipeptidyl peptidase 4"
    ):
        # Preserve the manifest/output key used by the original four-target run.
        ranking_output_key = "dpp4_unique_compound_ranking"

    output_paths = {
        "per_seed_predictions": args.output_dir / "per_seed_predictions.csv",
        "ensemble_predictions": args.output_dir / "ensemble_predictions.csv",
        "unique_compound_predictions": args.output_dir
        / "unique_compound_ensemble_predictions.csv",
        ranking_output_key: args.output_dir / args.ranking_output_name,
        "metrics": args.output_dir / "case_metrics.csv",
        "sample_attributions": args.output_dir / "sample_expert_attributions.csv",
        "target_attributions": args.output_dir / "target_expert_attribution_summary.csv",
        "report": args.output_dir / "case_summary.md",
    }
    atomic_write_csv(output_paths["per_seed_predictions"], per_seed_rows)
    atomic_write_csv(output_paths["ensemble_predictions"], ensemble_rows)
    atomic_write_csv(output_paths["unique_compound_predictions"], unique_rows)
    atomic_write_csv(
        output_paths[ranking_output_key], selected_unique_rows
    )
    atomic_write_csv(output_paths["metrics"], metrics)
    atomic_write_csv(output_paths["sample_attributions"], attribution_rows)
    atomic_write_csv(output_paths["target_attributions"], attribution_summary)
    atomic_write_text(
        output_paths["report"],
        markdown_report(
            metrics,
            unique_rows,
            args.seeds,
            args.report_title,
            args.prior_selection_exposure,
        ),
    )

    manifest = {
        "protocol": args.protocol,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_id": args.expected_config_id,
        "case_csv": str(args.case_csv.resolve()),
        "case_csv_sha256": case_hash,
        "case_rows": len(metadata),
        "unique_case_targets": len(unique_fastas),
        "unique_target_compound_pairs": len(unique_rows),
        "training_data_sha256": reference_data_hash,
        "code": {
            "inference_script": str(Path(__file__).resolve()),
            "inference_script_sha256": sha256_file(Path(__file__).resolve()),
            "corrected_entry_sha256": sha256_file(CORRECTED_ENTRY),
            "legacy_entry_sha256": sha256_file(LEGACY_ENTRY),
        },
        "seeds": args.seeds,
        "checkpoints": [
            {
                "seed": record["seed"],
                "path": str(record["path"].resolve()),
                "sha256": record["sha256"],
                "run_manifest_sha256": record["run_manifest_sha256"],
            }
            for record in records
        ],
        "esm2_path": str(args.esm2_path.resolve()),
        "esm_unique_fasta_cache": str(esm_cache.resolve()),
        "esm_window_size": full_config["window_size"],
        "esm_window_layout": full_config["window_layout"],
        "esm_tokenizer_max_length": 1024,
        "unique_fasta_lengths": [len(fasta) for fasta in unique_fastas],
        "batch_invariance_tolerance": args.batch_invariance_tolerance,
        "batch_invariance_checks": determinism,
        "seed_uncertainty": {
            "sd_ddof": 1,
            "ci95_method": "two-sided Student t interval for the five-seed mean",
            "t_critical_df4": T_CRITICAL_95_DF4,
            "not_a_predictive_interval": True,
        },
        "case_labels_used_for_final_refit": False,
        "case_labels_present_in_prior_hyperparameter_selection_cohort": (
            args.prior_selection_exposure
        ),
        "case_labels_used_for_training_or_selection": args.prior_selection_exposure,
        "expected_heldout_uniprot": args.expected_heldout_uniprot or None,
        "ranking_target_pattern": args.ranking_target_pattern,
        "attribution_scope": "expert_space_not_residue_or_atom_contacts",
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in output_paths.items()
        },
    }
    manifest_path = args.output_dir / "case_inference_manifest.json"
    atomic_write_json(manifest_path, manifest)
    (args.output_dir / ".complete").write_text(
        sha256_file(manifest_path) + "\n", encoding="utf-8"
    )
    print(f"Case inference completed: {args.output_dir}")
    print(f"Summary: {output_paths['report']}")


if __name__ == "__main__":
    main()
