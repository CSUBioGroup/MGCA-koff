#!/usr/bin/env python3
"""Stage-2 post-hoc occlusion analysis for the Factor Xa MGCA case study.

The script never updates model parameters.  For each of five independently
trained checkpoints it computes:

1. protein-window importance from pre-pooling ESM2 token representations;
2. ligand-feature importance by turning off one active (radius, Morgan bit);
3. second-order interactions for a pre-specified representative compound;
4. five-seed means, standard deviations, sign consistency, rank stability,
   and top-k overlap.

Protein occlusion is performed *after* ESM2 contextualisation.  It is therefore
a post-hoc sensitivity analysis, not sequence re-masking and not native
residue-atom co-attention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import infer_mgca_case_ensemble as base
import infer_mgca_case_ensemble_2773 as d2773


DEFAULT_SEEDS = [42, 142, 242, 342, 442]
DATASET_DEFAULTS = {
    "KinetX": {"epochs": 36, "config_id": base.EXPECTED_CONFIG_ID},
    "2773": {
        "epochs": d2773.DEFAULT_EPOCHS_2773,
        "config_id": d2773.EXPECTED_CONFIG_ID_2773,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), required=True)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-predictions",
        type=Path,
        help="Optional Stage-1 five-seed prediction CSV for an exact no-occlusion audit.",
    )
    parser.add_argument("--reference-audit-tolerance", type=float, default=1e-5)
    parser.add_argument("--target-uniprot", default="P00742")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--expected-config-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--protein-window-size", type=int, default=16)
    parser.add_argument("--protein-stride", type=int, default=4)
    parser.add_argument(
        "--sensitivity-window-sizes", type=int, nargs="*", default=[8, 32]
    )
    parser.add_argument(
        "--protein-baseline",
        choices=["sequence_mean", "zero"],
        default="sequence_mean",
    )
    parser.add_argument(
        "--include-zero-baseline-sensitivity",
        action="store_true",
        help="Also evaluate the main window size with zero-vector replacement.",
    )
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--stability-top-k", type=int, default=10)
    parser.add_argument(
        "--representative-compound-id",
        default="",
        help=(
            "Optional pre-declared compound_id for dual occlusion, normally chosen "
            "only after exact PDB-ligand eligibility screening. If omitted, the "
            "highest observed-pKoff compound is used."
        ),
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def resolve_settings(args: argparse.Namespace) -> None:
    defaults = DATASET_DEFAULTS[args.dataset]
    args.epochs = defaults["epochs"] if args.epochs is None else args.epochs
    args.expected_config_id = (
        defaults["config_id"]
        if args.expected_config_id is None
        else args.expected_config_id
    )
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise ValueError("Exactly five distinct seeds are required")
    if args.protein_window_size <= 0 or args.protein_stride <= 0:
        raise ValueError("Protein window size and stride must be positive")
    if args.top_features <= 0 or args.stability_top_k <= 0:
        raise ValueError("Top-feature settings must be positive")


def checkpoint_records(args: argparse.Namespace) -> list[dict]:
    return (
        d2773.checkpoint_paths_2773(args)
        if args.dataset == "2773"
        else base.checkpoint_paths(args)
    )


def validate_checkpoint(
    args: argparse.Namespace,
    checkpoint: dict,
    record: dict,
    reference_config: dict | None,
    reference_data_hash: str | None,
) -> tuple[dict, str]:
    validator = (
        d2773.validate_checkpoint_2773
        if args.dataset == "2773"
        else base.validate_checkpoint
    )
    return validator(
        checkpoint,
        record,
        args.expected_config_id,
        reference_config,
        reference_data_hash,
    )


def load_factor_xa_compounds(path: Path, legacy, uniprot: str) -> tuple[list[dict], str]:
    rows = [
        row
        for row in base.read_case_metadata(path)
        if row["uniprot_id"].upper() == uniprot.upper()
    ]
    if not rows:
        raise ValueError(f"No rows found for UniProt {uniprot}: {path}")
    fastas = {row["fasta"] for row in rows}
    if len(fastas) != 1:
        raise RuntimeError(f"Expected one sequence for {uniprot}, found {len(fastas)}")

    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in rows:
        canonical = base.canonicalize_smiles(legacy, row["smiles"])
        grouped.setdefault(canonical, []).append(row)

    compounds = []
    for index, (canonical, members) in enumerate(grouped.items(), start=1):
        observed = np.asarray([row["observed_pkoff"] for row in members], dtype=float)
        compounds.append(
            {
                "compound_id": f"factor_xa_{index:02d}",
                "canonical_smiles": canonical,
                "representative_smiles": members[0]["smiles"],
                "observed_pkoff": float(observed.mean()),
                "observed_pkoff_sd": (
                    float(observed.std(ddof=1)) if len(observed) > 1 else 0.0
                ),
                "n_measurements": len(members),
                "source": "|".join(sorted({row["source"] for row in members})),
            }
        )
    return compounds, next(iter(fastas))


def esm_token_cache_path(output_dir: Path, fasta: str, layers: list[int]) -> Path:
    layer_tag = "-".join(str(value) for value in layers)
    return output_dir / "cache" / (
        f"factor_xa_{base.sha256_text(fasta)[:12]}__layers_{layer_tag}.pt"
    )


def extract_selected_esm_tokens(
    fasta: str,
    legacy,
    esm2_path: Path,
    device: torch.device,
    selected_layers: list[int],
    cache_path: Path,
) -> dict:
    if cache_path.is_file():
        cached = base.torch_load_full_compat(cache_path, map_location="cpu")
        expected = {"selected_layers", "token_states", "attention_mask", "residue_indices"}
        if expected.issubset(cached) and cached["selected_layers"] == selected_layers:
            return cached
        raise RuntimeError(f"Malformed ESM token cache: {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting pre-pooling ESM2 token states for layers {selected_layers}...")
    tokenizer = legacy.AutoTokenizer.from_pretrained(str(esm2_path))
    esm_model = legacy.AutoModelForMaskedLM.from_pretrained(str(esm2_path)).to(device)
    esm_model.eval()
    encoded = tokenizer(
        [fasta],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        return_special_tokens_mask=True,
    )
    attention = encoded["attention_mask"][0].bool().cpu()
    special = encoded["special_tokens_mask"][0].bool().cpu()
    residue_indices = torch.nonzero(attention & ~special, as_tuple=False).reshape(-1)
    if int(residue_indices.numel()) != len(fasta):
        raise RuntimeError(
            "Tokenizer/residue mismatch: "
            f"sequence={len(fasta)}, non-special tokens={residue_indices.numel()}"
        )
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key != "special_tokens_mask"
    }
    with torch.no_grad():
        output = esm_model(**model_inputs, output_hidden_states=True)
    if max(selected_layers) >= len(output.hidden_states):
        raise RuntimeError(
            f"Requested ESM layer {max(selected_layers)}, "
            f"but only {len(output.hidden_states)} hidden states exist"
        )
    token_states = torch.stack(
        [output.hidden_states[layer][0].detach().float().cpu() for layer in selected_layers],
        dim=0,
    )
    payload = {
        "fasta_sha256": base.sha256_text(fasta),
        "sequence_length": len(fasta),
        "selected_layers": selected_layers,
        "token_states": token_states,
        "attention_mask": attention,
        "residue_indices": residue_indices,
        "pooling_contract": (
            "Legacy MGCA attention-mask mean includes special tokens; only residue "
            "tokens are replaced during occlusion and the denominator stays fixed."
        ),
    }
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    del output, esm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def window_starts(sequence_length: int, size: int, stride: int) -> list[int]:
    if size > sequence_length:
        raise ValueError(f"Window {size} exceeds sequence length {sequence_length}")
    starts = list(range(0, sequence_length - size + 1, stride))
    final = sequence_length - size
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def pooled_protein_features(
    token_payload: dict,
    depth_windows: list[list[int]],
    configurations: list[dict],
    fasta: str,
) -> tuple[torch.Tensor, dict[str, list[dict]], dict[str, torch.Tensor]]:
    layers = token_payload["selected_layers"]
    layer_to_offset = {layer: index for index, layer in enumerate(layers)}
    states = token_payload["token_states"].float()
    attention_indices = torch.nonzero(
        token_payload["attention_mask"], as_tuple=False
    ).reshape(-1)
    residue_indices = token_payload["residue_indices"].long()
    denominator = float(attention_indices.numel())
    attended_sums = states.index_select(1, attention_indices).sum(dim=1)
    residue_means = states.index_select(1, residue_indices).mean(dim=1)

    def assemble(layer_pools: dict[int, torch.Tensor]) -> torch.Tensor:
        experts = []
        for window in depth_windows:
            experts.append(torch.stack([layer_pools[layer] for layer in window]).mean(dim=0))
        return torch.stack(experts, dim=0)

    original_layer_pools = {
        layer: attended_sums[layer_to_offset[layer]] / denominator for layer in layers
    }
    original = assemble(original_layer_pools)
    records_by_config: dict[str, list[dict]] = {}
    tensors_by_config: dict[str, torch.Tensor] = {}
    for config in configurations:
        key = config["config_key"]
        records = []
        tensors = []
        size = int(config["window_size"])
        for start in window_starts(len(fasta), size, int(config["stride"])):
            selected_residues = residue_indices[start : start + size]
            masked_layer_pools = {}
            for layer in layers:
                offset = layer_to_offset[layer]
                removed = states[offset].index_select(0, selected_residues).sum(dim=0)
                if config["baseline"] == "sequence_mean":
                    replacement = residue_means[offset] * size
                elif config["baseline"] == "zero":
                    replacement = torch.zeros_like(removed)
                else:
                    raise ValueError(config["baseline"])
                masked_layer_pools[layer] = (
                    attended_sums[offset] - removed + replacement
                ) / denominator
            tensors.append(assemble(masked_layer_pools))
            records.append(
                {
                    "config_key": key,
                    "window_size": size,
                    "stride": int(config["stride"]),
                    "baseline": config["baseline"],
                    "window_start": start + 1,
                    "window_end": start + size,
                    "window_sequence": fasta[start : start + size],
                }
            )
        records_by_config[key] = records
        tensors_by_config[key] = torch.stack(tensors, dim=0)
    return original, records_by_config, tensors_by_config


def atom_environment(legacy, molecule, center: int, radius: int) -> dict:
    if radius == 0:
        atom_indices = [int(center)]
        fragment_smiles = legacy.Chem.MolFragmentToSmiles(
            molecule, atomsToUse=atom_indices, canonical=True, isomericSmiles=True
        )
    else:
        bonds = list(legacy.Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center))
        atoms = {int(center)}
        for bond_index in bonds:
            bond = molecule.GetBondWithIdx(int(bond_index))
            atoms.add(int(bond.GetBeginAtomIdx()))
            atoms.add(int(bond.GetEndAtomIdx()))
        atom_indices = sorted(atoms)
        fragment_smiles = legacy.Chem.MolFragmentToSmiles(
            molecule,
            atomsToUse=atom_indices,
            bondsToUse=[int(value) for value in bonds],
            canonical=True,
            isomericSmiles=True,
        )
    return {
        "center_atom": int(center),
        "environment_radius": int(radius),
        "atom_indices": atom_indices,
        "fragment_smiles": fragment_smiles,
    }


def prepare_ligand_features(compounds: list[dict], legacy, fingerprint_type: str):
    if fingerprint_type not in {"morgan", "fcfp"}:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
    molecules = [legacy.Chem.MolFromSmiles(row["canonical_smiles"]) for row in compounds]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Factor Xa contains an invalid canonical SMILES")

    all_features = []
    all_metadata = []
    for compound, molecule in zip(compounds, molecules):
        channels = []
        feature_metadata = []
        for channel_radius in range(4):
            invariants = (
                legacy.rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
                if fingerprint_type == "fcfp"
                else None
            )
            generator = legacy.rdFingerprintGenerator.GetMorganGenerator(
                radius=channel_radius,
                fpSize=2048,
                atomInvariantsGenerator=invariants,
            )
            additional = legacy.rdFingerprintGenerator.AdditionalOutput()
            additional.AllocateBitInfoMap()
            bit_vector = generator.GetFingerprint(molecule, additionalOutput=additional)
            array = np.zeros((2048,), dtype=np.uint8)
            legacy.DataStructs.ConvertToNumpyArray(bit_vector, array)
            channels.append(torch.from_numpy(array).float())
            bit_info = additional.GetBitInfoMap() or {}
            for bit_id in np.flatnonzero(array):
                occurrences = [
                    atom_environment(legacy, molecule, int(center), int(radius))
                    for center, radius in bit_info.get(int(bit_id), ())
                ]
                union_atoms = sorted(
                    {atom for item in occurrences for atom in item["atom_indices"]}
                )
                feature_metadata.append(
                    {
                        "feature_key": f"r{channel_radius}_b{int(bit_id)}",
                        "channel_radius": channel_radius,
                        "bit_id": int(bit_id),
                        "occurrence_count": len(occurrences),
                        "collision_or_repetition": len(occurrences) != 1,
                        "atom_indices_json": json.dumps(union_atoms),
                        "occurrences_json": json.dumps(occurrences, separators=(",", ":")),
                        "fragment_smiles_json": json.dumps(
                            [item["fragment_smiles"] for item in occurrences],
                            ensure_ascii=False,
                        ),
                    }
                )
        tensor = torch.stack(channels, dim=0)
        # Audit against the exact helper used in model training.
        for radius in range(4):
            expected, valid = legacy.get_fingerprint(
                radius, [molecule], device="cpu", fingerprint_type=fingerprint_type
            )
            if not bool(valid[0]) or not torch.equal(expected[0], tensor[radius]):
                raise RuntimeError(
                    f"Fingerprint audit failed for {compound['compound_id']}, radius={radius}"
                )
        all_features.append(tensor)
        all_metadata.append(feature_metadata)
    return molecules, torch.stack(all_features, dim=0), all_metadata


def ligand_occlusion_tensor(original: torch.Tensor, metadata: list[dict]) -> torch.Tensor:
    variants = original.unsqueeze(0).repeat(len(metadata), 1, 1)
    for index, feature in enumerate(metadata):
        variants[index, feature["channel_radius"], feature["bit_id"]] = 0.0
    return variants


def predict_cpu_features(
    model,
    protein_cpu: torch.Tensor,
    ligand_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if len(protein_cpu) != len(ligand_cpu):
        raise ValueError("Protein/ligand feature count mismatch")
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(protein_cpu), batch_size):
            stop = min(start + batch_size, len(protein_cpu))
            protein = protein_cpu[start:stop].to(device=device, dtype=torch.float32)
            ligand = ligand_cpu[start:stop].to(device=device, dtype=torch.float32)
            output, _ = model(protein, ligand)
            values.extend(output.reshape(-1).detach().cpu().tolist())
    return np.asarray(values, dtype=float)


def load_model(
    corrected,
    config: dict,
    record: dict,
    args: argparse.Namespace,
    reference_config: dict,
    reference_data_hash: str,
    device: torch.device,
):
    checkpoint = base.torch_load_full_compat(record["path"], map_location=device)
    validate_checkpoint(
        args, checkpoint, record, reference_config, reference_data_hash
    )
    model = corrected.FullRegressionTransformer(
        **base.model_kwargs_from_config(config)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    del checkpoint
    return model


def assign_ranks(rows: list[dict], group_fields: list[str]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    for members in groups.values():
        members.sort(key=lambda row: (-float(row["delta_pkoff"]), row["feature_key"]))
        for rank, row in enumerate(members, start=1):
            row["rank_within_seed"] = rank


def summarize_importance(
    rows: list[dict],
    feature_fields: list[str],
    top_k: int,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in feature_fields)].append(row)
    output = []
    for key, members in groups.items():
        ordered = sorted(members, key=lambda row: int(row["seed"]))
        deltas = np.asarray([float(row["delta_pkoff"]) for row in ordered], dtype=float)
        ranks = np.asarray([float(row["rank_within_seed"]) for row in ordered], dtype=float)
        positive = float(np.mean(deltas > 0))
        negative = float(np.mean(deltas < 0))
        common = {field: value for field, value in zip(feature_fields, key)}
        representative = members[0]
        passthrough = {
            field: representative[field]
            for field in representative
            if field not in common
            and field
            not in {
                "seed",
                "checkpoint_sha256",
                "original_prediction",
                "masked_prediction",
                "delta_pkoff",
                "rank_within_seed",
            }
        }
        output.append(
            {
                **common,
                **passthrough,
                "n_seeds": len(members),
                "mean_delta_pkoff": float(deltas.mean()),
                "sd_delta_pkoff": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "median_delta_pkoff": float(np.median(deltas)),
                "positive_seed_fraction": positive,
                "sign_consistency": max(positive, negative),
                "mean_rank": float(ranks.mean()),
                f"top{top_k}_seed_fraction": float(np.mean(ranks <= top_k)),
                "delta_by_seed_json": json.dumps(
                    {str(row["seed"]): float(row["delta_pkoff"]) for row in ordered}
                ),
            }
        )
    return sorted(output, key=lambda row: -float(row["mean_delta_pkoff"]))


def stability_metrics(
    rows: list[dict],
    group_fields: list[str],
    feature_field: str,
    top_k: int,
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, members in groups.items():
        by_seed: dict[int, dict[str, float]] = defaultdict(dict)
        for row in members:
            by_seed[int(row["seed"])][str(row[feature_field])] = float(row["delta_pkoff"])
        pair_spearman = []
        pair_jaccard = []
        seeds = sorted(by_seed)
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1 :]:
                common = sorted(set(by_seed[seed_a]) & set(by_seed[seed_b]))
                if len(common) >= 2:
                    ranks_a = {
                        feature: rank
                        for rank, feature in enumerate(
                            sorted(common, key=lambda f: (-by_seed[seed_a][f], f)), start=1
                        )
                    }
                    ranks_b = {
                        feature: rank
                        for rank, feature in enumerate(
                            sorted(common, key=lambda f: (-by_seed[seed_b][f], f)), start=1
                        )
                    }
                    vector_a = np.asarray([ranks_a[value] for value in common], dtype=float)
                    vector_b = np.asarray([ranks_b[value] for value in common], dtype=float)
                    pair_spearman.append(float(np.corrcoef(vector_a, vector_b)[0, 1]))
                top_a = set(
                    sorted(by_seed[seed_a], key=lambda f: (-by_seed[seed_a][f], f))[:top_k]
                )
                top_b = set(
                    sorted(by_seed[seed_b], key=lambda f: (-by_seed[seed_b][f], f))[:top_k]
                )
                union = top_a | top_b
                pair_jaccard.append(len(top_a & top_b) / len(union) if union else 1.0)
        result = {field: value for field, value in zip(group_fields, key)}
        result.update(
            {
                "feature_count": len(by_seed[seeds[0]]) if seeds else 0,
                "seed_pair_count": len(pair_jaccard),
                "mean_pairwise_spearman": (
                    float(np.nanmean(pair_spearman)) if pair_spearman else math.nan
                ),
                f"mean_pairwise_top{top_k}_jaccard": (
                    float(np.mean(pair_jaccard)) if pair_jaccard else math.nan
                ),
            }
        )
        output.append(result)
    return output


def dual_summary(rows: list[dict]) -> list[dict]:
    fields = [
        "training_dataset",
        "compound_id",
        "protein_feature_key",
        "ligand_feature_key",
        "window_start",
        "window_end",
        "channel_radius",
        "bit_id",
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, members in groups.items():
        interactions = np.asarray(
            [float(row["interaction_pkoff"]) for row in members], dtype=float
        )
        positive = float(np.mean(interactions > 0))
        negative = float(np.mean(interactions < 0))
        output.append(
            {
                **{field: value for field, value in zip(fields, key)},
                "window_sequence": members[0]["window_sequence"],
                "fragment_smiles_json": members[0]["fragment_smiles_json"],
                "atom_indices_json": members[0]["atom_indices_json"],
                "n_seeds": len(members),
                "mean_interaction_pkoff": float(interactions.mean()),
                "sd_interaction_pkoff": (
                    float(interactions.std(ddof=1)) if len(interactions) > 1 else 0.0
                ),
                "positive_seed_fraction": positive,
                "sign_consistency": max(positive, negative),
                "interaction_by_seed_json": json.dumps(
                    {
                        str(row["seed"]): float(row["interaction_pkoff"])
                        for row in sorted(members, key=lambda item: int(item["seed"]))
                    }
                ),
            }
        )
    return sorted(output, key=lambda row: -float(row["mean_interaction_pkoff"]))


def audit_original_predictions(
    path: Path,
    original_rows: list[dict],
    seeds: list[int],
    tolerance: float,
) -> dict:
    reference = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        read_reference_rows = list(csv.DictReader(handle))
    for row in read_reference_rows:
        target = str(row.get("uniprot_id", "")).upper()
        if target == "P00742":
            reference.append(row)
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
            if column not in row or (canonical, seed) not in lookup:
                continue
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
        "reference_sha256": base.sha256_file(path),
        "matched_predictions": matched,
        "maximum_absolute_difference": maximum,
        "tolerance": tolerance,
        "passed": True,
    }


def main() -> None:
    args = parse_args()
    resolve_settings(args)
    for path in (args.case_csv, args.checkpoint_root, args.esm2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    corrected = base.load_corrected_module()
    legacy = corrected.legacy
    compounds, fasta = load_factor_xa_compounds(
        args.case_csv, legacy, args.target_uniprot
    )
    records = checkpoint_records(args)
    first = base.torch_load_full_compat(records[0]["path"], map_location="cpu")
    reference_config, reference_data_hash = validate_checkpoint(
        args, first, records[0], None, None
    )
    config = first["config"]
    del first

    depth_windows = legacy.build_esm_depth_windows(
        36, int(config["window_size"]), config["window_layout"]
    )
    selected_layers = sorted({layer for window in depth_windows for layer in window})
    cache_path = esm_token_cache_path(args.output_dir, fasta, selected_layers)
    token_payload = extract_selected_esm_tokens(
        fasta, legacy, args.esm2_path, device, selected_layers, cache_path
    )

    configurations = [
        {
            "config_key": f"w{args.protein_window_size}_s{args.protein_stride}_{args.protein_baseline}",
            "window_size": args.protein_window_size,
            "stride": args.protein_stride,
            "baseline": args.protein_baseline,
            "primary": True,
        }
    ]
    for size in args.sensitivity_window_sizes:
        if size != args.protein_window_size:
            configurations.append(
                {
                    "config_key": f"w{size}_s{args.protein_stride}_{args.protein_baseline}",
                    "window_size": size,
                    "stride": args.protein_stride,
                    "baseline": args.protein_baseline,
                    "primary": False,
                }
            )
    if args.include_zero_baseline_sensitivity and args.protein_baseline != "zero":
        configurations.append(
            {
                "config_key": f"w{args.protein_window_size}_s{args.protein_stride}_zero",
                "window_size": args.protein_window_size,
                "stride": args.protein_stride,
                "baseline": "zero",
                "primary": False,
            }
        )
    original_protein, protein_metadata, protein_variants = pooled_protein_features(
        token_payload, depth_windows, configurations, fasta
    )
    molecules, original_ligands, ligand_metadata = prepare_ligand_features(
        compounds, legacy, config["fingerprint_type"]
    )

    primary_config = configurations[0]["config_key"]
    original_rows: list[dict] = []
    protein_rows: list[dict] = []
    ligand_rows: list[dict] = []
    for record in records:
        seed = int(record["seed"])
        print(f"[{args.dataset}] single occlusion, seed={seed}")
        model = load_model(
            corrected,
            config,
            record,
            args,
            reference_config,
            reference_data_hash,
            device,
        )
        protein_original_batch = original_protein.unsqueeze(0).repeat(len(compounds), 1, 1)
        originals = predict_cpu_features(
            model, protein_original_batch, original_ligands, device, args.batch_size
        )
        for compound, prediction in zip(compounds, originals):
            original_rows.append(
                {
                    "training_dataset": args.dataset,
                    "seed": seed,
                    **compound,
                    "predicted_pkoff": float(prediction),
                    "checkpoint_sha256": record["sha256"],
                }
            )

        for compound_index, compound in enumerate(compounds):
            original_prediction = float(originals[compound_index])
            ligand = original_ligands[compound_index]
            for config_item in configurations:
                config_key = config_item["config_key"]
                variants = protein_variants[config_key]
                predictions = predict_cpu_features(
                    model,
                    variants,
                    ligand.unsqueeze(0).repeat(len(variants), 1, 1),
                    device,
                    args.batch_size,
                )
                for meta, prediction in zip(protein_metadata[config_key], predictions):
                    protein_rows.append(
                        {
                            "training_dataset": args.dataset,
                            "seed": seed,
                            **compound,
                            **meta,
                            "is_primary_configuration": config_item["primary"],
                            "feature_key": f"{config_key}:{meta['window_start']}-{meta['window_end']}",
                            "original_prediction": original_prediction,
                            "masked_prediction": float(prediction),
                            "delta_pkoff": original_prediction - float(prediction),
                            "checkpoint_sha256": record["sha256"],
                        }
                    )

            feature_meta = ligand_metadata[compound_index]
            ligand_variants = ligand_occlusion_tensor(ligand, feature_meta)
            predictions = predict_cpu_features(
                model,
                original_protein.unsqueeze(0).repeat(len(feature_meta), 1, 1),
                ligand_variants,
                device,
                args.batch_size,
            )
            for meta, prediction in zip(feature_meta, predictions):
                ligand_rows.append(
                    {
                        "training_dataset": args.dataset,
                        "seed": seed,
                        **compound,
                        **meta,
                        "original_prediction": original_prediction,
                        "masked_prediction": float(prediction),
                        "delta_pkoff": original_prediction - float(prediction),
                        "checkpoint_sha256": record["sha256"],
                    }
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assign_ranks(
        protein_rows,
        ["training_dataset", "seed", "compound_id", "config_key"],
    )
    assign_ranks(
        ligand_rows,
        ["training_dataset", "seed", "compound_id"],
    )
    protein_summary = summarize_importance(
        protein_rows,
        [
            "training_dataset",
            "compound_id",
            "config_key",
            "feature_key",
            "window_start",
            "window_end",
        ],
        args.stability_top_k,
    )
    ligand_summary = summarize_importance(
        ligand_rows,
        [
            "training_dataset",
            "compound_id",
            "feature_key",
            "channel_radius",
            "bit_id",
        ],
        args.stability_top_k,
    )

    if args.representative_compound_id:
        matches = [
            row
            for row in compounds
            if row["compound_id"] == args.representative_compound_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "Unknown --representative-compound-id "
                f"{args.representative_compound_id!r}; available="
                f"{[row['compound_id'] for row in compounds]}"
            )
        representative = matches[0]
        representative_selection_rule = (
            "pre-declared compound_id, intended for exact PDB-ligand eligibility"
        )
    else:
        representative = max(
            compounds,
            key=lambda row: (float(row["observed_pkoff"]), row["compound_id"]),
        )
        representative_selection_rule = (
            "highest observed pKoff; fixed before inspecting occlusion"
        )
    protein_candidates = [
        row
        for row in protein_summary
        if row["compound_id"] == representative["compound_id"]
        and row["config_key"] == primary_config
    ][: args.top_features]
    ligand_candidates = [
        row
        for row in ligand_summary
        if row["compound_id"] == representative["compound_id"]
    ][: args.top_features]
    if not protein_candidates or not ligand_candidates:
        raise RuntimeError("No features selected for dual occlusion")

    compound_index = next(
        index
        for index, row in enumerate(compounds)
        if row["compound_id"] == representative["compound_id"]
    )
    protein_index = {
        f"{primary_config}:{meta['window_start']}-{meta['window_end']}": index
        for index, meta in enumerate(protein_metadata[primary_config])
    }
    ligand_index = {
        meta["feature_key"]: index
        for index, meta in enumerate(ligand_metadata[compound_index])
    }
    selected_protein_tensor = torch.stack(
        [
            protein_variants[primary_config][protein_index[row["feature_key"]]]
            for row in protein_candidates
        ]
    )
    all_ligand_variants = ligand_occlusion_tensor(
        original_ligands[compound_index], ligand_metadata[compound_index]
    )
    selected_ligand_tensor = torch.stack(
        [all_ligand_variants[ligand_index[row["feature_key"]]] for row in ligand_candidates]
    )

    original_lookup = {
        (int(row["seed"]), row["compound_id"]): float(row["predicted_pkoff"])
        for row in original_rows
    }
    protein_lookup = {
        (int(row["seed"]), row["compound_id"], row["feature_key"]): float(
            row["masked_prediction"]
        )
        for row in protein_rows
        if row["config_key"] == primary_config
    }
    ligand_lookup = {
        (int(row["seed"]), row["compound_id"], row["feature_key"]): float(
            row["masked_prediction"]
        )
        for row in ligand_rows
    }
    dual_rows = []
    pair_proteins = []
    pair_ligands = []
    pair_metadata = []
    for p_index, p_row in enumerate(protein_candidates):
        for l_index, l_row in enumerate(ligand_candidates):
            pair_proteins.append(selected_protein_tensor[p_index])
            pair_ligands.append(selected_ligand_tensor[l_index])
            pair_metadata.append((p_row, l_row))
    pair_proteins = torch.stack(pair_proteins)
    pair_ligands = torch.stack(pair_ligands)

    for record in records:
        seed = int(record["seed"])
        print(f"[{args.dataset}] dual occlusion, seed={seed}")
        model = load_model(
            corrected,
            config,
            record,
            args,
            reference_config,
            reference_data_hash,
            device,
        )
        double_predictions = predict_cpu_features(
            model, pair_proteins, pair_ligands, device, args.batch_size
        )
        original_prediction = original_lookup[(seed, representative["compound_id"])]
        for (p_row, l_row), double_prediction in zip(pair_metadata, double_predictions):
            protein_masked = protein_lookup[
                (seed, representative["compound_id"], p_row["feature_key"])
            ]
            ligand_masked = ligand_lookup[
                (seed, representative["compound_id"], l_row["feature_key"])
            ]
            interaction = (
                protein_masked
                + ligand_masked
                - original_prediction
                - float(double_prediction)
            )
            dual_rows.append(
                {
                    "training_dataset": args.dataset,
                    "seed": seed,
                    **representative,
                    "protein_feature_key": p_row["feature_key"],
                    "window_start": p_row["window_start"],
                    "window_end": p_row["window_end"],
                    "window_sequence": p_row["window_sequence"],
                    "ligand_feature_key": l_row["feature_key"],
                    "channel_radius": l_row["channel_radius"],
                    "bit_id": l_row["bit_id"],
                    "atom_indices_json": l_row["atom_indices_json"],
                    "fragment_smiles_json": l_row["fragment_smiles_json"],
                    "original_prediction": original_prediction,
                    "protein_masked_prediction": protein_masked,
                    "ligand_masked_prediction": ligand_masked,
                    "double_masked_prediction": float(double_prediction),
                    "protein_drop": original_prediction - protein_masked,
                    "ligand_drop": original_prediction - ligand_masked,
                    "double_drop": original_prediction - float(double_prediction),
                    "interaction_pkoff": interaction,
                    "interaction_definition": "yp+yl-y0-ypl",
                    "checkpoint_sha256": record["sha256"],
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dual_aggregate = dual_summary(dual_rows)
    protein_stability = stability_metrics(
        protein_rows,
        ["training_dataset", "compound_id", "config_key"],
        "feature_key",
        args.stability_top_k,
    )
    ligand_stability = stability_metrics(
        ligand_rows,
        ["training_dataset", "compound_id"],
        "feature_key",
        args.stability_top_k,
    )
    reference_audit = None
    if args.reference_predictions is not None:
        if not args.reference_predictions.is_file():
            raise FileNotFoundError(args.reference_predictions)
        reference_audit = audit_original_predictions(
            args.reference_predictions,
            original_rows,
            args.seeds,
            args.reference_audit_tolerance,
        )

    outputs = {
        "compound_catalog": args.output_dir / "factor_xa_compound_catalog.csv",
        "original_predictions": args.output_dir / "original_predictions_five_seeds.csv",
        "protein_per_seed": args.output_dir / "protein_window_importance_per_seed.csv",
        "protein_summary": args.output_dir / "protein_window_importance_summary.csv",
        "protein_stability": args.output_dir / "protein_window_seed_stability.csv",
        "ligand_per_seed": args.output_dir / "ligand_bit_importance_per_seed.csv",
        "ligand_summary": args.output_dir / "ligand_bit_importance_summary.csv",
        "ligand_stability": args.output_dir / "ligand_bit_seed_stability.csv",
        "dual_per_seed": args.output_dir / "dual_occlusion_interactions_per_seed.csv",
        "dual_summary": args.output_dir / "dual_occlusion_interactions_summary.csv",
        "selection": args.output_dir / "dual_occlusion_selection.json",
        "manifest": args.output_dir / "factor_xa_stage2_manifest.json",
    }
    base.atomic_write_csv(outputs["compound_catalog"], compounds)
    base.atomic_write_csv(outputs["original_predictions"], original_rows)
    base.atomic_write_csv(outputs["protein_per_seed"], protein_rows)
    base.atomic_write_csv(outputs["protein_summary"], protein_summary)
    base.atomic_write_csv(outputs["protein_stability"], protein_stability)
    base.atomic_write_csv(outputs["ligand_per_seed"], ligand_rows)
    base.atomic_write_csv(outputs["ligand_summary"], ligand_summary)
    base.atomic_write_csv(outputs["ligand_stability"], ligand_stability)
    base.atomic_write_csv(outputs["dual_per_seed"], dual_rows)
    base.atomic_write_csv(outputs["dual_summary"], dual_aggregate)
    atomic_write_json(
        outputs["selection"],
        {
            "selection_rule": representative_selection_rule,
            "representative_compound": representative,
            "protein_configuration": primary_config,
            "top_feature_count": args.top_features,
            "protein_features": protein_candidates,
            "ligand_features": ligand_candidates,
        },
    )
    manifest = {
        "protocol": "factor_xa_stage2_posthoc_occlusion_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_dataset": args.dataset,
        "target_uniprot": args.target_uniprot,
        "sequence_length": len(fasta),
        "compound_count": len(compounds),
        "case_csv": str(args.case_csv.resolve()),
        "case_csv_sha256": base.sha256_file(args.case_csv),
        "config_id": args.expected_config_id,
        "epochs": args.epochs,
        "checkpoint_training_data_sha256": reference_data_hash,
        "seeds": args.seeds,
        "depth_windows": depth_windows,
        "selected_esm_layers": selected_layers,
        "esm_token_cache": str(cache_path.resolve()),
        "protein_occlusion": {
            "level": "post-ESM pre-pooling token representation",
            "configurations": configurations,
            "special_tokens_preserved": True,
            "pooling_denominator_fixed": True,
        },
        "ligand_occlusion": {
            "feature_identity": "(Morgan maximum-radius channel, hashed bit ID)",
            "operation": "one active channel-specific bit changed from 1 to 0",
            "bit_occurrences_retained": True,
        },
        "dual_interaction_definition": "I=yp+yl-y0-ypl",
        "interpretation_boundary": (
            "Post-hoc occlusion sensitivity; not native residue-atom co-attention."
        ),
        "stage1_no_occlusion_prediction_audit": reference_audit,
        "checkpoints": [
            {
                "seed": record["seed"],
                "path": str(record["path"].resolve()),
                "sha256": record["sha256"],
            }
            for record in records
        ],
        "outputs": {key: str(path.resolve()) for key, path in outputs.items() if key != "manifest"},
    }
    atomic_write_json(outputs["manifest"], manifest)
    print(f"[{args.dataset}] Factor Xa Stage-2 completed: {args.output_dir}")


if __name__ == "__main__":
    main()
