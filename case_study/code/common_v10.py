#!/usr/bin/env python3
"""Shared, audited helpers for the frozen final MGCA 2773 case study."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
V10_DIR = PROJECT_ROOT / "runtime/mgca_hyperparameter_tuning/v10_unbounded"
V10_MODEL = V10_DIR / "model.py"
FROZEN_PATH = PROJECT_ROOT / "frozen/refit_config.json"
FROZEN = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
EXPECTED_CONFIG_ID = FROZEN['config_id']
EXPECTED_MODEL_SHA256 = FROZEN['model_sha256']
MODEL_VARIANT = FROZEN['model_variant']
PROTOCOL = 'mgca_final_unbounded_widegrid_2773_case_v1'
DEFAULT_SEEDS = (42, 142, 242, 342, 442)
DEFAULT_EPOCHS = FROZEN['refit_epochs']
T_CRITICAL_95_DF4 = 2.7764451051977987


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(str(temporary), str(path))


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def atomic_torch_save(torch_module, path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch_module.save(payload, temporary)
    os.replace(str(temporary), str(path))


def torch_load(torch_module, path: Path, map_location="cpu"):
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch_module.load(path, map_location=map_location)


def load_v10_module(model_path: Path = V10_MODEL):
    # This legacy API name is retained only for the copied case-study adapters.
    # The loaded model is the byte-identical, frozen unbounded benchmark model.
    model_path = Path(model_path).resolve()
    if model_path != V10_MODEL.resolve() or sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise RuntimeError('Only the frozen unbounded model may be loaded')
    spec = importlib.util.spec_from_file_location("mgca_case_v10_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import v10 model: {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_frozen_config(config: dict, model_path: Path = V10_MODEL) -> None:
    if config != FROZEN or config != read_json(FROZEN_PATH):
        raise RuntimeError('Frozen case-study configuration changed')
    if config.get("config_id") != EXPECTED_CONFIG_ID:
        raise RuntimeError(
            f"Expected v10 config {EXPECTED_CONFIG_ID}, found {config.get('config_id')!r}"
        )
    if config.get("dataset") != "2773" or config.get("fingerprint") != "morgan":
        raise RuntimeError("Frozen configuration must be v10/2773/Morgan")
    expected_params = {
        "batch_size": 64,
        "dropout": 0.15,
        "window_size": 2,
        "window_layout": "even_span_v2",
    }
    actual = {key: config.get("params", {}).get(key) for key in expected_params}
    if actual != expected_params:
        raise RuntimeError(f"Frozen v10 parameters changed: {actual}")
    recorded_hash = config.get("model_sha256")
    current_hash = sha256_file(Path(model_path))
    if recorded_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"Frozen configuration records an unexpected model hash: {recorded_hash}")
    if recorded_hash != current_hash:
        raise RuntimeError(
            f"v10 model hash mismatch: config={recorded_hash}, current={current_hash}"
        )


def model_kwargs_from_frozen_config(config: dict) -> dict:
    architecture = config["architecture"]
    params = config["params"]
    return {
        "proj_dim1": 2560,
        "proj_dim2": 2048,
        "hidden_dim": int(architecture["hidden_dim"]),
        "dropout": float(params["dropout"]),
        "nums_of_experts": int(architecture["protein_experts"]),
        "ablation": "no",
        "joint_rank": int(architecture["joint_rank"]),
        "protein_aux_weight": float(architecture["protein_aux_weight"]),
        "drug_utility_weight": float(architecture["drug_utility_weight"]),
        "joint_utility_weight": float(architecture["joint_utility_weight"]),
        "branch_margin": float(architecture["branch_margin"]),
        "drug_gate_init": float(params["drug_gate_init"]),
        "joint_gate_init": float(params["joint_gate_init"]),
        "joint_branch_dropout": float(architecture["joint_branch_dropout"]),
    }


def checkpoint_config_from_frozen(config: dict, epochs: int, seed: int) -> dict:
    params = config["params"]
    architecture = config["architecture"]
    return {
        **model_kwargs_from_frozen_config(config),
        "config_id": config["config_id"],
        "model_variant": MODEL_VARIANT,
        "fingerprint_type": "morgan",
        "window_size": int(params["window_size"]),
        "window_layout": params["window_layout"],
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "batch_size": int(params["batch_size"]),
        "epochs": int(epochs),
        "seed": int(seed),
        "protein_warmup_epochs": int(architecture["protein_warmup_epochs"]),
        "fixed_shrinkage_epochs": int(architecture["fixed_shrinkage_epochs"]),
    }


def checkpoint_records(
    checkpoint_root: Path,
    seeds=DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
) -> list[dict]:
    records = []
    for seed in seeds:
        run_dir = Path(checkpoint_root) / f"seed_{seed}"
        checkpoint = run_dir / f"mgca_final_full2773_seed_{seed}_epoch{epochs}.pt"
        complete = run_dir / ".complete"
        manifest = run_dir / "run_manifest.json"
        for path in (checkpoint, complete, manifest):
            if not path.is_file():
                raise FileNotFoundError(path)
        digest = sha256_file(checkpoint)
        if complete.read_text(encoding="utf-8").strip() != digest:
            raise RuntimeError(f"Completion/checkpoint hash mismatch: {checkpoint}")
        saved = read_json(manifest)
        for name, expected in saved.get('files', {}).items():
            if not (run_dir/name).is_file() or sha256_file(run_dir/name) != expected:
                raise RuntimeError('Checkpoint artifact missing/corrupt: '+str(run_dir/name))
        records.append(
            {
                "seed": int(seed),
                "path": checkpoint,
                "sha256": digest,
                "manifest": manifest,
                "manifest_sha256": sha256_file(manifest),
            }
        )
    return records


def validate_checkpoint(
    payload: dict,
    record: dict,
    expected_data_hash: str | None = None,
    epochs: int = DEFAULT_EPOCHS,
) -> str:
    config = payload.get("config", {})
    data = payload.get("data", {})
    if payload.get("checkpoint_type") != "mgca_final_2773_full_refit_final_state":
        raise RuntimeError(f"Wrong checkpoint type: {record['path']}")
    if config.get("config_id") != EXPECTED_CONFIG_ID:
        raise RuntimeError(f"Wrong config ID: {record['path']}")
    if config.get("model_variant") != MODEL_VARIANT:
        raise RuntimeError(f"Wrong model variant: {record['path']}")
    if payload.get("code", {}).get("model_sha256") != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"Checkpoint model-code hash mismatch: {record['path']}")
    if int(config.get("seed", -1)) != int(record["seed"]):
        raise RuntimeError(f"Seed mismatch: {record['path']}")
    if int(config.get("epochs", -1)) != int(epochs):
        raise RuntimeError(f"Epoch mismatch: {record['path']}")
    if epochs != DEFAULT_EPOCHS or config != checkpoint_config_from_frozen(FROZEN, epochs, record['seed']):
        raise RuntimeError('Checkpoint hyperparameters differ from the frozen protocol')
    if int(data.get("row_count", -1)) != 2773:
        raise RuntimeError(f"Training row-count mismatch: {record['path']}")
    if int(data.get("case_exact_fasta_overlap_count", -1)) != 0:
        raise RuntimeError(f"Case target leaked into full refit: {record['path']}")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"Checkpoint has no model state: {record['path']}")
    data_hash = str(data.get("csv_sha256", ""))
    if expected_data_hash is not None and data_hash != expected_data_hash:
        raise RuntimeError(f"Training-data hash differs across seeds: {record['path']}")
    return data_hash


def make_optimizer(torch_module, model, params):
    """Same two AdamW groups as the frozen benchmark; scalar coordinates get no decay."""
    scalars = [model.drug_shrinkage.logit, model.joint_shrinkage.logit]
    scalar_ids = {id(p) for p in scalars}
    return torch_module.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in scalar_ids], 'weight_decay': float(params['weight_decay'])},
        {'params': scalars, 'weight_decay': 0.0},
    ], lr=float(params['lr']))


def check_case_settings(epochs, seed=None, amp=False):
    if epochs != DEFAULT_EPOCHS or (seed is not None and seed not in DEFAULT_SEEDS) or amp:
        raise RuntimeError('Frozen protocol requires 33 epochs, five fixed seeds and AMP=false')


def artifact_files_valid(root, manifest):
    for item in manifest.get('outputs', {}).values():
        path = Path(root)/Path(item['path']).name
        if not path.is_file() or sha256_file(path) != item['sha256']:
            return False
    return bool(manifest.get('outputs'))


def read_case_rows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Case CSV has no header: {path}")
        lookup = {name.lower(): name for name in reader.fieldnames}
        missing = [key for key in ("target_name", "uniprot_id", "fasta", "smiles", "pkoff") if key not in lookup]
        if missing:
            raise ValueError(f"Missing columns {missing}: {path}")
        output = []
        for index, source in enumerate(reader, 1):
            fasta = "".join(source[lookup["fasta"]].split()).upper()
            smiles = source[lookup["smiles"]].strip()
            output.append(
                {
                    "sample_id": f"case_{index:05d}",
                    "row_index": index - 1,
                    "target_name": source[lookup["target_name"]].strip(),
                    "uniprot_id": source[lookup["uniprot_id"]].strip(),
                    "fasta": fasta,
                    "fasta_sha256": sha256_text(fasta),
                    "smiles": smiles,
                    "observed_pkoff": float(source[lookup["pkoff"]]),
                    "source": source.get(lookup.get("source", ""), "").strip(),
                    "category": source.get(lookup.get("category", ""), "").strip(),
                }
            )
    if not output:
        raise ValueError(f"Empty case CSV: {path}")
    return output


def canonical_smiles(legacy, smiles: str) -> str:
    molecule = legacy.Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return legacy.Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def prepare_case_features(
    metadata: list[dict],
    legacy,
    esm2_path: Path,
    device,
    cache_dir: Path,
    config: dict,
    torch_module,
    esm_batch_size: int = 1,
):
    unique_fastas = list(OrderedDict((row["fasta"], None) for row in metadata))
    fasta_index = {value: index for index, value in enumerate(unique_fastas)}
    case_key = sha256_text("\n".join(unique_fastas))[:16]
    cache = Path(cache_dir) / (
        f"esm2_case_{case_key}__ws{config['window_size']}__wl{config['window_layout']}.pt"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    unique_features = None
    if cache.is_file():
        cached = torch_load(torch_module, cache, map_location="cpu")
        tensor = cached.get("features") if isinstance(cached, dict) else cached
        digest = cached.get("fasta_sha256") if isinstance(cached, dict) else None
        if digest not in (None, sha256_text("\n".join(unique_fastas))):
            raise RuntimeError(f"ESM cache FASTA hash mismatch: {cache}")
        if tuple(tensor.shape) != (len(unique_fastas), 4, int(config["proj_dim1"])):
            raise RuntimeError(f"Unexpected ESM cache shape {tuple(tensor.shape)}: {cache}")
        unique_features = tensor
    if unique_features is None:
        tokenizer = legacy.AutoTokenizer.from_pretrained(str(esm2_path))
        esm_model = legacy.AutoModelForMaskedLM.from_pretrained(str(esm2_path)).to(device)
        esm_model.eval()
        with torch_module.inference_mode():
            unique_features = legacy.batch_extract_esm2(
                unique_fastas,
                tokenizer,
                esm_model,
                device,
                batch_size=esm_batch_size,
                window_size=int(config["window_size"]),
                window_layout=config["window_layout"],
            ).cpu()
        atomic_torch_save(
            torch_module,
            cache,
            {
                "features": unique_features,
                "fasta_sha256": sha256_text("\n".join(unique_fastas)),
                "window_size": int(config["window_size"]),
                "window_layout": config["window_layout"],
            },
        )
        del esm_model
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    indices = torch_module.tensor([fasta_index[row["fasta"]] for row in metadata], dtype=torch_module.long)
    protein = unique_features.index_select(0, indices)
    molecules = [legacy.Chem.MolFromSmiles(row["smiles"]) for row in metadata]
    ligand_parts = []
    for radius in range(4):
        fingerprint, valid = legacy.get_fingerprint(
            radius, molecules, device="cpu", fingerprint_type="morgan"
        )
        if not bool(valid.all().item()):
            raise RuntimeError(f"Invalid fingerprint in radius channel {radius}")
        ligand_parts.append(fingerprint.unsqueeze(1).cpu())
    ligand = torch_module.cat(ligand_parts, dim=1)
    return protein, ligand, cache, unique_fastas


def predict_with_aux(model, protein, ligand, device, batch_size: int):
    import torch

    predictions = []
    aux_keys = (
        "protein_expert_weights",
        "drug_expert_weights",
        "protein_prediction",
        "drug_candidate_prediction",
        "joint_candidate_prediction",
        "drug_gate",
        "joint_gate",
        "effective_drug_gate",
        "effective_joint_gate",
        "drug_contribution_rms",
        "joint_contribution_rms",
        "joint_attention_confidence",
        "attention_p2d",
        "attention_d2p",
    )
    chunks = {key: [] for key in aux_keys}
    model.eval()
    model.set_corrections_enabled(True)
    model.set_shrinkage_learnable(True)
    with torch.inference_mode():
        for start in range(0, len(protein), batch_size):
            stop = min(start + batch_size, len(protein))
            output, aux = model(
                protein[start:stop].float().to(device),
                ligand[start:stop].float().to(device),
            )
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite case prediction; no completed results written")
            predictions.append(output.reshape(-1).cpu().numpy())
            for key in aux_keys:
                if not torch.isfinite(aux[key]).all():
                    raise RuntimeError("Non-finite case diagnostic: " + key)
                chunks[key].append(aux[key].detach().cpu().numpy())
    return np.concatenate(predictions), {key: np.concatenate(value) for key, value in chunks.items()}


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def concordance_index(observed, predicted) -> float | None:
    concordant = comparable = 0.0
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            if observed[left] == observed[right]:
                continue
            comparable += 1.0
            product = (observed[left] - observed[right]) * (predicted[left] - predicted[right])
            if product > 0:
                concordant += 1.0
            elif product == 0:
                concordant += 0.5
    return concordant / comparable if comparable else None


def regression_metrics(observed, predicted) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    mse = float(np.mean(residual ** 2))

    def pearson(left, right):
        if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
            return None
        return finite_or_none(np.corrcoef(left, right)[0, 1])

    def average_ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
            start = stop
        return ranks

    def kendall_tau_b(left, right):
        concordant = discordant = tie_left = tie_right = 0
        for first in range(len(left)):
            for second in range(first + 1, len(left)):
                dx = np.sign(left[first] - left[second])
                dy = np.sign(right[first] - right[second])
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    tie_left += 1
                elif dy == 0:
                    tie_right += 1
                elif dx == dy:
                    concordant += 1
                else:
                    discordant += 1
        denominator = math.sqrt(
            (concordant + discordant + tie_left)
            * (concordant + discordant + tie_right)
        )
        return (concordant - discordant) / denominator if denominator else None

    sst = float(np.sum((observed - observed.mean()) ** 2))
    spearman = pearson(average_ranks(observed), average_ranks(predicted))
    return {
        "n": int(len(observed)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - float(np.sum(residual ** 2)) / sst if sst > 0 else None,
        "pearson": pearson(observed, predicted),
        "spearman": spearman,
        "kendall_tau": kendall_tau_b(observed, predicted),
        "c_index": concordance_index(observed, predicted),
    }


def mean_sd_ci(values) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = T_CRITICAL_95_DF4 * sd / math.sqrt(len(array)) if len(array) == 5 else 0.0
    return mean, sd, mean - half, mean + half
