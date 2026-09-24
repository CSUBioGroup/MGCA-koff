from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model as mgca
from utils import (
    atomic_npz,
    atomic_json,
    atomic_text,
    atomic_torch_save,
    canonical_id,
    environment_snapshot,
    sha256_file,
    write_csv,
)

CGRS_ARCHIVE_KEYS = {
    "drug_gate", "joint_gate", "drug_gate_logits", "joint_gate_logits",
    "effective_drug_gate", "effective_joint_gate", "drug_availability",
    "joint_availability", "drug_contribution_rms", "drug_contribution_energy",
    "joint_contribution_rms", "joint_contribution_energy", "protein_prediction",
    "drug_candidate_prediction", "joint_candidate_prediction",
    "protein_expert_weights", "drug_expert_weights",
    "joint_attention_confidence", "attention_p2d", "attention_d2p",
    "drug_prediction_shift", "joint_prediction_shift",
    "drug_branch_gain", "joint_branch_gain", "drug_branch_raw_gain",
    "joint_branch_raw_gain", "drug_branch_improves",
    "joint_branch_improves",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"1", "true", "yes", "y"}:
        return True
    if str(value).lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def validate_cgrs_archive(path: Path, expected_rows: int):
    try:
        with np.load(path) as archive:
            missing = sorted(CGRS_ARCHIVE_KEYS - set(archive.files))
            if missing:
                raise RuntimeError(f"missing arrays {missing}")
            for key in CGRS_ARCHIVE_KEYS:
                value = archive[key]
                if value.shape[0] != expected_rows:
                    raise RuntimeError(
                        f"{key} has {value.shape[0]} rows, expected {expected_rows}"
                    )
                if not np.isfinite(value).all():
                    raise RuntimeError(f"{key} contains non-finite values")
    except Exception as exc:
        raise RuntimeError(f"invalid CGRS artifact {path}: {exc}") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Artifact-rich MGCA-CGRS v10 trainer")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=["KinetX", "2773"])
    parser.add_argument("--protocol", default="warm", choices=["warm", "drug_cold", "protein_cold"])
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", default="no", choices=sorted(mgca.ABLATIONS))
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fingerprint", default="morgan", choices=["morgan"])
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--window-layout", default="even_span_v2")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--joint-rank", type=int, default=128)
    parser.add_argument("--protein-aux-weight", type=float, default=0.10)
    parser.add_argument("--drug-utility-weight", type=float, default=0.02)
    parser.add_argument("--joint-utility-weight", type=float, default=0.02)
    parser.add_argument("--branch-margin", type=float, default=0.01)
    parser.add_argument("--drug-gate-init", type=float, default=0.10)
    parser.add_argument("--joint-gate-init", type=float, default=0.03)
    parser.add_argument("--joint-branch-dropout", type=float, default=0.15)
    parser.add_argument("--protein-warmup-epochs", type=int, default=5)
    parser.add_argument("--fixed-shrinkage-epochs", type=int, default=5)
    parser.add_argument("--save-best-model", type=parse_bool, default=True)
    parser.add_argument("--save-resume-state", type=parse_bool, default=True)
    parser.add_argument("--save-cgrs-outputs", type=parse_bool, default=True)
    parser.add_argument("--amp", type=parse_bool, default=False)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--config-id", default="unfrozen")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_state(generator):
    payload = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload, generator):
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    generator.set_state(payload["loader_generator"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def prepare_data(args, device):
    paths = [args.train_csv, args.val_csv]
    if not args.selection_only:
        if args.test_csv is None:
            raise ValueError("formal training requires --test-csv")
        paths.append(args.test_csv)
    elif args.test_csv is not None:
        raise ValueError("selection-only mode refuses --test-csv")
    rows_by_split = [mgca.read_labeled_rows(str(path)) for path in paths]
    all_rows = [row for split_rows in rows_by_split for row in split_rows]
    cache_path = mgca.get_combined_esm_cache_path(
        [str(path) for path in paths],
        window_size=args.window_size,
        window_layout=args.window_layout,
    )
    cache_path = Path(cache_path)
    cache_stem = re.sub(r"__ws\d+(?:__wl[a-zA-Z0-9_-]+)?$", "", cache_path.stem)
    drug_signature = canonical_id({
        "fingerprint": args.fingerprint,
        "inputs": [sha256_file(path) for path in paths],
        "n_rows": len(all_rows),
    })
    drug_cache_path = cache_path.with_name(
        f"{cache_stem}__morgan_r0-3_{drug_signature}.pt"
    )
    esm_cache_hit_before = cache_path.is_file()
    morgan_cache_hit_before = drug_cache_path.is_file()
    loaded_fast_path = False
    if cache_path.is_file() and drug_cache_path.is_file():
        try:
            fasta = torch.load(cache_path, map_location=device)
            drug = torch.load(drug_cache_path, map_location=device)
            loaded_fast_path = (
                tuple(fasta.shape) == (len(all_rows), 4, 2560)
                and tuple(drug.shape) == (len(all_rows), 4, 2048)
            )
        except Exception:
            loaded_fast_path = False
    if loaded_fast_path:
        labels = torch.tensor([row[2] for row in all_rows], dtype=torch.float32, device=device)
        print(f"Loading cached Morgan features: {drug_cache_path}")
    else:
        fasta, drug, labels, _ = mgca.preprocess_rows(
            all_rows,
            str(args.esm2_path),
            device,
            esm_cache=str(cache_path),
            cache_label="+".join(path.stem for path in paths),
            fingerprint_type=args.fingerprint,
            window_size=args.window_size,
            window_layout=args.window_layout,
        )
        atomic_torch_save(torch, drug_cache_path, drug.detach().cpu())
    datasets = []
    offset = 0
    for split_rows in rows_by_split:
        end = offset + len(split_rows)
        datasets.append(mgca.ESM2MorganDataset(drug[offset:end], fasta[offset:end], labels[offset:end]))
        offset = end
    return datasets, rows_by_split, {
        "esm2": cache_path,
        "morgan": drug_cache_path,
        "esm2_hit_before": esm_cache_hit_before,
        "morgan_hit_before": morgan_cache_hit_before,
    }


def regression_metrics(labels, predictions):
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise FloatingPointError("non-finite regression input; not a missing correlation")
    values = mgca.compute_metrics(np.asarray(labels), np.asarray(predictions))
    result = {}
    for key in ("mse", "rmse", "mae", "r2", "pearson", "spearman"):
        value = float(values[key])
        if not math.isfinite(value) and key in {"mse", "rmse", "mae"}:
            raise FloatingPointError("non-finite primary metric: " + key)
        result[key] = value if math.isfinite(value) else None
    return result


def branch_gain_artifacts(labels, aux, margin):
    target = np.asarray(labels, dtype=float).reshape(-1)
    protein_error = (np.asarray(aux["protein_prediction"]).reshape(-1) - target) ** 2
    drug_error = (np.asarray(aux["drug_candidate_prediction"]).reshape(-1) - target) ** 2
    joint_error = (np.asarray(aux["joint_candidate_prediction"]).reshape(-1) - target) ** 2
    drug_raw_gain = protein_error - drug_error
    joint_raw_gain = drug_error - joint_error
    drug_gain = drug_raw_gain / (protein_error + drug_error + 1e-6)
    joint_gain = joint_raw_gain / (drug_error + joint_error + 1e-6)
    # The Boolean diagnostics use the same raw squared-error margin as the
    # sequential hinge losses; normalized gains are retained for scale-free
    # comparisons between datasets and protocols.
    drug_improves = drug_raw_gain > margin
    joint_improves = joint_raw_gain > margin
    arrays = {
        "drug_branch_gain": drug_gain.astype(np.float32),
        "joint_branch_gain": joint_gain.astype(np.float32),
        "drug_branch_raw_gain": drug_raw_gain.astype(np.float32),
        "joint_branch_raw_gain": joint_raw_gain.astype(np.float32),
        "drug_branch_improves": drug_improves.astype(np.float32),
        "joint_branch_improves": joint_improves.astype(np.float32),
    }
    diagnostics = {
        "branch_margin": float(margin),
        "drug_improves_fraction": float(drug_improves.mean()),
        "joint_improves_fraction": float(joint_improves.mean()),
        "drug_mean_normalized_gain": float(drug_gain.mean()),
        "joint_mean_normalized_gain": float(joint_gain.mean()),
        "drug_global_gate": float(np.asarray(aux["drug_gate"]).reshape(-1)[0]),
        "joint_global_gate": float(np.asarray(aux["joint_gate"]).reshape(-1)[0]),
    }
    return arrays, diagnostics


def sample_id(row):
    fasta, smiles, _ = row
    return hashlib.sha256(f"{fasta}\0{smiles}".encode("utf-8")).hexdigest()[:24]


def evaluate(model, loader, device, collect_aux=False):
    model.eval()
    predictions, labels = [], []
    collected: dict[str, list[np.ndarray]] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for drug, protein, target in loader:
            drug = drug.float().to(device)
            protein = protein.float().to(device)
            prediction, aux = model(protein, drug)
            predictions.append(prediction.reshape(-1).cpu().numpy())
            labels.append(target.reshape(-1).cpu().numpy())
            if collect_aux:
                for key, value in aux.items():
                    collected.setdefault(key, []).append(value.detach().cpu().numpy())
    cuda_sync(device)
    duration = time.perf_counter() - started
    joined_aux = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    return np.concatenate(labels), np.concatenate(predictions), joined_aux, duration


def checkpoint_payload(model, optimizer, scaler, epoch, best_mse, best_epoch, history, signature, generator):
    return {
        "signature": signature,
        "epoch": epoch,
        "best_mse": best_mse,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
        "rng_state": rng_state(generator),
    }


def main():
    args = parse_args()
    for path in (args.train_csv, args.val_csv, args.esm2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.test_csv is not None and not args.test_csv.exists():
        raise FileNotFoundError(args.test_csv)
    if not args.save_best_model:
        raise ValueError("best checkpoint required for final evaluation")
    if args.hidden_dim != 512:
        raise ValueError("v10 formal protocol fixes hidden_dim=512")
    if args.protein_warmup_epochs < 0 or args.fixed_shrinkage_epochs < 0:
        raise ValueError("staged-training epoch counts must be non-negative")
    if args.epochs <= args.protein_warmup_epochs + args.fixed_shrinkage_epochs:
        raise ValueError("epochs must include at least one learnable-shrinkage epoch")
    if args.joint_rank < 8:
        raise ValueError("joint-rank must be >=8")
    if min(args.drug_utility_weight, args.joint_utility_weight, args.branch_margin) < 0:
        raise ValueError("utility/prior weights and branch margin must be non-negative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.drug_gate_init, args.joint_gate_init)):
        raise ValueError("initial coefficients must be finite and positive")
    if not 0 <= args.joint_branch_dropout < 1:
        raise ValueError("joint-branch-dropout must be in [0,1)")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    identity = {
        "model_variant": mgca.MODEL_VARIANT,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "run": args.run,
        "seed": args.seed,
        "ablation": args.ablation,
        "selection_only": args.selection_only,
        "config_id": args.config_id,
        "device": args.device,
        "amp": bool(args.amp),
        "hyperparameters": {
            "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "dropout": args.dropout,
            "window_size": args.window_size, "window_layout": args.window_layout,
            "epochs": args.epochs, "patience": args.patience,
            "joint_rank": args.joint_rank,
            "protein_aux_weight": args.protein_aux_weight,
            "drug_utility_weight": args.drug_utility_weight,
            "joint_utility_weight": args.joint_utility_weight,
            "branch_margin": args.branch_margin,
            "drug_gate_init": args.drug_gate_init,
            "joint_gate_init": args.joint_gate_init,
            "joint_branch_dropout": args.joint_branch_dropout,
            "protein_warmup_epochs": args.protein_warmup_epochs,
            "fixed_shrinkage_epochs": args.fixed_shrinkage_epochs,
            "gradient_clip_norm": mgca.MAX_GRAD_NORM,
            "numerical_stability_revision": "uncapped_softplus_v1",
        },
        "source_v10_sha256": sha256_file(mgca.SOURCE_V10_SCRIPT),
        "fusion_parameterization": "protein=1; drug/joint=softplus(raw); no cap; no gate prior",
        "scalar_weight_decay": 0.0,
        "model_sha256": sha256_file(Path(mgca.__file__)),
        "trainer_sha256": sha256_file(Path(__file__)),
        "utils_sha256": sha256_file(SCRIPT_DIR / "utils.py"),
        "legacy_dependency_sha256": sha256_file(mgca.LEGACY_SCRIPT),
        "train_sha256": sha256_file(args.train_csv),
        "val_sha256": sha256_file(args.val_csv),
        "test_sha256": sha256_file(args.test_csv) if args.test_csv else None,
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )
        },
        "artifact_policy": {
            "save_best_model": args.save_best_model,
            "save_resume_state": args.save_resume_state,
            "save_cgrs_outputs": args.save_cgrs_outputs,
        },
    }
    signature = canonical_id(identity, 64)
    complete = args.output_dir / ".complete"
    metrics_path = args.output_dir / "metrics.json"
    if complete.is_file() and metrics_path.is_file():
        required = [
            args.output_dir / "identity.json",
            args.output_dir / "environment.json",
            args.output_dir / "history.csv",
            args.output_dir / "train_predictions.csv",
            args.output_dir / "validation_predictions.csv",
        ]
        if not args.selection_only:
            required.append(args.output_dir / "test_predictions.csv")
            if args.save_cgrs_outputs:
                required.append(args.output_dir / "cgrs_outputs.npz")
        else:
            if args.save_cgrs_outputs:
                required.append(args.output_dir / "validation_cgrs_outputs.npz")
        if args.save_best_model:
            required.append(args.output_dir / "best_model.pt")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "completion marker exists but required artifacts are missing: "
                + ", ".join(missing)
            )
        if args.save_cgrs_outputs:
            archive_path = args.output_dir / (
                "validation_cgrs_outputs.npz" if args.selection_only else "cgrs_outputs.npz"
            )
            expected_rows = len(mgca.read_labeled_rows(str(
                args.val_csv if args.selection_only else args.test_csv
            )))
            validate_cgrs_archive(archive_path, expected_rows)
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError(f"completed output has incompatible signature: {args.output_dir}")
        print(json.dumps(existing["val_metrics" if args.selection_only else "test_metrics"]))
        return

    atomic_json(args.output_dir / "environment.json", environment_snapshot(torch))
    atomic_json(args.output_dir / "identity.json", {"signature": signature, **identity})
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")
        torch.cuda.set_device(device)
        torch.cuda.init()
    run_started = time.perf_counter()
    setup_started = time.perf_counter()
    feature_started = time.perf_counter()
    datasets, rows_by_split, cache_paths = prepare_data(args, device)
    feature_preparation_duration = time.perf_counter() - feature_started
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets[0], batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0,
    )
    train_eval_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = None
    if not args.selection_only:
        test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = mgca.FullRegressionTransformer(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        ablation=args.ablation,
        joint_rank=args.joint_rank,
        protein_aux_weight=args.protein_aux_weight,
        drug_utility_weight=args.drug_utility_weight,
        joint_utility_weight=args.joint_utility_weight,
        branch_margin=args.branch_margin,
        drug_gate_init=args.drug_gate_init,
        joint_gate_init=args.joint_gate_init,
        joint_branch_dropout=args.joint_branch_dropout,
    ).to(device)
    # Decay of negative raw coordinates pulls alpha toward log(2), not zero.
    # Exclude only the two scalar coordinates; retain v10 decay elsewhere.
    scalars = [model.drug_shrinkage.logit, model.joint_shrinkage.logit]
    scalar_ids = {id(p) for p in scalars}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in scalar_ids],
         "weight_decay": args.weight_decay},
        {"params": scalars, "weight_decay": 0.0},
    ], lr=args.lr)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    setup_duration = time.perf_counter() - setup_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, start_epoch, best_mse, best_epoch, bad_epochs = [], 1, math.inf, 0, 0
    resume_path = args.output_dir / "last_state.pt"
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu")
        if resume.get("signature") != signature:
            raise RuntimeError(f"resume checkpoint has incompatible signature: {resume_path}")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        history = resume["history"]
        start_epoch = int(resume["epoch"]) + 1
        best_mse = float(resume["best_mse"])
        best_epoch = int(resume["best_epoch"])
        bad_epochs = max(0, start_epoch - 1 - best_epoch)
        restore_rng(resume["rng_state"], generator)

    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        # A crash after the last early-stopping checkpoint must not add an epoch.
        if (best_epoch > 0 and args.patience > 0 and bad_epochs >= args.patience):
            break
        corrections_enabled = epoch > args.protein_warmup_epochs
        shrinkage_learnable = epoch > args.protein_warmup_epochs + args.fixed_shrinkage_epochs
        model.set_corrections_enabled(corrections_enabled)
        model.set_shrinkage_learnable(shrinkage_learnable)
        if epoch == args.protein_warmup_epochs + args.fixed_shrinkage_epochs + 1:
            # Warm-up/fixed-scalar checkpoints are intentionally ineligible.
            best_mse, best_epoch, bad_epochs = math.inf, 0, 0
        model.train()
        epoch_started = time.perf_counter()
        loss_keys = (
            "protein_aux", "drug_utility", "joint_utility", "gate_prior",
        )
        component_totals = {key: 0.0 for key in loss_keys}
        total_main = total_loss = 0.0
        total_items = 0
        scalar_sums = {
            key: 0.0 for key in (
                "drug_gate", "drug_gate_square", "joint_gate", "joint_gate_square",
                "effective_drug_gate", "effective_joint_gate", "drug_contribution_rms",
                "joint_contribution_rms", "drug_availability", "joint_availability",
                "joint_attention_confidence",
                "drug_prediction_shift", "joint_prediction_shift",
            )
        }
        gradient_norm_sum = gradient_norm_max = 0.0
        protein_expert_sum = np.zeros(4, dtype=np.float64)
        drug_expert_sum = np.zeros(4, dtype=np.float64)
        for drug, protein, target in train_loader:
            drug = drug.float().to(device)
            protein = protein.float().to(device)
            target = target.float().to(device).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else contextlib.nullcontext()
            with autocast:
                prediction, aux = model(protein, drug)
                if not torch.isfinite(prediction).all():
                    bad_aux = [key for key, value in aux.items() if not torch.isfinite(value).all()]
                    raise FloatingPointError(
                        f"non-finite prediction at epoch={epoch}; non-finite aux={bad_aux}"
                    )
                main_loss = torch.mean((prediction.reshape(-1) - target) ** 2)
                components = model.loss_components(target, aux)
                loss = main_loss + components["total"]
                if not torch.isfinite(loss):
                    scalar_components = {
                        key: float(value.detach()) for key, value in components.items()
                    }
                    raise FloatingPointError(
                        f"non-finite loss at epoch={epoch}: main={float(main_loss.detach())}, "
                        f"components={scalar_components}"
                    )
            scaler.scale(loss).backward()
            if amp_enabled:
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), mgca.MAX_GRAD_NORM, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(target)
            total_main += float(main_loss.detach()) * count
            for key in loss_keys:
                component_totals[key] += float(components[key].detach()) * count
            total_loss += float(loss.detach()) * count
            total_items += count
            gradient_norm_value = float(gradient_norm.detach())
            gradient_norm_sum += gradient_norm_value * count
            gradient_norm_max = max(gradient_norm_max, gradient_norm_value)
            drug_gate = aux["drug_gate"].detach().reshape(-1)
            joint_gate = aux["joint_gate"].detach().reshape(-1)
            scalar_sums["drug_gate"] += float(drug_gate.sum())
            scalar_sums["drug_gate_square"] += float((drug_gate * drug_gate).sum())
            scalar_sums["joint_gate"] += float(joint_gate.sum())
            scalar_sums["joint_gate_square"] += float((joint_gate * joint_gate).sum())
            for key in (
                "effective_drug_gate", "effective_joint_gate", "drug_contribution_rms",
                "joint_contribution_rms", "drug_availability", "joint_availability",
                "joint_attention_confidence",
                "drug_prediction_shift", "joint_prediction_shift",
            ):
                value = aux[key].detach()
                if key.endswith("availability"):
                    value = value > 0
                scalar_sums[key] += float(value.sum())
            protein_expert_sum += aux["protein_expert_weights"].detach().sum(dim=0).cpu().numpy()
            drug_expert_sum += aux["drug_expert_weights"].detach().sum(dim=0).cpu().numpy()

        val_true, val_pred, _, val_duration = evaluate(model, val_loader, device)
        val_metrics = regression_metrics(val_true, val_pred)
        cuda_sync(device)
        epoch_duration = time.perf_counter() - epoch_started
        drug_gate_mean = scalar_sums["drug_gate"] / total_items
        joint_gate_mean = scalar_sums["joint_gate"] / total_items
        row = {
            "epoch": epoch,
            "corrections_enabled": corrections_enabled,
            "shrinkage_learnable": shrinkage_learnable,
            "train_main_mse": total_main / total_items,
            "train_total_loss": total_loss / total_items,
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "val_pearson": val_metrics["pearson"],
            "val_spearman": val_metrics["spearman"],
            "drug_gate_mean": drug_gate_mean,
            "drug_gate_std": math.sqrt(max(scalar_sums["drug_gate_square"] / total_items - drug_gate_mean ** 2, 0.0)),
            "joint_gate_mean": joint_gate_mean,
            "joint_gate_std": math.sqrt(max(scalar_sums["joint_gate_square"] / total_items - joint_gate_mean ** 2, 0.0)),
            "gradient_norm_mean_before_clip": gradient_norm_sum / total_items,
            "gradient_norm_max_before_clip": gradient_norm_max,
            "gradient_clip_norm": mgca.MAX_GRAD_NORM,
            "epoch_duration_sec": epoch_duration,
            "validation_duration_sec": val_duration,
            "configured_concurrency": args.concurrency,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for key, total in component_totals.items():
            row[f"train_{key}"] = total / total_items
        for key in (
            "effective_drug_gate", "effective_joint_gate", "drug_contribution_rms",
            "joint_contribution_rms", "drug_availability", "joint_availability",
            "joint_attention_confidence",
            "drug_prediction_shift", "joint_prediction_shift",
        ):
            suffix = "fraction" if key.endswith("availability") else "mean"
            row[f"{key}_{suffix}"] = scalar_sums[key] / total_items
        for index in range(4):
            row[f"protein_expert{index}_mean"] = protein_expert_sum[index] / total_items
            row[f"drug_expert{index}_mean"] = drug_expert_sum[index] / total_items
        history.append(row)
        improved = shrinkage_learnable and val_metrics["mse"] < best_mse - 1e-12
        if improved:
            best_mse = val_metrics["mse"]
            best_epoch = epoch
            bad_epochs = 0
            if args.save_best_model:
                atomic_torch_save(torch, args.output_dir / "best_model.pt", {
                    "signature": signature,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "identity": identity,
                    "val_metrics": val_metrics,
                })
        else:
            bad_epochs += 1
        write_csv(args.output_dir / "history.csv", history)
        if args.save_resume_state:
            atomic_torch_save(
                torch, resume_path,
                checkpoint_payload(model, optimizer, scaler, epoch, best_mse, best_epoch, history, signature, generator),
            )
        if shrinkage_learnable and args.patience > 0 and bad_epochs >= args.patience:
            break
    cuda_sync(device)
    training_duration_this_attempt = time.perf_counter() - training_started
    training_duration = sum(float(row["epoch_duration_sec"]) for row in history)

    best_path = args.output_dir / "best_model.pt"
    if args.save_best_model and best_path.is_file():
        best_payload = torch.load(best_path, map_location=device)
        if best_payload.get("signature") != signature:
            raise RuntimeError(f"best checkpoint has incompatible signature: {best_path}")
        model.load_state_dict(best_payload["model_state_dict"])
    model.set_corrections_enabled(True)
    model.set_shrinkage_learnable(True)
    train_true, train_pred, _, train_eval_duration = evaluate(model, train_eval_loader, device)
    val_true, val_pred, val_aux, val_eval_duration = evaluate(model, val_loader, device, collect_aux=True)
    train_metrics = regression_metrics(train_true, train_pred)
    val_metrics = regression_metrics(val_true, val_pred)
    val_branch_arrays, val_branch_diagnostics = branch_gain_artifacts(
        val_true, val_aux, args.branch_margin
    )

    final_payload = {
        "signature": signature,
        "identity": identity,
        "selection_only": args.selection_only,
        "test_accessed": not args.selection_only,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "parameter_count": mgca.parameter_count(model),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "validation_branch_diagnostics": val_branch_diagnostics,
        "input": {
            "train_csv": str(args.train_csv.resolve()),
            "val_csv": str(args.val_csv.resolve()),
            "test_csv": str(args.test_csv.resolve()) if args.test_csv else None,
            "esm2_feature_cache": str(cache_paths["esm2"].resolve()),
            "esm2_feature_cache_sha256": sha256_file(cache_paths["esm2"]),
            "morgan_feature_cache": str(cache_paths["morgan"].resolve()),
            "morgan_feature_cache_sha256": sha256_file(cache_paths["morgan"]),
            "esm2_cache_hit_before_run": cache_paths["esm2_hit_before"],
            "morgan_cache_hit_before_run": cache_paths["morgan_hit_before"],
        },
        "timing": {
            "protocol": "synchronized_wall_clock_v1",
            "configured_concurrency": args.concurrency,
            "concurrency_history": sorted({int(row.get("configured_concurrency", 1)) for row in history}),
            "shared_device_timing": any(int(row.get("configured_concurrency", 1)) > 1 for row in history),
            "setup_duration_sec": setup_duration,
            "feature_preparation_duration_sec": feature_preparation_duration,
            "training_duration_sec": training_duration,
            "training_duration_this_attempt_sec": training_duration_this_attempt,
            "training_duration_definition": "sum of completed epoch durations across resume; includes validation, excludes inter-epoch checkpoint writes",
            "training_samples_per_sec": (
                len(datasets[0]) * max(len(history), 1) / training_duration
                if training_duration > 0 else None
            ),
            "train_evaluation_duration_sec": train_eval_duration,
            "validation_evaluation_duration_sec": val_eval_duration,
            "mean_epoch_duration_sec": training_duration / max(len(history), 1),
            "run_to_metrics_duration_sec": None,
            "training_peak_gpu_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None
            ),
            "training_peak_gpu_memory_reserved_mb": (
                torch.cuda.max_memory_reserved(device) / 1024 ** 2 if device.type == "cuda" else None
            ),
        },
    }

    prediction_rows = [
        {"source_row": index, "sample_id": sample_id(rows_by_split[1][index]),
         "y_true": float(y), "y_pred": float(p),
         "error": float(p - y), "abs_error": float(abs(p - y))}
        for index, (y, p) in enumerate(zip(val_true, val_pred))
    ]
    train_prediction_rows = [
        {"source_row": index, "sample_id": sample_id(rows_by_split[0][index]),
         "y_true": float(y), "y_pred": float(p),
         "error": float(p - y), "abs_error": float(abs(p - y))}
        for index, (y, p) in enumerate(zip(train_true, train_pred))
    ]
    test_rows = None
    test_aux = None
    test_branch_arrays = None
    if not args.selection_only:
        test_true, test_pred, test_aux, test_duration = evaluate(
            model, test_loader, device, collect_aux=True
        )
        final_payload["test_metrics"] = regression_metrics(test_true, test_pred)
        test_branch_arrays, test_branch_diagnostics = branch_gain_artifacts(
            test_true, test_aux, args.branch_margin
        )
        final_payload["test_branch_diagnostics"] = test_branch_diagnostics
        final_payload["timing"]["test_evaluation_duration_sec"] = test_duration
        final_payload["timing"]["test_inference_samples_per_sec"] = (
            len(test_true) / test_duration if test_duration > 0 else None
        )
        test_rows = [
            {"source_row": index, "sample_id": sample_id(rows_by_split[2][index]),
             "y_true": float(y), "y_pred": float(p),
             "error": float(p - y), "abs_error": float(abs(p - y))}
            for index, (y, p) in enumerate(zip(test_true, test_pred))
        ]

    artifact_started = time.perf_counter()
    write_csv(args.output_dir / "train_predictions.csv", train_prediction_rows)
    write_csv(args.output_dir / "validation_predictions.csv", prediction_rows)
    if not args.selection_only:
        write_csv(args.output_dir / "test_predictions.csv", test_rows)
        if args.save_cgrs_outputs:
            atomic_npz(
                args.output_dir / "cgrs_outputs.npz",
                drug_gate=test_aux["drug_gate"],
                joint_gate=test_aux["joint_gate"],
                drug_gate_logits=test_aux["drug_gate_logits"],
                joint_gate_logits=test_aux["joint_gate_logits"],
                effective_drug_gate=test_aux["effective_drug_gate"],
                effective_joint_gate=test_aux["effective_joint_gate"],
                drug_availability=test_aux["drug_availability"],
                joint_availability=test_aux["joint_availability"],
                drug_contribution_rms=test_aux["drug_contribution_rms"],
                drug_contribution_energy=test_aux["drug_contribution_energy"],
                joint_contribution_rms=test_aux["joint_contribution_rms"],
                joint_contribution_energy=test_aux["joint_contribution_energy"],
                protein_prediction=test_aux["protein_prediction"],
                drug_candidate_prediction=test_aux["drug_candidate_prediction"],
                joint_candidate_prediction=test_aux["joint_candidate_prediction"],
                protein_expert_weights=test_aux["protein_expert_weights"],
                drug_expert_weights=test_aux["drug_expert_weights"],
                joint_attention_confidence=test_aux["joint_attention_confidence"],
                attention_p2d=test_aux["attention_p2d"],
                attention_d2p=test_aux["attention_d2p"],
                drug_prediction_shift=test_aux["drug_prediction_shift"],
                joint_prediction_shift=test_aux["joint_prediction_shift"],
                drug_branch_gain=test_branch_arrays["drug_branch_gain"],
                joint_branch_gain=test_branch_arrays["joint_branch_gain"],
                drug_branch_raw_gain=test_branch_arrays["drug_branch_raw_gain"],
                joint_branch_raw_gain=test_branch_arrays["joint_branch_raw_gain"],
                drug_branch_improves=test_branch_arrays["drug_branch_improves"],
                joint_branch_improves=test_branch_arrays["joint_branch_improves"],
            )
    elif args.save_cgrs_outputs:
        atomic_npz(
            args.output_dir / "validation_cgrs_outputs.npz",
            drug_gate=val_aux["drug_gate"],
            joint_gate=val_aux["joint_gate"],
            drug_gate_logits=val_aux["drug_gate_logits"],
            joint_gate_logits=val_aux["joint_gate_logits"],
            effective_drug_gate=val_aux["effective_drug_gate"],
            effective_joint_gate=val_aux["effective_joint_gate"],
            drug_availability=val_aux["drug_availability"],
            joint_availability=val_aux["joint_availability"],
            drug_contribution_rms=val_aux["drug_contribution_rms"],
            drug_contribution_energy=val_aux["drug_contribution_energy"],
            joint_contribution_rms=val_aux["joint_contribution_rms"],
            joint_contribution_energy=val_aux["joint_contribution_energy"],
            protein_prediction=val_aux["protein_prediction"],
            drug_candidate_prediction=val_aux["drug_candidate_prediction"],
            joint_candidate_prediction=val_aux["joint_candidate_prediction"],
            protein_expert_weights=val_aux["protein_expert_weights"],
            drug_expert_weights=val_aux["drug_expert_weights"],
            joint_attention_confidence=val_aux["joint_attention_confidence"],
            attention_p2d=val_aux["attention_p2d"],
            attention_d2p=val_aux["attention_d2p"],
            drug_prediction_shift=val_aux["drug_prediction_shift"],
            joint_prediction_shift=val_aux["joint_prediction_shift"],
            drug_branch_gain=val_branch_arrays["drug_branch_gain"],
            joint_branch_gain=val_branch_arrays["joint_branch_gain"],
            drug_branch_raw_gain=val_branch_arrays["drug_branch_raw_gain"],
            joint_branch_raw_gain=val_branch_arrays["joint_branch_raw_gain"],
            drug_branch_improves=val_branch_arrays["drug_branch_improves"],
            joint_branch_improves=val_branch_arrays["joint_branch_improves"],
        )

    if args.save_cgrs_outputs:
        validate_cgrs_archive(
            args.output_dir / (
                "validation_cgrs_outputs.npz" if args.selection_only else "cgrs_outputs.npz"
            ),
            len(rows_by_split[1] if args.selection_only else rows_by_split[2]),
        )

    final_payload["timing"]["final_evaluation_duration_sec"] = (
        train_eval_duration + val_eval_duration
        + final_payload["timing"].get("test_evaluation_duration_sec", 0.0)
    )
    final_payload["timing"]["artifact_write_duration_sec"] = time.perf_counter() - artifact_started
    final_payload["timing"]["run_to_metrics_duration_sec"] = time.perf_counter() - run_started
    best_path = args.output_dir / "best_model.pt"
    final_payload["checkpoint_size_bytes"] = best_path.stat().st_size if best_path.is_file() else None
    atomic_json(metrics_path, final_payload)
    atomic_text(complete, "complete\n")
    if resume_path.exists():
        resume_path.unlink()
    print(json.dumps(final_payload["val_metrics" if args.selection_only else "test_metrics"]))


if __name__ == "__main__":
    main()
