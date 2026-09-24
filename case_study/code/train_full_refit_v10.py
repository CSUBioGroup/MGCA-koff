#!/usr/bin/env python3
"""Train one MGCA-CGRS v10 checkpoint on all 2,773 rows for fixed epochs."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import common_v10 as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, default=common.V10_MODEL)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=common.DEFAULT_EPOCHS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


def run_identity(args, config, manifest):
    return {
        "protocol": common.PROTOCOL,
        "config_id": config["config_id"],
        "seed": args.seed,
        "epochs": args.epochs,
        "data_sha256": common.sha256_file(args.data_csv),
        "data_manifest_sha256": common.sha256_file(args.data_manifest),
        "model_sha256": common.sha256_file(args.model_file),
        "trainer_sha256": common.sha256_file(Path(__file__).resolve()),
        "config_sha256": common.sha256_file(args.config_json),
        "common_sha256": common.sha256_file(Path(common.__file__)),
        "lock_sha256": common.sha256_file(common.PROJECT_ROOT/'frozen/experiment_lock.json'),
        "package_sha256": common.sha256_file(common.PROJECT_ROOT/'release_manifest.json'),
        "row_count": manifest.get("row_count"),
        "case_exact_fasta_overlap_count": manifest.get("case_exact_fasta_overlap_count"),
    }


def complete_matches(output_dir, identity, checkpoint_name):
    marker = output_dir / ".complete"
    manifest = output_dir / "run_manifest.json"
    checkpoint = output_dir / checkpoint_name
    if not (marker.is_file() and manifest.is_file() and checkpoint.is_file()):
        return False
    saved = common.read_json(manifest)
    return (
        saved.get("run_identity") == identity
        and marker.read_text(encoding="utf-8").strip() == common.sha256_file(checkpoint)
    )


def evaluate(model, loader, device):
    observed, predicted = [], []
    model.eval()
    model.set_corrections_enabled(True)
    model.set_shrinkage_learnable(True)
    with torch.inference_mode():
        for ligand, protein, target in loader:
            output, _ = model(protein.float().to(device), ligand.float().to(device))
            observed.extend(target.reshape(-1).cpu().tolist())
            predicted.extend(output.reshape(-1).cpu().tolist())
    return common.regression_metrics(observed, predicted)


def main():
    args = parse_args()
    common.check_case_settings(args.epochs, args.seed, args.amp)
    for path in (args.data_csv, args.data_manifest, args.config_json, args.model_file, args.esm2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    config = common.read_json(args.config_json)
    common.validate_frozen_config(config, args.model_file)
    manifest = common.read_json(args.data_manifest)
    if manifest.get("row_count") != 2773:
        raise RuntimeError("Full-refit manifest must contain exactly 2,773 rows")
    if manifest.get("output_sha256") != common.sha256_file(args.data_csv):
        raise RuntimeError("Full-refit CSV hash does not match manifest")
    if manifest.get("case_exact_fasta_overlap_count") != 0:
        raise RuntimeError("Four-target exact FASTA overlap must be zero")
    warmup = int(config["architecture"]["protein_warmup_epochs"])
    fixed = int(config["architecture"]["fixed_shrinkage_epochs"])
    if args.epochs <= warmup + fixed:
        raise RuntimeError("Fixed epoch budget must include learnable-shrinkage training")
    identity = run_identity(args, config, manifest)
    checkpoint_name = f"mgca_final_full2773_seed_{args.seed}_epoch{args.epochs}.pt"
    print(json.dumps(identity, indent=2))
    if args.preflight_only:
        print("Training preflight passed; no model was trained.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if complete_matches(args.output_dir, identity, checkpoint_name):
        common.checkpoint_records(args.output_dir.parent, [args.seed], args.epochs)
        print(f"Matching completed run exists; skipping {args.output_dir}")
        return
    if (args.output_dir / ".complete").exists():
        raise RuntimeError(f"Stale completion marker in {args.output_dir}")

    mgca = common.load_v10_module(args.model_file)
    legacy = mgca.legacy
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    set_seeds(args.seed)
    rows = legacy.read_labeled_rows(str(args.data_csv))
    if len(rows) != 2773:
        raise RuntimeError(f"Loaded {len(rows)} rows instead of 2,773")
    data_hash = identity["data_sha256"]
    cache_input = args.data_csv.with_name(
        f"{args.data_csv.stem}__sha256_{data_hash[:12]}{args.data_csv.suffix}"
    )
    esm_cache = legacy.get_default_esm_cache_path(
        str(cache_input),
        window_size=int(config["params"]["window_size"]),
        window_layout=config["params"]["window_layout"],
    )
    feature_started = time.perf_counter()
    protein, ligand, labels, _ = legacy.preprocess_rows(
        rows,
        str(args.esm2_path),
        device,
        esm_cache=esm_cache,
        cache_label="v10 2773 full-refit cohort",
        fingerprint_type="morgan",
        window_size=int(config["params"]["window_size"]),
        window_layout=config["params"]["window_layout"],
    )
    feature_seconds = time.perf_counter() - feature_started
    dataset = legacy.ESM2MorganDataset(ligand, protein, labels)
    # Feature cache hits/misses must not change training RNG initialization.
    set_seeds(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["params"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=int(config["params"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model_kwargs = common.model_kwargs_from_frozen_config(config)
    model = mgca.FullRegressionTransformer(**model_kwargs).to(device)
    optimizer = common.make_optimizer(torch, model, config['params'])
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history, start_epoch, global_step = [], 1, 0
    resume_path = args.output_dir / "last_state.pt"
    if resume_path.is_file():
        resume = common.torch_load(torch, resume_path, map_location='cpu')
        if resume.get("run_identity") != identity:
            raise RuntimeError(f"Resume state identity mismatch: {resume_path}")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        history = resume["history"]
        start_epoch = int(resume["epoch"]) + 1
        global_step = int(resume["global_step"])
        restore_rng(resume["rng_state"], generator)
        print(f"Resuming from epoch {start_epoch}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        corrections = epoch > warmup
        shrinkage = epoch > warmup + fixed
        model.set_corrections_enabled(corrections)
        model.set_shrinkage_learnable(shrinkage)
        model.train()
        totals = {key: 0.0 for key in ("main", "total", "protein_aux", "drug_utility", "joint_utility", "gate_prior")}
        diagnostics = {key: 0.0 for key in ("drug_gate", "joint_gate", "drug_rms", "joint_rms", "attention_confidence")}
        items = 0
        if device.type == 'cuda': torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        for drug_batch, protein_batch, target in loader:
            drug_batch = drug_batch.float().to(device)
            protein_batch = protein_batch.float().to(device)
            target = target.float().to(device).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else contextlib.nullcontext()
            with autocast:
                prediction, aux = model(protein_batch, drug_batch)
                main_loss = torch.mean((prediction.reshape(-1) - target) ** 2)
                components = model.loss_components(target, aux)
                loss = main_loss + components["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            if amp_enabled:
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), mgca.MAX_GRAD_NORM, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(target)
            items += count
            global_step += 1
            totals["main"] += float(main_loss.detach()) * count
            totals["total"] += float(loss.detach()) * count
            for key in ("protein_aux", "drug_utility", "joint_utility", "gate_prior"):
                totals[key] += float(components[key].detach()) * count
            diagnostics["drug_gate"] += float(aux["drug_gate"].detach().sum())
            diagnostics["joint_gate"] += float(aux["joint_gate"].detach().sum())
            diagnostics["drug_rms"] += float(aux["drug_contribution_rms"].detach().sum())
            diagnostics["joint_rms"] += float(aux["joint_contribution_rms"].detach().sum())
            diagnostics["attention_confidence"] += float(aux["joint_attention_confidence"].detach().sum())
        if device.type == 'cuda': torch.cuda.synchronize(device)
        row = {
            "epoch": epoch,
            "optimizer_steps": global_step,
            "corrections_enabled": corrections,
            "shrinkage_learnable": shrinkage,
            "train_main_mse": totals["main"] / items,
            "train_total_loss": totals["total"] / items,
            "train_protein_aux": totals["protein_aux"] / items,
            "train_drug_utility": totals["drug_utility"] / items,
            "train_joint_utility": totals["joint_utility"] / items,
            "train_gate_prior": totals["gate_prior"] / items,
            "drug_gate_mean": diagnostics["drug_gate"] / items,
            "joint_gate_mean": diagnostics["joint_gate"] / items,
            "drug_contribution_rms_mean": diagnostics["drug_rms"] / items,
            "joint_contribution_rms_mean": diagnostics["joint_rms"] / items,
            "joint_attention_confidence_mean": diagnostics["attention_confidence"] / items,
            "gradient_norm_last_before_clip": float(gradient_norm.detach()),
            "epoch_duration_sec": time.perf_counter() - epoch_started,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        common.atomic_csv(args.output_dir / "training_history.csv", history)
        common.atomic_torch_save(
            torch,
            resume_path,
            {
                "run_identity": identity,
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": rng_state(generator),
                "history": history,
            },
        )
        print(
            f"seed={args.seed} epoch={epoch:02d}/{args.epochs} "
            f"mse={row['train_main_mse']:.6f} alphaD={row['drug_gate_mean']:.5f} "
            f"alphaJ={row['joint_gate_mean']:.5f} steps={global_step}"
        )
    training_seconds = sum(r['epoch_duration_sec'] for r in history)
    final_metrics = evaluate(model, eval_loader, device)
    model.set_corrections_enabled(True)
    model.set_shrinkage_learnable(True)
    checkpoint_path = args.output_dir / checkpoint_name
    checkpoint = {
        "checkpoint_type": "mgca_final_2773_full_refit_final_state",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity": identity,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": common.checkpoint_config_from_frozen(config, args.epochs, args.seed),
        "data": {
            "csv_path": str(args.data_csv.resolve()),
            "csv_sha256": identity["data_sha256"],
            "manifest_path": str(args.data_manifest.resolve()),
            "manifest_sha256": identity["data_manifest_sha256"],
            "row_count": 2773,
            "case_exact_fasta_overlap_count": 0,
        },
        "code": {
            "trainer_path": str(Path(__file__).resolve()),
            "trainer_sha256": common.sha256_file(Path(__file__).resolve()),
            "model_path": str(args.model_file.resolve()),
            "model_sha256": identity["model_sha256"],
            "frozen_config_path": str(args.config_json.resolve()),
            "frozen_config_sha256": identity["config_sha256"],
        },
        "training": {
            "fixed_epoch_budget": args.epochs,
            "budget_source": config['epoch_rule'],
            "validation_selected_epochs": config['validation_selected_epochs'],
            "early_stopping": False,
            "validation_or_test_loader_constructed": False,
            "optimizer_steps": global_step,
            "feature_preparation_seconds": feature_seconds,
            "training_seconds": training_seconds,
            "timing_definition": "sum of synchronized training epoch durations, excludes checkpoint writes",
            "scalar_weight_decay": 0.0,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0.0,
            "configured_concurrency": args.concurrency,
            "amp": amp_enabled,
        },
        "history": history,
        "final_full_training_metrics": final_metrics,
        "parameter_count": mgca.parameter_count(model),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
        },
    }
    common.atomic_torch_save(torch, checkpoint_path, checkpoint)
    reload_payload = common.torch_load(torch, checkpoint_path, map_location=device)
    reload_model = mgca.FullRegressionTransformer(**model_kwargs).to(device)
    reload_model.load_state_dict(reload_payload["model_state_dict"], strict=True)
    sample_count = min(8, len(dataset))
    model.eval(); reload_model.eval()
    with torch.inference_mode():
        before, _ = model(protein[:sample_count].float().to(device), ligand[:sample_count].float().to(device))
        after, _ = reload_model(protein[:sample_count].float().to(device), ligand[:sample_count].float().to(device))
    reload_difference = float((before - after).abs().max().cpu())
    if reload_difference > 1e-6:
        raise RuntimeError(f"Reload prediction mismatch: {reload_difference}")
    checkpoint_hash = common.sha256_file(checkpoint_path)
    predicted, aux = common.predict_with_aux(model, protein, ligand, device, config['params']['batch_size'])
    observed = labels.detach().cpu().reshape(-1).numpy()
    common.atomic_csv(args.output_dir/'training_predictions.csv', [
        {'source_row':i,'sample_id':common.sha256_text(rows[i][0]+'\0'+rows[i][1])[:24],
         'y_true':float(y),'y_pred':float(p),'error':float(p-y),'abs_error':float(abs(p-y))}
        for i,(y,p) in enumerate(zip(observed,predicted))])
    aux_path=args.output_dir/'training_diagnostics.npz'
    with aux_path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle,**aux)
    os.replace(aux_path.with_suffix('.tmp'),aux_path)
    run_manifest = {
        "run_identity": identity,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "final_full_training_metrics": final_metrics,
        "reload_max_abs_difference": reload_difference,
        "files": {name:common.sha256_file(args.output_dir/name) for name in
                  [checkpoint_name,'training_history.csv','training_predictions.csv','training_diagnostics.npz']},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    common.atomic_json(args.output_dir / "run_manifest.json", run_manifest)
    common.atomic_text(args.output_dir / ".complete", checkpoint_hash + "\n")
    print(f"Completed checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
