#!/usr/bin/env python3
"""Two-stage warm-only Bayesian hyperparameter selection for MGCA.

Stage 1 runs Optuna TPE on warm run 1. Stage 2 evaluates the top-k unique
configurations on all five warm validation splits. The configuration with the
lowest five-run mean validation MSE is frozen. Test CSVs are never loaded.
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import socket
import statistics
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINE_ROOT = SCRIPT_DIR.parent
# The bundle is expected at <Bio project>/ONLINE. Override BIO_PROJECT_ROOT if
# it is copied elsewhere while the datasets remain in another project root.
DATA_ROOT = Path(os.environ.get("BIO_PROJECT_ROOT", ONLINE_ROOT)).resolve()
MGCA_SCRIPT = SCRIPT_DIR / "ESM_Morgan_Hybrid_Fusion_nonredundant.py"
MODEL_DEPENDENCY_SCRIPT = SCRIPT_DIR / "ESM_Morgan_Hybrid_Fusion.py"
ALIGNED_COMMON = ONLINE_ROOT / "baselines" / "common"
sys.path.insert(0, str(ALIGNED_COMMON))
from tuning_config import atomic_write_json, canonical_config_id


WEIGHT_DECAYS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
BATCH_SIZES = [16, 32, 64]
DROPOUTS = [0.05, 0.10, 0.15, 0.17, 0.20, 0.25, 0.30]
WINDOW_SIZES = [1, 2, 4, 6, 8]
WINDOW_LAYOUTS = ("legacy_anchors_v1", "even_span_v2")
DEFAULT_CONFIG = {
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "batch_size": 64,
    "dropout": 0.17,
    "window_size": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("KinetX", "2773"))
    parser.add_argument("--fingerprint", default="morgan", choices=("morgan", "fcfp"))
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "hyperparameter_tuning"))
    parser.add_argument("--mgca-script", type=Path, default=MGCA_SCRIPT)
    parser.add_argument(
        "--model-dependency-script",
        type=Path,
        default=MODEL_DEPENDENCY_SCRIPT,
        help="Additional implementation file whose hash is part of the immutable study signature.",
    )
    parser.add_argument(
        "--model-variant",
        default="mha_interaction_rmsnorm_sigmoid_fixed3h_2expert",
    )
    parser.add_argument(
        "--protocol-version",
        default="mgca_single_objective_warm_val_mse_tpe_top5_v1",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=WINDOW_SIZES,
        help="Categorical ESM2 window sizes searched by TPE",
    )
    parser.add_argument(
        "--window-layout",
        choices=WINDOW_LAYOUTS,
        default="legacy_anchors_v1",
        help="Fixed ESM2 depth-window placement strategy",
    )
    parser.add_argument(
        "--study-name-suffix",
        default="online_bundle_warm_mse_v1",
        help="Optuna study-name suffix; change it only with a new output root",
    )
    parser.add_argument("--esm2-path", default=str(DATA_ROOT.parent / "pretrained_model" / "esm2_t36"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-failed-trials", type=int, default=30)
    parser.add_argument("--search-jobs", type=int, default=1)
    parser.add_argument("--review-jobs", type=int, default=1)
    parser.add_argument(
        "--esm-batch-size", type=int, default=2,
        help="ESM2 feature-extraction batch size; execution-only, not a tuned hyperparameter",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--moe-num-experts", type=int, default=2)
    parser.add_argument(
        "--accept-model-hash-change",
        action="store_true",
        help=(
            "Resume an existing study only when the sole manifest difference is "
            "mgca_script_sha256; records an audited manifest migration"
        ),
    )
    return parser.parse_args()


def load_mgca_module(mgca_script: Path):
    if not mgca_script.is_file():
        raise FileNotFoundError(mgca_script)
    spec = importlib.util.spec_from_file_location("mgca_tuning_model", mgca_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import MGCA module from {mgca_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def warm_paths(dataset: str, run: int) -> tuple[Path, Path]:
    if dataset == "KinetX":
        root = DATA_ROOT / "KinetX" / "random_split_mgca_input"
    else:
        root = DATA_ROOT / "2773" / "new_folds" / "warm"
    train_csv = root / f"train_run{run}.csv"
    val_csv = root / f"val_run{run}.csv"
    for path in (train_csv, val_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    return train_csv, val_csv


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def search_space_description(
    moe_num_experts: int = 4,
    window_sizes=WINDOW_SIZES,
    window_layout: str = "legacy_anchors_v1",
) -> dict:
    return {
        "lr": {"type": "log_uniform", "range": [1e-5, 5e-4]},
        "weight_decay": {"type": "categorical", "values": WEIGHT_DECAYS},
        "batch_size": {"type": "categorical", "values": BATCH_SIZES},
        "dropout": {"type": "categorical", "values": DROPOUTS},
        "window_size": {
            "type": "categorical",
            "values": list(window_sizes),
            "description": "adjacent ESM2 layers averaged for each of four depth experts",
        },
        "window_layout": {
            "type": "fixed",
            "value": window_layout,
            "description": "placement of four ESM2 depth windows",
        },
        "fixed_architecture": {
            "hidden_dim": 512,
            "protein_experts": 4,
            "drug_experts": 4,
            "moe_experts": moe_num_experts,
            "attention_heads": 8,
        },
    }


def suggest_params(trial, window_sizes=WINDOW_SIZES) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", WEIGHT_DECAYS),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
        "dropout": trial.suggest_categorical("dropout", DROPOUTS),
        "window_size": trial.suggest_categorical("window_size", list(window_sizes)),
    }


def data_manifest(dataset: str) -> list[dict]:
    manifest = []
    for run in range(1, 6):
        train_csv, val_csv = warm_paths(dataset, run)
        manifest.append({
            "run": run,
            "train_csv": str(train_csv.resolve()),
            "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv.resolve()),
            "val_sha256": sha256_file(val_csv),
        })
    return manifest


def acquire_lock(study_root: Path) -> None:
    lock_path = study_root / "study.lock"
    token = {"pid": os.getpid(), "host": socket.gethostname(), "started_at_unix": time.time()}
    if lock_path.exists():
        try:
            previous = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
        active = previous.get("host") != socket.gethostname()
        if previous.get("host") == socket.gethostname() and isinstance(previous.get("pid"), int):
            if os.name == "nt":
                import ctypes
                process = ctypes.windll.kernel32.OpenProcess(0x1000, False, previous["pid"])
                if process:
                    code = ctypes.c_ulong()
                    ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(code))
                    ctypes.windll.kernel32.CloseHandle(process)
                    active = code.value == 259
            else:
                try:
                    os.kill(previous["pid"], 0)
                    active = True
                except OSError:
                    active = False
        if active:
            raise RuntimeError(f"Another tuning process may own {lock_path}: {previous}")
        lock_path.unlink()
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(token, handle)

    def release():
        try:
            if json.loads(lock_path.read_text(encoding="utf-8")) == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(release)


def resolve_device(requested: str, torch):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def prepare_split(
    module, dataset: str, run: int, fingerprint: str,
    window_size: int, window_layout: str, esm2_path: Path, device,
):
    train_csv, val_csv = warm_paths(dataset, run)
    train_rows = module.read_labeled_rows(str(train_csv))
    val_rows = module.read_labeled_rows(str(val_csv))
    merged = train_rows + val_rows
    cache_path = module.get_combined_esm_cache_path(
        [str(train_csv), str(val_csv)],
        window_size=window_size,
        window_layout=window_layout,
    )
    fasta, drug, labels, _ = module.preprocess_rows(
        merged,
        str(esm2_path),
        device,
        esm_cache=cache_path,
        cache_label=(
            f"MGCA tuning {dataset} warm run{run} train+val "
            f"ws={window_size} layout={window_layout}"
        ),
        fingerprint_type=fingerprint,
        window_size=window_size,
        window_layout=window_layout,
    )
    n_train = len(train_rows)
    if len(labels) != len(merged):
        raise RuntimeError(
            f"Preprocessing changed sample count for {dataset}/run{run}: "
            f"expected {len(merged)}, got {len(labels)}"
        )
    train_ds = module.ESM2MorganDataset(drug[:n_train], fasta[:n_train], labels[:n_train])
    val_ds = module.ESM2MorganDataset(drug[n_train:], fasta[n_train:], labels[n_train:])
    return train_ds, val_ds, train_csv, val_csv


def precompute_esm_caches(
    *, module, dataset: str, window_sizes, runs,
    window_layout: str, esm2_path: Path, device,
    phase_label: str, esm_batch_size: int,
) -> None:
    """Build requested ESM caches serially before parallel MGCA training.

    A top-k review may launch several worker threads. Without this barrier,
    workers that need the same missing cache each load a full ESM2-t36 model
    and may concurrently write the same file, causing host OOM or corruption.
    """
    import torch

    window_sizes = sorted({int(value) for value in window_sizes})
    runs = sorted({int(value) for value in runs})
    jobs = [(run, window_size) for window_size in window_sizes for run in runs]
    print(
        f"{phase_label} cache preflight: serially verifying/building "
        f"{len(jobs)} run-window ESM2 caches (windows={window_sizes})"
    )

    tokenizer = None
    esm_model = None
    try:
        for index, (run, window_size) in enumerate(jobs, 1):
            train_csv, val_csv = warm_paths(dataset, run)
            train_rows = module.read_labeled_rows(str(train_csv))
            val_rows = module.read_labeled_rows(str(val_csv))
            merged = train_rows + val_rows
            cache_path = Path(module.get_combined_esm_cache_path(
                [str(train_csv), str(val_csv)],
                window_size=window_size,
                window_layout=window_layout,
            ))

            cache_valid = False
            if cache_path.is_file():
                try:
                    cached = torch.load(str(cache_path), map_location="cpu")
                    cache_valid = (
                        hasattr(cached, "shape") and int(cached.shape[0]) == len(merged)
                    )
                    del cached
                except Exception as exc:
                    print(f"  [{index}/{len(jobs)}] invalid cache {cache_path.name}: {exc}")

            if cache_valid:
                print(f"  [{index}/{len(jobs)}] cached: run{run} ws={window_size}")
                continue

            print(f"  [{index}/{len(jobs)}] building: run{run} ws={window_size}")
            if esm_model is None:
                print(f"  Loading one shared ESM2 model for all missing {phase_label} caches")
                tokenizer = module.AutoTokenizer.from_pretrained(str(esm2_path))
                # ``low_cpu_mem_usage`` requires Accelerate.  Avoid first
                # attempting it when Accelerate is absent: that failed attempt
                # can leave shard allocations behind and make the fallback
                # peak-memory usage much larger.
                try:
                    accelerate_available = importlib.util.find_spec("accelerate") is not None
                except (ImportError, ValueError):
                    accelerate_available = False
                try:
                    if not accelerate_available:
                        raise ImportError("accelerate is not installed")
                    esm_model = module.AutoModelForMaskedLM.from_pretrained(
                        str(esm2_path), low_cpu_mem_usage=True
                    ).to(device)
                except (TypeError, ImportError):
                    esm_model = module.AutoModelForMaskedLM.from_pretrained(
                        str(esm2_path)
                    ).to(device)

            fasta_list = [row[0] for row in merged]
            features = module.batch_extract_esm2(
                fasta_list,
                tokenizer,
                esm_model,
                device,
                batch_size=esm_batch_size,
                window_size=window_size,
                window_layout=window_layout,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache = Path(f"{cache_path}.tmp.{os.getpid()}")
            try:
                torch.save(features.detach().cpu(), str(tmp_cache))
                os.replace(str(tmp_cache), str(cache_path))
            finally:
                if tmp_cache.exists():
                    tmp_cache.unlink()
            del features
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if esm_model is not None:
            del esm_model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"{phase_label} cache preflight complete")


def train_selection(
    *, module, dataset: str, fingerprint: str, run: int, params: dict,
    output_dir: Path, esm2_path: Path, device, seed: int,
    epochs: int, patience: int, hidden_dim: int, window_layout: str,
    moe_num_experts: int = 4,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if (output_dir / ".complete").is_file() and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected = {
            "dataset": dataset, "fingerprint": fingerprint, "run": run,
            "seed": seed, "epochs": epochs, "patience": patience,
            "hidden_dim": hidden_dim, "moe_num_experts": moe_num_experts,
            "params": params,
        }
        if metrics.get("identity") != expected:
            raise RuntimeError(f"Incompatible completed output: {output_dir}")
        return metrics

    module.set_seed(seed)
    train_ds, val_ds, train_csv, val_csv = prepare_split(
        module, dataset, run, fingerprint, int(params["window_size"]),
        window_layout, esm2_path, device
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=int(params["batch_size"]), shuffle=True,
        generator=generator, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(params["batch_size"]), shuffle=False, num_workers=0,
    )
    model = module.FullRegressionTransformer(
        proj_dim1=2560,
        proj_dim2=2048,
        hidden_dim=hidden_dim,
        dropout=float(params["dropout"]),
        nums_of_experts=4,
        num_heads=8,
        moe_num_experts=moe_num_experts,
        ablation="no",
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_mse = math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for drug_batch, protein_batch, labels in train_loader:
            drug_batch = drug_batch.float().to(device)
            protein_batch = protein_batch.float().to(device)
            labels = labels.float().to(device).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(protein_batch, drug_batch)
            loss = criterion(predictions.reshape(-1), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)

        model.eval()
        squared_error = 0.0
        absolute_error = 0.0
        count = 0
        with torch.no_grad():
            for drug_batch, protein_batch, labels in val_loader:
                drug_batch = drug_batch.float().to(device)
                protein_batch = protein_batch.float().to(device)
                labels = labels.float().to(device).reshape(-1)
                predictions, _ = model(protein_batch, drug_batch)
                residual = predictions.reshape(-1) - labels
                squared_error += float(torch.sum(residual * residual).item())
                absolute_error += float(torch.sum(torch.abs(residual)).item())
                count += len(labels)
        val_mse = squared_error / count
        val_mae = absolute_error / count
        history.append({
            "epoch": epoch,
            "train_mse": total_loss / total_items,
            "val_mse": val_mse,
            "val_rmse": math.sqrt(val_mse),
            "val_mae": val_mae,
        })
        if val_mse < best_mse - 1e-12:
            best_mse = val_mse
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        if patience > 0 and bad_epochs >= patience:
            break

    identity = {
        "dataset": dataset, "fingerprint": fingerprint, "run": run,
        "seed": seed, "epochs": epochs, "patience": patience,
        "hidden_dim": hidden_dim, "moe_num_experts": moe_num_experts,
        "params": params,
    }
    payload = {
        "identity": identity,
        "selection_only": True,
        "test_accessed": False,
        "esm2_window_size": int(params["window_size"]),
        "esm2_window_layout": window_layout,
        "esm2_cache_suffix": (
            f"__ws{int(params['window_size'])}"
            + ("" if window_layout == "legacy_anchors_v1" else f"__wl{window_layout}")
            + ".pt"
        ),
        "best_epoch": best_epoch,
        "val_metrics": {"mse": best_mse, "rmse": math.sqrt(best_mse)},
        "duration_seconds": time.time() - started,
        "input": {
            "train_csv": str(train_csv.resolve()), "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv.resolve()), "val_sha256": sha256_file(val_csv),
            "test_csv": None,
        },
        "history": history,
    }
    atomic_write_json(metrics_path, payload)
    (output_dir / ".complete").write_text("complete\n", encoding="utf-8")
    del model, optimizer, train_loader, val_loader, train_ds, val_ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def unique_top_trials(study, top_k: int):
    from optuna.trial import TrialState
    completed = sorted(
        (t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None),
        key=lambda t: t.value,
    )
    selected = []
    seen = set()
    for trial in completed:
        signature = json.dumps(trial.params, sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(trial)
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        raise RuntimeError(f"Need {top_k} unique completed trials, found {len(selected)}")
    return selected


def main() -> None:
    args = parse_args()
    if min(args.n_trials, args.top_k, args.max_failed_trials, args.search_jobs, args.review_jobs) <= 0:
        raise ValueError("Trial counts and job counts must be positive")
    if args.esm_batch_size <= 0:
        raise ValueError("esm-batch-size must be positive")
    if args.n_trials < args.top_k:
        raise ValueError("n-trials must be at least top-k")
    if args.hidden_dim != 512:
        raise ValueError("For fair hyperparameter selection, hidden_dim must remain fixed at 512")
    if args.moe_num_experts <= 0:
        raise ValueError("moe-num-experts must be positive")
    if not args.window_sizes or any(value <= 0 for value in args.window_sizes):
        raise ValueError("window-sizes must contain positive integers")
    if len(set(args.window_sizes)) != len(args.window_sizes):
        raise ValueError(f"window-sizes must be unique: {args.window_sizes}")
    args.window_sizes = tuple(args.window_sizes)
    if args.search_jobs > 1 and args.device != "cpu":
        print(
            "WARNING: parallel TPE jobs share one accelerator and proposal order becomes "
            "scheduling-dependent. Use --search-jobs 1 for the paper run.",
            file=sys.stderr,
        )
    try:
        import optuna
        from optuna.trial import TrialState
        import torch
    except ImportError as exc:
        raise ImportError("MGCA tuning requires torch and optuna") from exc

    esm2_path = Path(args.esm2_path).resolve()
    if not esm2_path.exists():
        raise FileNotFoundError(f"ESM2 model path does not exist: {esm2_path}")
    device = resolve_device(args.device, torch)
    mgca_script = args.mgca_script.resolve()
    model_dependency_script = args.model_dependency_script.resolve()
    if not model_dependency_script.is_file():
        raise FileNotFoundError(model_dependency_script)
    module = load_mgca_module(mgca_script)
    study_root = Path(args.output_root).resolve() / f"mgca_{args.fingerprint}" / args.dataset
    stage1_root = study_root / "stage1_run1"
    review_root = study_root / "top_candidates_review"
    study_root.mkdir(parents=True, exist_ok=True)
    acquire_lock(study_root)

    manifest = {
        "protocol_version": args.protocol_version,
        "model": "MGCA",
        "fingerprint": args.fingerprint,
        "dataset": args.dataset,
        "target_completed_trials": args.n_trials,
        "top_k": args.top_k,
        "sampler": "Optuna TPESampler",
        "sampler_seed": args.sampler_seed,
        "search_space": search_space_description(
            args.moe_num_experts, args.window_sizes, args.window_layout
        ),
        "training": {
            "epochs": args.epochs, "patience": args.patience,
            "seed": args.seed, "optimizer": "AdamW",
            "objective": "validation_mse",
        },
        "esm2_path": str(esm2_path),
        "esm2_path_exists": True,
        "mgca_script": str(mgca_script),
        "mgca_script_sha256": sha256_file(mgca_script),
        "warm_data_manifest": data_manifest(args.dataset),
        "test_accessed_during_selection": False,
        "esm2_window_layout": args.window_layout,
    }
    is_extended_model_signature = (
        args.model_variant != "legacy_mgca" or model_dependency_script != mgca_script
    )
    if is_extended_model_signature:
        manifest.update({
            "model_variant": args.model_variant,
            "model_dependency_script": str(model_dependency_script),
            "model_dependency_script_sha256": sha256_file(model_dependency_script),
        })
    current_manifest_signature = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = study_root / "study_manifest.json"
    manifest_migrated = False
    old_manifest_signature = None
    approved_signature_migration_from = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            changed_keys = sorted(
                key
                for key in set(existing) | set(manifest)
                if existing.get(key) != manifest.get(key)
            )
            print("Study manifest differences:", file=sys.stderr)
            for key in changed_keys:
                print(
                    f"  {key}: existing={existing.get(key)!r}; current={manifest.get(key)!r}",
                    file=sys.stderr,
                )
            if args.accept_model_hash_change and changed_keys == ["mgca_script_sha256"]:
                old_manifest_signature = hashlib.sha256(
                    json.dumps(existing, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                audit_path = study_root / "manifest_migrations.json"
                if audit_path.is_file():
                    migrations = json.loads(audit_path.read_text(encoding="utf-8"))
                    if not isinstance(migrations, list):
                        raise RuntimeError(f"Invalid manifest migration audit: {audit_path}")
                else:
                    migrations = []
                migrations.append({
                    "timestamp_unix": time.time(),
                    "reason": "explicit resume after cache-only tuning-driver update",
                    "changed_keys": changed_keys,
                    "old_mgca_script_sha256": existing.get("mgca_script_sha256"),
                    "new_mgca_script_sha256": manifest.get("mgca_script_sha256"),
                    "old_manifest_signature_sha256": old_manifest_signature,
                    "new_manifest_signature_sha256": current_manifest_signature,
                })
                atomic_write_json(audit_path, migrations)
                atomic_write_json(manifest_path, manifest)
                manifest_migrated = True
                print(
                    "Accepted the sole MGCA source-hash difference and recorded an audited "
                    "manifest migration.",
                    file=sys.stderr,
                )
            else:
                hint = (
                    " If and only if the sole difference shown above is mgca_script_sha256, "
                    "rerun with --accept-model-hash-change."
                )
                raise RuntimeError(
                    f"Study signature changed; use a new --output-root instead of mixing "
                    f"trials: {study_root}.{hint}"
                )
        else:
            # Recover safely if the process stopped after writing the audited
            # manifest migration but before updating the Optuna user attribute.
            audit_path = study_root / "manifest_migrations.json"
            if audit_path.is_file():
                migrations = json.loads(audit_path.read_text(encoding="utf-8"))
                if isinstance(migrations, list) and migrations:
                    latest = migrations[-1]
                    if latest.get("new_manifest_signature_sha256") == current_manifest_signature:
                        approved_signature_migration_from = latest.get(
                            "old_manifest_signature_sha256"
                        )
    else:
        atomic_write_json(manifest_path, manifest)

    storage_path = study_root / "study.db"
    if not storage_path.exists() and stage1_root.exists() and any(stage1_root.iterdir()):
        raise RuntimeError(f"Stage-1 outputs exist but study.db is missing: {study_root}")
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed, n_startup_trials=min(10, args.n_trials)
    )
    study_name = f"mgca_{args.fingerprint}_{args.dataset}_{args.study_name_suffix}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    immutable_signature = current_manifest_signature
    previous_signature = study.user_attrs.get("immutable_signature_sha256")
    if previous_signature and previous_signature != immutable_signature:
        if (
            (manifest_migrated and previous_signature == old_manifest_signature)
            or previous_signature == approved_signature_migration_from
        ):
            study.set_user_attr("immutable_signature_sha256", immutable_signature)
        else:
            raise RuntimeError("Optuna study signature does not match study_manifest.json")
    if previous_signature is None:
        if study.trials:
            raise RuntimeError("Existing trials have no immutable signature; use a fresh output root")
        study.set_user_attr("immutable_signature_sha256", immutable_signature)
    if not study.trials:
        study.enqueue_trial(DEFAULT_CONFIG, user_attrs={"source": "current_MGCA_default"})

    def objective(trial):
        params = suggest_params(trial, args.window_sizes)
        trial_dir = stage1_root / f"trial_{trial.number:04d}"
        trial.set_user_attr("output_dir", str(trial_dir))
        try:
            metrics = train_selection(
                module=module, dataset=args.dataset, fingerprint=args.fingerprint,
                run=1, params=params, output_dir=trial_dir,
                esm2_path=esm2_path, device=device, seed=args.seed,
                epochs=args.epochs, patience=args.patience, hidden_dim=args.hidden_dim,
                window_layout=args.window_layout,
                moe_num_experts=args.moe_num_experts,
            )
        except RuntimeError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise
        value = float(metrics["val_metrics"]["mse"])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite validation MSE: {trial_dir}")
        trial.set_user_attr("best_epoch", int(metrics["best_epoch"]))
        return value

    completed_before = sum(t.state == TrialState.COMPLETE for t in study.trials)
    if completed_before > args.n_trials:
        raise RuntimeError(
            f"Study already has {completed_before} completed trials, exceeding fixed budget "
            f"{args.n_trials}; keep the original budget or use a new output root"
        )
    if completed_before < args.n_trials:
        # Build every requested run-1 window variant once. This keeps SEARCH_JOBS > 1
        # from loading multiple ESM2-t36 copies or racing on a shared cache.
        precompute_esm_caches(
            module=module,
            dataset=args.dataset,
            window_sizes=args.window_sizes,
            runs=(1,),
            window_layout=args.window_layout,
            esm2_path=esm2_path,
            device=device,
            phase_label="Stage 1",
            esm_batch_size=args.esm_batch_size,
        )
    while True:
        complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
        failed = sum(t.state == TrialState.FAIL for t in study.trials)
        remaining = args.n_trials - complete
        if remaining <= 0:
            break
        if failed >= args.max_failed_trials:
            raise RuntimeError(
                f"Aborting after {failed} failed trials with {complete}/{args.n_trials} complete"
            )
        attempts = min(remaining, args.max_failed_trials - failed)
        print(f"Stage 1: {complete} complete, {failed} failed; launching {attempts} attempts")
        study.optimize(objective, n_trials=attempts, n_jobs=args.search_jobs, catch=(RuntimeError,))

    complete = sum(t.state == TrialState.COMPLETE for t in study.trials)
    failed = sum(t.state == TrialState.FAIL for t in study.trials)
    trial_rows = []
    for trial in study.trials:
        trial_rows.append({
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            **trial.params,
            "best_epoch": trial.user_attrs.get("best_epoch"),
            "output_dir": trial.user_attrs.get("output_dir"),
        })
    write_csv(study_root / "trials.csv", trial_rows)
    top_trials = unique_top_trials(study, args.top_k)

    # ESM2 extraction is deliberately serialized. REVIEW_JOBS controls only
    # the much smaller downstream MGCA training after all caches are complete.
    precompute_esm_caches(
        module=module,
        dataset=args.dataset,
        window_sizes=(trial.params["window_size"] for trial in top_trials),
        runs=range(1, 6),
        window_layout=args.window_layout,
        esm2_path=esm2_path,
        device=device,
        phase_label="Stage 2",
        esm_batch_size=args.esm_batch_size,
    )

    tasks = [
        (rank, trial, run)
        for rank, trial in enumerate(top_trials, 1)
        for run in range(1, 6)
    ]

    def review(task):
        rank, trial, run = task
        output_dir = review_root / f"rank_{rank:02d}_trial_{trial.number:04d}" / f"run{run}"
        metrics = train_selection(
            module=module, dataset=args.dataset, fingerprint=args.fingerprint,
            run=run, params=trial.params, output_dir=output_dir,
            esm2_path=esm2_path, device=device, seed=args.seed,
            epochs=args.epochs, patience=args.patience, hidden_dim=args.hidden_dim,
            window_layout=args.window_layout,
            moe_num_experts=args.moe_num_experts,
        )
        return {
            "candidate_rank": rank,
            "trial_number": trial.number,
            "run": run,
            "stage1_val_mse": float(trial.value),
            "val_mse": float(metrics["val_metrics"]["mse"]),
            "best_epoch": int(metrics["best_epoch"]),
            **trial.params,
        }

    print(f"Stage 2: reviewing top {args.top_k} candidates on five warm validation splits")
    review_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.review_jobs) as executor:
        futures = [executor.submit(review, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            review_rows.append(future.result())
    review_rows.sort(key=lambda row: (row["candidate_rank"], row["run"]))
    write_csv(study_root / "candidate_validation.csv", review_rows)

    summaries = []
    for rank, trial in enumerate(top_trials, 1):
        values = [row["val_mse"] for row in review_rows if row["candidate_rank"] == rank]
        summaries.append({
            "candidate_rank": rank,
            "trial_number": trial.number,
            "stage1_val_mse": float(trial.value),
            "mean_val_mse": statistics.fmean(values),
            "std_val_mse": statistics.stdev(values),
            "min_val_mse": min(values),
            "max_val_mse": max(values),
            **trial.params,
        })
    summaries.sort(
        key=lambda row: (
            row["mean_val_mse"], row["std_val_mse"],
            row["stage1_val_mse"], row["trial_number"],
        )
    )
    write_csv(study_root / "top_candidates_summary.csv", summaries)
    winner = summaries[0]
    winner_trial = next(t for t in top_trials if t.number == winner["trial_number"])

    config = {
        "model": "MGCA",
        "fingerprint": args.fingerprint,
        "dataset": args.dataset,
        "selection_protocol": "warm_only_two_stage",
        "protocol_version": args.protocol_version,
        "test_accessed_during_selection": False,
        "source_trial_number": int(winner_trial.number),
        "params": {
            key: float(value) if isinstance(value, float) else value
            for key, value in winner_trial.params.items()
        },
        "fixed_architecture": {
            "hidden_dim": args.hidden_dim,
            "protein_experts": 4,
            "drug_experts": 4,
            "moe_experts": args.moe_num_experts,
            "attention_heads": 8,
        },
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "optimizer": "AdamW",
        },
        "stage1": {
            "target_completed_trials": args.n_trials,
            "actual_completed_trials": complete,
            "failed_trials": failed,
            "split": "warm_run1",
            "objective": "validation_mse",
            "sampler": "Optuna TPESampler",
            "sampler_seed": args.sampler_seed,
            "selected_trial_val_mse": float(winner_trial.value),
        },
        "stage2": {
            "top_k": args.top_k,
            "validation_runs": [1, 2, 3, 4, 5],
            "selection_metric": "mean_validation_mse",
        },
        "validation": {
            "mean_mse": float(winner["mean_val_mse"]),
            "std_mse": float(winner["std_val_mse"]),
        },
        "search_space": search_space_description(
            args.moe_num_experts, args.window_sizes, args.window_layout
        ),
        "warm_data_manifest": manifest["warm_data_manifest"],
        "generated_at_unix": time.time(),
        "esm2_window_layout": args.window_layout,
    }
    if is_extended_model_signature:
        config.update({
            "model_variant": args.model_variant,
            "mgca_script": str(mgca_script),
            "mgca_script_sha256": manifest["mgca_script_sha256"],
            "model_dependency_script": str(model_dependency_script),
            "model_dependency_script_sha256": manifest["model_dependency_script_sha256"],
        })
    config["training_cli_args"] = [
        "--lr", str(config["params"]["lr"]),
        "--weight_decay", str(config["params"]["weight_decay"]),
        "--batch_size", str(config["params"]["batch_size"]),
        "--dropout", str(config["params"]["dropout"]),
        "--window_size", str(config["params"]["window_size"]),
        "--window_layout", args.window_layout,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--val_freq", "1",
    ]
    config["config_id"] = canonical_config_id(config)
    atomic_write_json(study_root / "best_params.json", config)
    write_csv(study_root / "best_params.csv", [{
        "config_id": config["config_id"],
        **config["params"],
        "window_layout": args.window_layout,
        **config["training"],
        "mean_val_mse": config["validation"]["mean_mse"],
        "std_val_mse": config["validation"]["std_mse"],
    }])
    (study_root / "best_params.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"MGCA_CONFIG_ID='{config['config_id']}'\n"
        f"MGCA_LR='{config['params']['lr']}'\n"
        f"MGCA_WEIGHT_DECAY='{config['params']['weight_decay']}'\n"
        f"MGCA_BATCH_SIZE='{config['params']['batch_size']}'\n"
        f"MGCA_DROPOUT='{config['params']['dropout']}'\n"
        f"MGCA_WINDOW_SIZE='{config['params']['window_size']}'\n"
        f"MGCA_WINDOW_LAYOUT='{args.window_layout}'\n"
        f"MGCA_EPOCHS='{args.epochs}'\n"
        f"MGCA_PATIENCE='{args.patience}'\n"
        f"MGCA_MOE_NUM_EXPERTS='{args.moe_num_experts}'\n"
        "MGCA_VAL_FREQ='1'\n",
        encoding="utf-8",
    )
    (study_root / ".complete").write_text(config["config_id"] + "\n", encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
