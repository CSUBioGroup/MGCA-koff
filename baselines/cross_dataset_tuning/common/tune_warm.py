#!/usr/bin/env python3
"""Warm-only Bayesian tuning followed by top-k five-validation review."""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import hashlib
import json
import math
import subprocess
import sys
import time
import os
import socket
from pathlib import Path

import pandas as pd


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_ROOT.parent
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train.py")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tuning_config import atomic_write_json, canonical_config_id


MODEL_DEFAULTS = {
    "deepdta": {
        "epochs": 300, "patience": 30,
        "published": {"lr": 1e-3, "weight_decay": 1e-2, "batch_size": 64, "dropout": 0.1},
    },
    "graphdta": {
        "epochs": 1000, "patience": 50,
        "published": {"lr": 5e-4, "weight_decay": 0.0, "batch_size": 64, "dropout": 0.2},
    },
    "attentiondta": {
        "epochs": 300, "patience": 30,
        "published": {"lr": 1e-3, "weight_decay": 1e-2, "batch_size": 64, "dropout": 0.1},
    },
}

WEIGHT_DECAYS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def optimizer_name_for_model(model: str) -> str:
    return "Adam" if model == "graphdta" else "AdamW"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_DEFAULTS))
    parser.add_argument("--dataset", required=True, choices=("KinetX", "2773"))
    parser.add_argument("--output-root", default=str(BASELINE_ROOT / "hyperparameter_tuning"))
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Target number of successfully completed stage-1 trials")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-failed-trials", type=int, default=30,
                        help="Abort after this many failed/OOM stage-1 trials")
    parser.add_argument("--search-jobs", type=int, default=1)
    parser.add_argument("--review-jobs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser.parse_args()


def warm_paths(dataset: str, run: int) -> tuple[Path, Path]:
    if dataset == "KinetX":
        root = PROJECT_ROOT / "KinetX" / "random_split_mgca_input"
    else:
        root = PROJECT_ROOT / "2773" / "new_folds" / "warm"
    train_csv = root / f"train_run{run}.csv"
    val_csv = root / f"val_run{run}.csv"
    for path in (train_csv, val_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    return train_csv, val_csv


def suggest_params(trial, model: str) -> dict:
    if model == "deepdta":
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch_sizes = [32, 64, 128]
        dropouts = [0.0, 0.1, 0.2, 0.3]
    elif model == "graphdta":
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_sizes = [32, 64, 128]
        dropouts = [0.0, 0.1, 0.2, 0.3, 0.5]
    else:
        lr = trial.suggest_float("lr", 1e-5, 2e-3, log=True)
        batch_sizes = [8, 16, 32, 64]
        dropouts = [0.0, 0.1, 0.2, 0.3]
    return {
        "lr": lr,
        "weight_decay": trial.suggest_categorical("weight_decay", WEIGHT_DECAYS),
        "batch_size": trial.suggest_categorical("batch_size", batch_sizes),
        "dropout": trial.suggest_categorical("dropout", dropouts),
    }


def search_space_description(model: str) -> dict:
    if model == "deepdta":
        lr, batch_sizes, dropouts = [1e-5, 3e-3], [32, 64, 128], [0.0, 0.1, 0.2, 0.3]
    elif model == "graphdta":
        lr, batch_sizes, dropouts = [1e-5, 1e-3], [32, 64, 128], [0.0, 0.1, 0.2, 0.3, 0.5]
    else:
        lr, batch_sizes, dropouts = [1e-5, 2e-3], [8, 16, 32, 64], [0.0, 0.1, 0.2, 0.3]
    return {
        "lr": {"type": "log_uniform", "range": lr},
        "weight_decay": {"type": "categorical", "values": WEIGHT_DECAYS},
        "batch_size": {"type": "categorical", "values": batch_sizes},
        "dropout": {"type": "categorical", "values": dropouts},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_warm_manifest(dataset: str) -> list[dict]:
    manifest = []
    for run in range(1, 6):
        train_csv, val_csv = warm_paths(dataset, run)
        manifest.append({
            "run": run,
            "train_csv": str(train_csv.resolve()), "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv.resolve()), "val_sha256": sha256_file(val_csv),
        })
    return manifest


def acquire_study_lock(study_root: Path) -> None:
    lock_path = study_root / "study.lock"
    token = {"pid": os.getpid(), "host": socket.gethostname(), "started_at_unix": time.time()}
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        active = False
        if existing.get("host") == socket.gethostname() and isinstance(existing.get("pid"), int):
            if os.name == "nt":
                import ctypes
                process = ctypes.windll.kernel32.OpenProcess(0x1000, False, existing["pid"])
                if process:
                    exit_code = ctypes.c_ulong()
                    ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code))
                    ctypes.windll.kernel32.CloseHandle(process)
                    active = exit_code.value == 259  # STILL_ACTIVE
            else:
                try:
                    os.kill(existing["pid"], 0)
                    active = True
                except OSError:
                    active = False
        foreign_host = existing.get("host") not in (None, socket.gethostname())
        if active:
            raise RuntimeError(f"Another tuning process owns {lock_path}: {existing}")
        if foreign_host:
            if os.environ.get("ALLOW_STALE_STUDY_LOCK") != "1":
                raise RuntimeError(
                    f"Another tuning process may own {lock_path}: {existing}. "
                    "If the recorded host/container has definitely stopped, rerun with "
                    "ALLOW_STALE_STUDY_LOCK=1 to archive the stale lock and resume."
                )
            safe_host = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in str(existing.get("host", "unknown"))
            )
            stale_path = lock_path.with_name(
                f"study.lock.stale.{safe_host}.{existing.get('pid', 'unknown')}."
                f"{int(time.time())}.json"
            )
            lock_path.replace(stale_path)
            print(f"Archived explicitly confirmed stale study lock: {stale_path}")
        else:
            lock_path.unlink()
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another tuning process acquired {lock_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(token, handle)

    def release():
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current == token:
                lock_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(release)


def params_match(stored: dict, expected: dict) -> bool:
    for key, value in expected.items():
        if key not in stored:
            return False
        if isinstance(value, float):
            if not math.isclose(float(stored[key]), value, rel_tol=1e-12, abs_tol=1e-15):
                return False
        elif stored[key] != value:
            return False
    return True


def run_selection(
    *, model: str, dataset: str, run: int, params: dict, output_dir: Path,
    device: str, num_workers: int, seed: int, epochs: int, patience: int,
    keep_checkpoints: bool,
) -> dict:
    train_csv, val_csv = warm_paths(dataset, run)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if (output_dir / ".complete").is_file() and metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        expected = {
            **params, "epochs": epochs, "patience": patience,
            "optimizer": "Adam" if model == "graphdta" else "AdamW",
        }
        expected_input = {
            "train_csv": str(train_csv.resolve()), "train_sha256": sha256_file(train_csv),
            "val_csv": str(val_csv.resolve()), "val_sha256": sha256_file(val_csv),
            "test_csv": None, "test_sha256": None,
        }
        identity_matches = (
            metrics.get("model") == model and metrics.get("dataset") == dataset
            and metrics.get("split") == "warm" and metrics.get("run") == run
            and metrics.get("seed") == seed and metrics.get("input") == expected_input
        )
        if (not metrics.get("selection_only") or not identity_matches
                or not params_match(metrics.get("hyperparameters", {}), expected)):
            raise RuntimeError(f"Completed tuning output has incompatible parameters: {output_dir}")
        return metrics

    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--model", model,
        "--dataset", dataset,
        "--split", "warm",
        "--run", str(run),
        "--seed", str(seed),
        "--train-csv", str(train_csv),
        "--val-csv", str(val_csv),
        "--output-dir", str(output_dir),
        "--device", device,
        "--num-workers", str(num_workers),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--lr", str(params["lr"]),
        "--weight-decay", str(params["weight_decay"]),
        "--batch-size", str(params["batch_size"]),
        "--dropout", str(params["dropout"]),
        "--selection-only",
    ]
    if not keep_checkpoints:
        command.append("--discard-checkpoint-after-eval")
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Training failed with exit {completed.returncode}; see {log_path}")
    with metrics_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def unique_top_trials(study, top_k: int):
    from optuna.trial import TrialState

    completed = sorted(
        (trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None),
        key=lambda trial: trial.value,
    )
    result = []
    seen = set()
    for trial in completed:
        key = json.dumps(trial.params, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        result.append(trial)
        if len(result) == top_k:
            break
    if len(result) < top_k:
        raise RuntimeError(f"Need {top_k} unique completed trials, found {len(result)}")
    return result


def main():
    args = parse_args()
    if min(args.n_trials, args.top_k, args.search_jobs, args.review_jobs, args.max_failed_trials) <= 0:
        raise ValueError("trial counts and job counts must be positive")
    if args.n_trials < args.top_k:
        raise ValueError("n-trials must be at least top-k")
    if args.search_jobs > 1 and args.device != "cpu":
        print(
            "WARNING: multiple Optuna jobs share the requested device. This may cause OOM and makes "
            "TPE proposal order scheduling-dependent; use --search-jobs 1 for the paper run.",
            file=sys.stderr,
        )
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        raise ImportError("Warm-only tuning requires Optuna: pip install optuna") from exc

    defaults = MODEL_DEFAULTS[args.model]
    epochs = args.epochs or defaults["epochs"]
    patience = args.patience or defaults["patience"]
    optimizer_name = optimizer_name_for_model(args.model)
    study_root = Path(args.output_root).resolve() / args.model / args.dataset
    stage1_root = study_root / "stage1_run1"
    review_root = study_root / "top_candidates_review"
    study_root.mkdir(parents=True, exist_ok=True)
    acquire_study_lock(study_root)
    data_manifest = build_warm_manifest(args.dataset)

    study_manifest = {
        "protocol_version": "warm_only_tpe_top5_v1",
        "model": args.model, "dataset": args.dataset,
        "target_completed_trials": args.n_trials, "top_k": args.top_k,
        "sampler": "Optuna TPESampler", "sampler_seed": args.sampler_seed,
        "search_space": search_space_description(args.model),
        "training": {"epochs": epochs, "patience": patience, "seed": args.seed,
                     "optimizer": optimizer_name},
        "warm_data_manifest": data_manifest,
    }
    manifest_path = study_root / "study_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        if existing_manifest != study_manifest:
            raise RuntimeError(
                f"Study signature changed; refusing to mix trials in {study_root}. "
                "Use a new --output-root for a different budget, search space, seed, or dataset version."
            )
    else:
        atomic_write_json(manifest_path, study_manifest)

    storage_path = study_root / "study.db"
    if not storage_path.exists() and stage1_root.exists() and any(stage1_root.iterdir()):
        raise RuntimeError(f"Stage-1 outputs exist but study.db is missing: {study_root}")
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{storage_path.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, n_startup_trials=min(10, args.n_trials))
    study = optuna.create_study(
        study_name=f"{args.model}_{args.dataset}_warm_v1",
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )
    signature = hashlib.sha256(
        json.dumps(study_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    existing_signature = study.user_attrs.get("immutable_signature_sha256")
    if existing_signature and existing_signature != signature:
        raise RuntimeError("Optuna study immutable signature does not match study_manifest.json")
    if existing_signature is None:
        if study.trials:
            raise RuntimeError("Existing Optuna trials have no immutable signature; use a fresh output root")
        study.set_user_attr("immutable_signature_sha256", signature)
    if not study.trials:
        study.enqueue_trial(defaults["published"], user_attrs={"source": "published_reimplementation_default"})

    def objective(trial):
        params = suggest_params(trial, args.model)
        trial_dir = stage1_root / f"trial_{trial.number:04d}"
        trial.set_user_attr("output_dir", str(trial_dir))
        metrics = run_selection(
            model=args.model, dataset=args.dataset, run=1, params=params, output_dir=trial_dir,
            device=args.device, num_workers=args.num_workers, seed=args.seed,
            epochs=epochs, patience=patience, keep_checkpoints=args.keep_checkpoints,
        )
        val_mse = float(metrics["val_metrics"]["mse"])
        if not math.isfinite(val_mse):
            raise RuntimeError(f"Non-finite validation MSE in {trial_dir}")
        trial.set_user_attr("best_epoch", metrics["best_epoch"])
        return val_mse

    completed_before = sum(trial.state == TrialState.COMPLETE for trial in study.trials)
    failed_before_run = sum(trial.state == TrialState.FAIL for trial in study.trials)
    if completed_before > args.n_trials:
        raise RuntimeError(
            f"Study already contains {completed_before} completed trials, exceeding fixed budget {args.n_trials}"
        )
    while True:
        completed_now = sum(trial.state == TrialState.COMPLETE for trial in study.trials)
        failed_now = sum(trial.state == TrialState.FAIL for trial in study.trials)
        failed_this_run = failed_now - failed_before_run
        remaining = max(0, args.n_trials - completed_now)
        if remaining == 0:
            break
        if failed_this_run >= args.max_failed_trials:
            raise RuntimeError(
                f"Aborting after {failed_this_run} new failed trials "
                f"({failed_now} total) with only {completed_now}/{args.n_trials} complete. "
                "Inspect stage1 train.log files, reduce the search space if OOM is systematic, and resume."
            )
        attempts = min(remaining, args.max_failed_trials - failed_this_run)
        print(
            f"Stage 1: {completed_now} complete, {failed_now} total failed "
            f"({failed_this_run} this run); launching {attempts} attempts"
        )
        study.optimize(objective, n_trials=attempts, n_jobs=args.search_jobs, catch=(RuntimeError,))
    completed_after = sum(trial.state == TrialState.COMPLETE for trial in study.trials)
    failed_after = sum(trial.state == TrialState.FAIL for trial in study.trials)
    if completed_after < args.top_k:
        raise RuntimeError(
            f"Only {completed_after} trials completed successfully; need at least {args.top_k}. "
            "Inspect stage1 logs and rerun to add replacement trials."
        )
    study.trials_dataframe().to_csv(study_root / "trials.csv", index=False)
    top_trials = unique_top_trials(study, args.top_k)

    tasks = []
    for rank, trial in enumerate(top_trials, start=1):
        for run in range(1, 6):
            tasks.append((rank, trial, run))

    def review_task(task):
        rank, trial, run = task
        output_dir = review_root / f"rank_{rank:02d}_trial_{trial.number:04d}" / f"run{run}"
        metrics = run_selection(
            model=args.model, dataset=args.dataset, run=run, params=trial.params, output_dir=output_dir,
            device=args.device, num_workers=args.num_workers, seed=args.seed,
            epochs=epochs, patience=patience, keep_checkpoints=args.keep_checkpoints,
        )
        return {
            "candidate_rank": rank, "trial_number": trial.number, "run": run,
            "stage1_val_mse": float(trial.value), "val_mse": float(metrics["val_metrics"]["mse"]),
            "best_epoch": int(metrics["best_epoch"]), **trial.params,
        }

    print(f"Stage 2: reviewing {args.top_k} candidates across five warm validation splits")
    review_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.review_jobs) as executor:
        futures = [executor.submit(review_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            review_rows.append(future.result())
    review_frame = pd.DataFrame(review_rows).sort_values(["candidate_rank", "run"])
    review_frame.to_csv(study_root / "candidate_validation.csv", index=False)
    summary = (
        review_frame.groupby(["candidate_rank", "trial_number"], as_index=False)
        .agg(stage1_val_mse=("stage1_val_mse", "first"),
             mean_val_mse=("val_mse", "mean"), std_val_mse=("val_mse", "std"),
             min_val_mse=("val_mse", "min"), max_val_mse=("val_mse", "max"))
        .sort_values(["mean_val_mse", "std_val_mse", "stage1_val_mse", "trial_number"])
    )
    summary.to_csv(study_root / "top_candidates_summary.csv", index=False)
    best_row = summary.iloc[0]
    best_trial = next(trial for trial in top_trials if trial.number == int(best_row.trial_number))

    config_payload = {
        "model": args.model,
        "dataset": args.dataset,
        "selection_protocol": "warm_only_two_stage",
        "protocol_version": "warm_only_tpe_top5_v1",
        "test_accessed_during_selection": False,
        "stage1": {"target_completed_trials": args.n_trials, "actual_completed_trials": completed_after,
                   "failed_trials": failed_after, "total_recorded_trials": len(study.trials),
                   "split": "warm_run1", "objective": "validation_mse", "sampler": "Optuna TPESampler",
                   "sampler_seed": args.sampler_seed,
                   "selected_trial_val_mse": float(best_trial.value)},
        "stage2": {"top_k": args.top_k, "validation_runs": [1, 2, 3, 4, 5],
                   "selection_metric": "mean_validation_mse"},
        "source_trial_number": int(best_trial.number),
        "search_space": search_space_description(args.model),
        "params": {key: float(value) if isinstance(value, float) else value for key, value in best_trial.params.items()},
        "training": {"epochs": epochs, "patience": patience, "seed": args.seed,
                     "optimizer": optimizer_name},
        "validation": {"mean_mse": float(best_row.mean_val_mse), "std_mse": float(best_row.std_val_mse)},
        "warm_data_manifest": data_manifest,
    }
    config_payload["generated_at_unix"] = time.time()
    config_payload["config_id"] = canonical_config_id(config_payload)
    atomic_write_json(study_root / "best_params.json", config_payload)
    pd.DataFrame([{**config_payload["params"], **config_payload["training"],
                   **config_payload["validation"], "config_id": config_payload["config_id"]}]).to_csv(
        study_root / "best_params.csv", index=False
    )
    complete_tmp = study_root / f".complete.tmp.{os.getpid()}"
    complete_tmp.write_text(config_payload["config_id"] + "\n", encoding="utf-8")
    complete_tmp.replace(study_root / ".complete")
    print(json.dumps(config_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
