#!/usr/bin/env python3
"""Run the three missing 15-run training-time studies strictly serially."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = HERE.parent
SPLITS = ("warm", "drug_cold", "protein_cold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=100)
    parser.add_argument("--existing-result-action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def split_files(project_root: Path, dataset: str, split: str, run: int):
    if dataset == "KinetX":
        if split == "warm":
            root = project_root / "KinetX" / "random_split_mgca_input"
            return {
                role: root / ("%s_run%d.csv" % (role, run))
                for role in ("train", "val", "test")
            }
        if split == "drug_cold":
            root = project_root / "KinetX" / "drug_cold_start_canonical_5fold" / ("fold%d" % run)
            return {role: root / (role + ".csv") for role in ("train", "val", "test")}
        root = project_root / "KinetX" / "cold_start"
        return {role: root / (role + ".csv") for role in ("train", "val", "test")}
    if split == "warm":
        root = project_root / "2773" / "new_folds" / "warm"
        return {
            role: root / ("%s_run%d.csv" % (role, run))
            for role in ("train", "val", "test")
        }
    if split == "drug_cold":
        root = project_root / "2773" / "new_folds" / "drug-cold"
        return {
            role: root / ("%s_run%d.csv" % (role, run))
            for role in ("train", "val", "test")
        }
    root = project_root / "2773" / "new_folds" / "target-cold"
    return {role: root / (role + ".csv") for role in ("train", "val", "test")}


def make_bicoa_kinetx_default_config(path: Path, cross_root: Path) -> dict:
    sys.path.insert(0, str(cross_root))
    from common.tuning_config import canonical_config_id

    payload = {
        "model": "bicoa",
        "dataset": "KinetX",
        "selection_protocol": "published_default_no_hyperparameter_selection",
        "protocol_version": "bicoa_kinetx_published_default_v1",
        "test_accessed_during_selection": False,
        "params": {
            "lr": 0.0002,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "dropout": 0.15,
        },
        "training": {
            "epochs": 250,
            "patience": 50,
            "seed": 42,
            "optimizer": "AdamW",
        },
        "architecture": {
            "d_model": 768,
            "n_blocks": 4,
            "n_heads": 12,
            "d_ff": 3072,
            "warmup_epochs": 15,
            "mixup_alpha": 0.2,
            "ema_decay": 0.999,
        },
        "formal_run_design": {
            "warm": "five aligned pre-generated warm split runs, seed 42",
            "drug_cold": "five canonical folds, seed 42",
            "protein_cold": "fixed target-cold split, five seeds",
        },
    }
    payload["config_id"] = canonical_config_id(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def study_specs(project_root: Path, output_root: Path, cross_root: Path):
    default_config = output_root / "configs" / "bicoa_kinetx_published_default.json"
    make_bicoa_kinetx_default_config(default_config, cross_root)
    return [
        {
            "study": "bicoa_kinetx_default",
            "model": "bicoa",
            "dataset": "KinetX",
            "config": default_config,
            "warm_design": "five_aligned_split_runs",
        },
        {
            "study": "bicoa_2773_tuned",
            "model": "bicoa",
            "dataset": "2773",
            "config": cross_root / "hyperparameter_tuning" / "bicoa" / "2773" / "best_params.json",
            "warm_design": "five_aligned_split_runs",
        },
        {
            "study": "moe_kinetx_tuned",
            "model": "moe",
            "dataset": "KinetX",
            "config": cross_root / "hyperparameter_tuning" / "moe" / "KinetX" / "best_params.json",
            "warm_design": "five_aligned_split_runs",
        },
    ]


def seed_and_file_run(study: dict, split: str, run: int, base_seed: int, seed_step: int):
    if split == "protein_cold":
        return base_seed + (run - 1) * seed_step, run
    return base_seed, run


def safe_remove_run(run_dir: Path, study_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = study_root.resolve()
    if resolved_root not in resolved_run.parents or not resolved_run.name.startswith("run"):
        raise ValueError("Refusing to remove unsafe run path: %s" % resolved_run)
    shutil.rmtree(str(resolved_run))


def validate_complete(run_dir: Path, task: dict) -> None:
    metrics_path = run_dir / "metrics.json"
    process_path = run_dir / "process_timing.json"
    if not metrics_path.is_file() or not process_path.is_file():
        raise FileNotFoundError("Timed run is missing metrics or process timing: %s" % run_dir)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = (task["model"], task["dataset"], task["split"], task["run"], task["seed"])
    actual = (
        payload.get("model"),
        payload.get("dataset"),
        payload.get("split"),
        payload.get("run"),
        payload.get("seed"),
    )
    if actual != expected:
        raise ValueError("Completed timed-run identity mismatch: %r != %r" % (actual, expected))
    timing = payload.get("timing", {})
    if timing.get("protocol") != "synchronized_wall_clock_v1":
        raise ValueError("Completed run lacks the required timing protocol: %s" % run_dir)


def make_tasks(args, studies, cross_root: Path, cache_root: Path):
    tasks = []
    for study in studies:
        if not study["config"].is_file():
            raise FileNotFoundError(study["config"])
        for split in SPLITS:
            for run in range(1, 6):
                seed, file_run = seed_and_file_run(
                    study, split, run, args.base_seed, args.seed_step
                )
                files = split_files(args.project_root, study["dataset"], split, file_run)
                for path in files.values():
                    if not path.is_file():
                        raise FileNotFoundError(path)
                run_dir = args.output_root / study["study"] / split / ("run%d" % run)
                trainer = cross_root / (study["model"] + "_train_final.py")
                command = [
                    sys.executable,
                    str(trainer),
                    "--model",
                    study["model"],
                    "--dataset",
                    study["dataset"],
                    "--split",
                    split,
                    "--run",
                    str(run),
                    "--seed",
                    str(seed),
                    "--train-csv",
                    str(files["train"]),
                    "--val-csv",
                    str(files["val"]),
                    "--test-csv",
                    str(files["test"]),
                    "--best-params",
                    str(study["config"]),
                    "--output-dir",
                    str(run_dir),
                    "--device",
                    args.device,
                    "--num-workers",
                    str(args.num_workers),
                    "--discard-checkpoint-after-eval",
                ]
                if study["model"] == "bicoa":
                    command.extend(["--cache-root", str(cache_root)])
                tasks.append(
                    {
                        **study,
                        "split": split,
                        "run": run,
                        "file_run": file_run,
                        "seed": seed,
                        "files": files,
                        "run_dir": run_dir,
                        "command": command,
                    }
                )
    return tasks


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    cross_root = args.project_root / "cross_dataset_bayesian_tuning"
    cache_root = (
        args.cache_root
        or Path(os.environ.get("BICOA_FEATURE_CACHE", args.project_root / "bicoa_cross_tuning_cache"))
    ).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    studies = study_specs(args.project_root, args.output_root, cross_root)
    tasks = make_tasks(args, studies, cross_root, cache_root)
    manifest = {
        "protocol": "missing_formal_training_times_serial_v1",
        "concurrency": 1,
        "total_runs": len(tasks),
        "studies": [
            {
                "study": row["study"],
                "model": row["model"],
                "dataset": row["dataset"],
                "configuration": str(row["config"]),
                "warm_design": row["warm_design"],
                "runs": 15,
            }
            for row in studies
        ],
        "timing_primary_endpoint": "metrics.timing.training_duration_sec",
        "feature_precomputation_included": False,
        "task_order": [
            {
                "ordinal": index,
                "study": task["study"],
                "split": task["split"],
                "run": task["run"],
                "seed": task["seed"],
                "file_run": task["file_run"],
                "run_dir": str(task["run_dir"]),
            }
            for index, task in enumerate(tasks, start=1)
        ],
    }
    (args.output_root / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Training-time preflight OK")
    print("  studies: 3")
    print("  runs: 3 x 15 = %d" % len(tasks))
    print("  concurrency: 1 (hard-coded serial scheduler)")
    print("  output: %s" % args.output_root)
    if args.preflight_only:
        return

    failures = []
    for ordinal, task in enumerate(tasks, start=1):
        run_dir = task["run_dir"]
        study_root = args.output_root / task["study"]
        if (run_dir / ".complete").is_file() and args.existing_result_action == "skip":
            validate_complete(run_dir, task)
            print("[%d/%d] skip verified %s/%s/run%d" % (
                ordinal, len(tasks), task["study"], task["split"], task["run"]
            ))
            continue
        if run_dir.exists():
            if args.existing_result_action == "overwrite":
                safe_remove_run(run_dir, study_root)
            else:
                raise RuntimeError(
                    "Incomplete output exists at %s. Set EXISTING_RESULT_ACTION=overwrite "
                    "to rerun it after inspection." % run_dir
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        print("[%d/%d] start %s/%s/run%d seed=%d" % (
            ordinal,
            len(tasks),
            task["study"],
            task["split"],
            task["run"],
            task["seed"],
        ))
        started = time.perf_counter()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                task["command"], stdout=log, stderr=subprocess.STDOUT, env=environment
            )
        process_duration = time.perf_counter() - started
        process_payload = {
            "study": task["study"],
            "split": task["split"],
            "run": task["run"],
            "seed": task["seed"],
            "returncode": result.returncode,
            "subprocess_wall_clock_duration_sec": process_duration,
            "includes": [
                "python_startup",
                "data_and_cache_loading",
                "model_initialization",
                "training_and_validation",
                "final_evaluation",
                "artifact_writes",
            ],
            "feature_precomputation_included": False,
        }
        (run_dir / "process_timing.json").write_text(
            json.dumps(process_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.returncode:
            failures.append(task)
            print("FAILED; inspect %s" % log_path, file=sys.stderr)
            break
        validate_complete(run_dir, task)
        print("[%d/%d] complete in %.3f s" % (ordinal, len(tasks), process_duration))
    if failures:
        raise RuntimeError("A timed run failed. Fix it and rerun; completed runs will be resumed.")
    print("All 45 timed training runs completed serially: %s" % args.output_root)


if __name__ == "__main__":
    main()
