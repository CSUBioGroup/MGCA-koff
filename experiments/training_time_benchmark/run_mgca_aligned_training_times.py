#!/usr/bin/env python3
"""Run the 15 controlled MGCA-Morgan/KinetX timing jobs strictly serially."""

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
    parser.add_argument("--best-params", type=Path, required=True)
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=100)
    parser.add_argument(
        "--existing-result-action", choices=("skip", "overwrite"), default="skip"
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def split_files(project_root: Path, split: str, run: int):
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


def seed_for(split: str, run: int, base_seed: int, seed_step: int) -> int:
    return base_seed + (run - 1) * seed_step if split == "protein_cold" else base_seed


def safe_remove_run(run_dir: Path, study_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = study_root.resolve()
    if resolved_root not in resolved_run.parents or not resolved_run.name.startswith("run"):
        raise ValueError("Refusing to remove unsafe run path: %s" % resolved_run)
    shutil.rmtree(str(resolved_run))


def validate_complete(run_dir: Path, task: dict) -> None:
    for required in (".complete", "metrics.json", "process_timing.json"):
        if not (run_dir / required).is_file():
            raise FileNotFoundError("Incomplete MGCA timed run: %s" % run_dir)
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    actual = (
        payload.get("model"),
        payload.get("dataset"),
        payload.get("split"),
        payload.get("run"),
        payload.get("seed"),
    )
    expected = (
        "mgca_morgan",
        "KinetX",
        task["split"],
        task["run"],
        task["seed"],
    )
    if actual != expected:
        raise ValueError("MGCA timed-run identity mismatch: %r != %r" % (actual, expected))
    timing = payload.get("timing", {})
    if timing.get("protocol") != "synchronized_wall_clock_v1":
        raise ValueError("MGCA run lacks synchronized timing: %s" % run_dir)
    if timing.get("feature_precomputation_included") is not False:
        raise ValueError("MGCA training time unexpectedly includes feature precomputation")


def make_tasks(args: argparse.Namespace):
    trainer = args.project_root / "training_time_benchmark" / "mgca_train_aligned_timed.py"
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    tasks = []
    study_root = args.output_root / "mgca_kinetx_tuned"
    for split in SPLITS:
        for run in range(1, 6):
            files = split_files(args.project_root, split, run)
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
            seed = seed_for(split, run, args.base_seed, args.seed_step)
            run_dir = study_root / split / ("run%d" % run)
            command = [
                sys.executable,
                str(trainer),
                "--dataset",
                "KinetX",
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
                str(args.best_params),
                "--esm2-path",
                str(args.esm2_path),
                "--output-dir",
                str(run_dir),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
                "--discard-checkpoint-after-eval",
            ]
            tasks.append(
                {
                    "split": split,
                    "run": run,
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
    args.best_params = args.best_params.resolve()
    args.esm2_path = args.esm2_path.resolve()
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    for path in (args.best_params, args.esm2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(args)
    manifest = {
        "protocol": "mgca_kinetx_aligned_serial_synchronized_v1",
        "concurrency": 1,
        "total_runs": len(tasks),
        "study": "mgca_kinetx_tuned",
        "configuration": str(args.best_params),
        "timing_primary_endpoint": "metrics.timing.training_duration_sec",
        "feature_precomputation_included": False,
        "design": {
            "warm": "five aligned pre-generated split runs, seed 42",
            "drug_cold": "five canonical folds, seed 42",
            "protein_cold": "one fixed split, seeds 42/142/242/342/442",
        },
        "task_order": [
            {
                "ordinal": index,
                "split": task["split"],
                "run": task["run"],
                "seed": task["seed"],
                "run_dir": str(task["run_dir"]),
            }
            for index, task in enumerate(tasks, start=1)
        ],
    }
    (args.output_root / "mgca_aligned_benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("MGCA aligned timing preflight OK")
    print("  runs: 3 protocols x 5 = %d" % len(tasks))
    print("  concurrency: 1 (hard-coded serial scheduler)")
    print("  config: %s" % args.best_params)
    print("  output: %s" % args.output_root)
    if args.preflight_only:
        return

    study_root = args.output_root / "mgca_kinetx_tuned"
    for ordinal, task in enumerate(tasks, start=1):
        run_dir = task["run_dir"]
        if (run_dir / ".complete").is_file() and args.existing_result_action == "skip":
            validate_complete(run_dir, task)
            print(
                "[%d/%d] skip verified mgca_kinetx_tuned/%s/run%d"
                % (ordinal, len(tasks), task["split"], task["run"])
            )
            continue
        if run_dir.exists():
            if args.existing_result_action == "overwrite":
                safe_remove_run(run_dir, study_root)
            else:
                raise RuntimeError(
                    "Incomplete output exists at %s. Inspect it, then set "
                    "EXISTING_RESULT_ACTION=overwrite to rerun that target." % run_dir
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        print(
            "[%d/%d] start mgca_kinetx_tuned/%s/run%d seed=%d"
            % (ordinal, len(tasks), task["split"], task["run"], task["seed"])
        )
        started = time.perf_counter()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                task["command"], stdout=log, stderr=subprocess.STDOUT, env=environment
            )
        process_duration = time.perf_counter() - started
        process_payload = {
            "study": "mgca_kinetx_tuned",
            "split": task["split"],
            "run": task["run"],
            "seed": task["seed"],
            "returncode": result.returncode,
            "subprocess_wall_clock_duration_sec": process_duration,
            "includes": [
                "python_startup_and_imports",
                "data_and_cache_loading",
                "model_initialization",
                "training_and_validation",
                "final_train_validation_test_evaluation",
                "artifact_writes",
            ],
            "feature_precomputation_included": False,
        }
        (run_dir / "process_timing.json").write_text(
            json.dumps(process_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.returncode:
            print("FAILED; inspect %s" % log_path, file=sys.stderr)
            raise RuntimeError("MGCA timed run failed; completed runs can be resumed")
        validate_complete(run_dir, task)
        print("[%d/%d] complete in %.3f s" % (ordinal, len(tasks), process_duration))
    print("All 15 MGCA aligned timing runs completed serially: %s" % args.output_root)


if __name__ == "__main__":
    main()
