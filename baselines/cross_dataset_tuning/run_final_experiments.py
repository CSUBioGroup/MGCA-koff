#!/usr/bin/env python3
"""Run one tuned cross-dataset model over warm/drug/protein-cold final tests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from final_common import (
    atomic_write_json,
    compare_hyperparameters,
    expected_hyperparameters,
    load_best_params,
    sha256_file,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SPLITS = ("warm", "drug_cold", "protein_cold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("moe", "bicoa"))
    parser.add_argument("--dataset", required=True, choices=("KinetX", "2773"))
    parser.add_argument("--best-params", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "best_params_final_results")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=100)
    parser.add_argument("--existing-result-action", choices=("skip", "overwrite"), default="skip")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def split_files(project_root: Path, dataset: str, split: str, run: int) -> Dict[str, Path]:
    if dataset == "KinetX":
        if split == "warm":
            root = project_root / "KinetX" / "random_split_mgca_input"
            return {role: root / ("%s_run%d.csv" % (role, run)) for role in ("train", "val", "test")}
        if split == "drug_cold":
            root = project_root / "KinetX" / "drug_cold_start_canonical_5fold" / ("fold%d" % run)
            return {role: root / (role + ".csv") for role in ("train", "val", "test")}
        root = project_root / "KinetX" / "cold_start"
        return {role: root / (role + ".csv") for role in ("train", "val", "test")}
    if split == "warm":
        root = project_root / "2773" / "new_folds" / "warm"
        return {role: root / ("%s_run%d.csv" % (role, run)) for role in ("train", "val", "test")}
    if split == "drug_cold":
        root = project_root / "2773" / "new_folds" / "drug-cold"
        return {role: root / ("%s_run%d.csv" % (role, run)) for role in ("train", "val", "test")}
    root = project_root / "2773" / "new_folds" / "target-cold"
    return {role: root / (role + ".csv") for role in ("train", "val", "test")}


def validate_warm_manifest(config: Dict, files_by_run: Dict[int, Dict[str, Path]]) -> None:
    manifest = {int(row["run"]): row for row in config.get("warm_data_manifest", [])}
    if sorted(manifest) != [1, 2, 3, 4, 5]:
        raise ValueError("Frozen configuration has an incomplete warm_data_manifest")
    for run, files in files_by_run.items():
        recorded = manifest[run]
        for role in ("train", "val"):
            expected = recorded[role + "_sha256"]
            actual = sha256_file(files[role])
            if actual != expected:
                raise ValueError(
                    "Warm %s run%d changed after tuning: expected %s, found %s (%s)"
                    % (role, run, expected, actual, files[role])
                )


def validate_completed_run(
    run_dir: Path,
    config: Dict,
    best_params_sha: str,
    model: str,
    dataset: str,
    split: str,
    run: int,
    seed: int,
    files: Dict[str, Path],
) -> None:
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "test_predictions.csv"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Completed run is missing metrics/predictions: %s" % run_dir)
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    expected_identity = (model, dataset, split, run, seed, config["config_id"])
    actual_identity = (
        metrics.get("model"),
        metrics.get("dataset"),
        metrics.get("split"),
        metrics.get("run"),
        metrics.get("seed"),
        metrics.get("tuning_config_id"),
    )
    if actual_identity != expected_identity:
        raise ValueError("Completed run identity mismatch at %s: %r != %r" % (run_dir, actual_identity, expected_identity))
    if metrics.get("selection_only") is not False or not metrics.get("test_metrics"):
        raise ValueError("Completed marker points to a non-final result: %s" % run_dir)
    if metrics.get("best_params_sha256") != best_params_sha:
        raise ValueError("Frozen best-params file changed for completed run: %s" % run_dir)
    matches, reason = compare_hyperparameters(
        metrics.get("hyperparameters", {}), expected_hyperparameters(config)
    )
    if not matches:
        raise ValueError("Completed run hyperparameter mismatch at %s: %s" % (run_dir, reason))
    for role, path in files.items():
        if metrics["input"].get(role + "_sha256") != sha256_file(path):
            raise ValueError("Completed run %s input changed at %s" % (role, run_dir))


def safe_remove_run(run_dir: Path, config_root: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = config_root.resolve()
    if resolved_root not in resolved_run.parents or not resolved_run.name.startswith("run"):
        raise ValueError("Refusing to remove unsafe run path: %s" % resolved_run)
    shutil.rmtree(str(resolved_run))


def launch_task(task: Dict) -> Dict:
    run_dir = task["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(task["command"], stdout=log, stderr=subprocess.STDOUT)
    return {"task": task, "returncode": result.returncode}


def main() -> None:
    args = parse_args()
    allowed = (args.model == "moe" and args.dataset == "KinetX") or (
        args.model == "bicoa" and args.dataset == "2773"
    )
    if not allowed:
        raise ValueError("Allowed final pairs are moe/KinetX and bicoa/2773")
    if args.max_parallel <= 0 or args.num_workers < 0:
        raise ValueError("max-parallel must be positive and num-workers non-negative")
    if args.model == "bicoa" and args.max_parallel > 1:
        raise ValueError("BiCoA-Net final runs must remain serial on one 24 GiB GPU")

    project_root = args.project_root.resolve()
    best_params = args.best_params.resolve()
    config = load_best_params(best_params, args.model, args.dataset)
    best_params_sha = sha256_file(best_params)
    all_files = {
        split: {run: split_files(project_root, args.dataset, split, run) for run in range(1, 6)}
        for split in SPLITS
    }
    for files_by_run in all_files.values():
        for files in files_by_run.values():
            for path in files.values():
                if not path.is_file():
                    raise FileNotFoundError(path)
    validate_warm_manifest(config, all_files["warm"])

    output_root = args.output_root.resolve()
    config_root = output_root / args.model / args.dataset / config["config_id"]
    print("Final benchmark preflight OK")
    print("  model/dataset: %s/%s" % (args.model, args.dataset))
    print("  config_id: %s" % config["config_id"])
    print("  runs: 3 splits x 5 = 15")
    print("  output: %s" % config_root)
    if args.preflight_only:
        return

    config_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_params), str(config_root / "selected_hyperparameters.json"))
    atomic_write_json(
        config_root / "final_run_manifest.json",
        {
            "model": args.model,
            "dataset": args.dataset,
            "tuning_config_id": config["config_id"],
            "best_params_sha256": best_params_sha,
            "splits": list(SPLITS),
            "runs_per_split": 5,
            "base_seed": args.base_seed,
            "seed_step_for_protein_cold": args.seed_step,
            "checkpoint_retained": bool(args.keep_checkpoints),
        },
    )
    trainer = HERE / (args.model + "_train_final.py")
    if not trainer.is_file():
        raise FileNotFoundError(trainer)

    for split in SPLITS:
        tasks: List[Dict] = []
        split_root = config_root / split
        for run in range(1, 6):
            files = all_files[split][run]
            seed = args.base_seed + (run - 1) * args.seed_step if split == "protein_cold" else args.base_seed
            run_dir = split_root / ("run%d" % run)
            if (run_dir / ".complete").is_file() and args.existing_result_action == "skip":
                validate_completed_run(
                    run_dir,
                    config,
                    best_params_sha,
                    args.model,
                    args.dataset,
                    split,
                    run,
                    seed,
                    files,
                )
                print("Skipping verified completed %s run%d" % (split, run))
                continue
            if run_dir.exists() and args.existing_result_action == "overwrite":
                safe_remove_run(run_dir, config_root)
            command = [
                sys.executable,
                str(trainer),
                "--model",
                args.model,
                "--dataset",
                args.dataset,
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
                str(best_params),
                "--output-dir",
                str(run_dir),
                "--device",
                args.device,
                "--num-workers",
                str(args.num_workers),
            ]
            if args.model == "bicoa" and args.cache_root:
                command.extend(["--cache-root", str(args.cache_root.resolve())])
            if not args.keep_checkpoints:
                command.append("--discard-checkpoint-after-eval")
            tasks.append({"split": split, "run": run, "run_dir": run_dir, "command": command})

        if tasks:
            print("Launching %d pending %s run(s), max_parallel=%d" % (len(tasks), split, args.max_parallel))
            failures = []
            with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
                futures = [pool.submit(launch_task, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    task = result["task"]
                    if result["returncode"] == 0:
                        print("Completed %s run%d" % (task["split"], task["run"]))
                    else:
                        failures.append(task)
                        print(
                            "FAILED %s run%d; see %s"
                            % (task["split"], task["run"], task["run_dir"] / "train.log"),
                            file=sys.stderr,
                        )
            if failures:
                raise RuntimeError("%d final run(s) failed; rerun to resume pending runs" % len(failures))

    subprocess.run(
        [sys.executable, str(HERE / "summarize_final_results.py"), "--config-root", str(config_root)],
        check=True,
    )
    print("Final benchmark complete: %s" % config_root)


if __name__ == "__main__":
    main()
