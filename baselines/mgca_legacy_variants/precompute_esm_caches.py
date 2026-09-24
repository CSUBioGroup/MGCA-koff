#!/usr/bin/env python3
"""Build production ESM2 caches for tuned MGCA experiments.

Existing warm-tuning caches are first converted into a sequence-level feature
bank. Production caches are assembled from that bank whenever possible. A
shared FP32 ESM2 model is loaded only for genuinely unseen protein sequences.
Labels are never consumed for fitting during this cache-only preflight.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get("BIO_PROJECT_ROOT", ONLINE_ROOT)).resolve()
MGCA_SCRIPT = SCRIPT_DIR / "ESM_Morgan_Hybrid_Fusion.py"
CONFIG_ROOT = ONLINE_ROOT / "stage_data" / "mgca_morgan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("KinetX", "2773"), required=True)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("warm", "drug_cold", "protein_cold"),
        required=True,
    )
    parser.add_argument("--esm2-path", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=CONFIG_ROOT)
    parser.add_argument(
        "--window-size", type=int, default=None,
        help=(
            "Explicit ESM2 window size for all selected datasets. When set, "
            "the cache preflight does not inspect a tuner-specific config schema."
        ),
    )
    parser.add_argument(
        "--window-layout",
        choices=("legacy_anchors_v1", "even_span_v2"),
        default=None,
        help="Explicit ESM2 window layout; otherwise read it from best_params.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def load_mgca_module():
    spec = importlib.util.spec_from_file_location("mgca_best_params_cache", MGCA_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import MGCA implementation: {MGCA_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_paths(dataset: str, protocol: str, run: int) -> tuple[Path, Path, Path]:
    if dataset == "KinetX":
        if protocol == "warm":
            root = PROJECT_ROOT / "KinetX" / "random_split_mgca_input"
            names = (f"train_run{run}.csv", f"val_run{run}.csv", f"test_run{run}.csv")
        elif protocol == "drug_cold":
            root = PROJECT_ROOT / "KinetX" / "drug_cold_start_canonical_5fold" / f"fold{run}"
            names = ("train.csv", "val.csv", "test.csv")
        else:
            root = PROJECT_ROOT / "KinetX" / "cold_start"
            names = ("train.csv", "val.csv", "test.csv")
    else:
        if protocol == "warm":
            root = PROJECT_ROOT / "2773" / "new_folds" / "warm"
            names = (f"train_run{run}.csv", f"val_run{run}.csv", f"test_run{run}.csv")
        elif protocol == "drug_cold":
            root = PROJECT_ROOT / "2773" / "new_folds" / "drug-cold"
            names = (f"train_run{run}.csv", f"val_run{run}.csv", f"test_run{run}.csv")
        else:
            root = PROJECT_ROOT / "2773" / "new_folds" / "target-cold"
            names = ("train.csv", "val.csv", "test.csv")
    paths = tuple(root / name for name in names)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths  # type: ignore[return-value]


def window_config(config_root: Path, dataset: str) -> tuple[int, str]:
    path = config_root / dataset / "best_params.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    # The warm-only tuner records ``fingerprint=morgan`` explicitly.  The
    # cross-protocol tuner identifies the same implementation as
    # ``model=mgca`` and omits the redundant fingerprint field.
    is_morgan = (
        config.get("fingerprint") == "morgan"
        or (config.get("model") == "mgca" and "fingerprint" not in config)
    )
    if config.get("dataset") != dataset or not is_morgan:
        raise ValueError(f"Unexpected tuned configuration identity: {path}")
    layout = config.get("esm2_window_layout", "legacy_anchors_v1")
    return int(config["params"]["window_size"]), str(layout)


def load_feature_bank(
    module, torch, dataset: str, ws: int, window_layout: str
) -> dict[str, object]:
    """Load FASTA->feature entries from the five warm train+val tuning caches."""
    bank: dict[str, object] = {}
    loaded_caches = 0
    for run in range(1, 6):
        train_csv, val_csv, _ = split_paths(dataset, "warm", run)
        rows = module.read_labeled_rows(str(train_csv))
        rows.extend(module.read_labeled_rows(str(val_csv)))
        cache_path = Path(module.get_combined_esm_cache_path(
            [str(train_csv), str(val_csv)],
            window_size=ws,
            window_layout=window_layout,
        ))
        if not cache_path.is_file():
            print(f"Warm tuning cache unavailable for feature-bank reuse: {cache_path}")
            continue
        try:
            cached = torch.load(str(cache_path), map_location="cpu")
        except Exception as exc:
            print(f"Cannot reuse warm tuning cache {cache_path}: {exc}")
            continue
        valid = (
            getattr(cached, "ndim", 0) == 3
            and int(cached.shape[0]) == len(rows)
            and int(cached.shape[1]) == 4
        )
        if not valid:
            print(
                f"Warm tuning cache shape mismatch, skipping: {cache_path}; "
                f"shape={getattr(cached, 'shape', None)}, rows={len(rows)}"
            )
            del cached
            continue
        for row, feature in zip(rows, cached):
            if row[0] not in bank:
                bank[row[0]] = feature.detach().clone()
        loaded_caches += 1
        del cached
    print(
        f"{dataset} ws={ws} layout={window_layout} feature bank: "
        f"{len(bank)} unique FASTA sequences "
        f"from {loaded_caches}/5 warm tuning caches"
    )
    return bank


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.window_size is not None and args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if not args.esm2_path.exists():
        raise FileNotFoundError(args.esm2_path)

    import torch

    module = load_mgca_module()
    device = torch.device(args.device)
    configs = {}
    for dataset in args.datasets:
        if args.window_size is not None:
            configs[dataset] = (
                int(args.window_size),
                args.window_layout or "legacy_anchors_v1",
            )
        else:
            ws, layout = window_config(args.config_root.resolve(), dataset)
            configs[dataset] = (ws, args.window_layout or layout)
    feature_banks = {
        dataset: load_feature_bank(
            module, torch, dataset, configs[dataset][0], configs[dataset][1]
        )
        for dataset in args.datasets
    }
    jobs: list[tuple[str, str, int, tuple[Path, Path, Path], int, str]] = []
    seen: set[tuple[tuple[str, ...], int, str]] = set()
    for dataset in args.datasets:
        ws, window_layout = configs[dataset]
        for protocol in args.protocols:
            # Protein-cold repeats one fixed split with five seeds, so its ESM
            # cache is generated once. Warm/drug-cold have five distinct folds.
            runs = (1,) if protocol == "protein_cold" else range(1, 6)
            for run in runs:
                paths = split_paths(dataset, protocol, run)
                identity = (
                    tuple(str(path.resolve()) for path in paths), ws, window_layout
                )
                if identity not in seen:
                    seen.add(identity)
                    jobs.append((dataset, protocol, run, paths, ws, window_layout))

    tokenizer = None
    esm_model = None
    built = 0
    reused = 0
    try:
        for index, (dataset, protocol, run, paths, ws, window_layout) in enumerate(jobs, 1):
            rows = []
            for path in paths:
                rows.extend(module.read_labeled_rows(str(path)))
            cache_path = Path(module.get_combined_esm_cache_path(
                [str(path) for path in paths],
                window_size=ws,
                window_layout=window_layout,
            ))

            valid = False
            cached = None
            if cache_path.is_file():
                try:
                    cached = torch.load(str(cache_path), map_location="cpu")
                    valid = (
                        getattr(cached, "ndim", 0) == 3
                        and int(cached.shape[0]) == len(rows)
                        and int(cached.shape[1]) == 4
                    )
                except Exception as exc:
                    print(f"[{index}/{len(jobs)}] Invalid cache {cache_path}: {exc}")

            label = f"{dataset}/{protocol}/run{run}/ws{ws}/{window_layout}"
            if valid:
                # A previous production cache can extend the sequence bank for
                # subsequent protocols without any ESM2 inference.
                bank = feature_banks[dataset]
                for row, feature in zip(rows, cached):
                    if row[0] not in bank:
                        bank[row[0]] = feature.detach().clone()
                del cached
                reused += 1
                print(f"[{index}/{len(jobs)}] Reusing {label}: {cache_path}")
                continue
            if cached is not None:
                del cached

            bank = feature_banks[dataset]
            missing_sequences = list(dict.fromkeys(
                row[0] for row in rows if row[0] not in bank
            ))
            if missing_sequences:
                print(
                    f"[{index}/{len(jobs)}] {label}: {len(missing_sequences)} unique "
                    "FASTA sequence(s) are absent from reusable caches"
                )
                if esm_model is None:
                    print("Loading one shared FP32 ESM2 model for only the missing sequences")
                    tokenizer = module.AutoTokenizer.from_pretrained(str(args.esm2_path))
                    if importlib.util.find_spec("accelerate") is not None:
                        try:
                            esm_model = module.AutoModelForMaskedLM.from_pretrained(
                                str(args.esm2_path), low_cpu_mem_usage=True
                            ).to(device)
                        except TypeError:
                            esm_model = module.AutoModelForMaskedLM.from_pretrained(
                                str(args.esm2_path)
                            ).to(device)
                    else:
                        # Avoid a failed low_cpu_mem_usage attempt followed by a
                        # second checkpoint load when Accelerate is unavailable.
                        print("Accelerate not installed; loading the local checkpoint once normally")
                        esm_model = module.AutoModelForMaskedLM.from_pretrained(
                            str(args.esm2_path)
                        ).to(device)

                missing_features = module.batch_extract_esm2(
                    missing_sequences,
                    tokenizer,
                    esm_model,
                    device,
                    batch_size=args.batch_size,
                    window_size=ws,
                    window_layout=window_layout,
                ).detach().cpu()
                for sequence, feature in zip(missing_sequences, missing_features):
                    bank[sequence] = feature.clone()
                del missing_features
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"[{index}/{len(jobs)}] {label}: fully covered by reusable ESM features")

            print(f"[{index}/{len(jobs)}] Assembling {label}: {cache_path}")
            features = torch.stack([bank[row[0]] for row in rows], dim=0)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(f"{cache_path}.tmp.{os.getpid()}")
            try:
                torch.save(features, str(temporary))
                os.replace(str(temporary), str(cache_path))
            finally:
                if temporary.exists():
                    temporary.unlink()
            del features
            built += 1
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

    print(f"Production ESM cache preflight complete: {reused} reused, {built} built")


if __name__ == "__main__":
    main()
