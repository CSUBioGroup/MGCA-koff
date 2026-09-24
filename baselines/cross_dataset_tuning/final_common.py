#!/usr/bin/env python3
"""Shared integrity, metric, and atomic-I/O helpers for final benchmark runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from common.tuning_config import canonical_config_id


METRIC_NAMES = ("mse", "rmse", "mae", "r2", "pearson", "spearman", "ci")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_complete(output_dir: Path, message: str = "final benchmark complete\n") -> None:
    output_dir = Path(output_dir)
    temporary = output_dir / (".complete.tmp.%d" % os.getpid())
    temporary.write_text(message, encoding="utf-8")
    temporary.replace(output_dir / ".complete")


def load_best_params(path: Path, expected_model: str, expected_dataset: str) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("model") != expected_model or payload.get("dataset") != expected_dataset:
        raise ValueError(
            "Frozen configuration identity mismatch: expected %s/%s, found %s/%s"
            % (expected_model, expected_dataset, payload.get("model"), payload.get("dataset"))
        )
    required_params = {"lr", "weight_decay", "batch_size", "dropout"}
    missing = required_params - set(payload.get("params", {}))
    if missing:
        raise ValueError("Frozen configuration is missing parameters: %s" % sorted(missing))
    recorded_id = payload.get("config_id")
    calculated_id = canonical_config_id(payload)
    if recorded_id != calculated_id:
        raise ValueError(
            "Frozen configuration hash mismatch: recorded=%r calculated=%r"
            % (recorded_id, calculated_id)
        )
    if payload.get("test_accessed_during_selection") is not False:
        raise ValueError("Frozen configuration does not certify test isolation")
    return payload


def concordance_index(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Harrell-style CI with 0.5 credit for tied predictions."""
    true = np.asarray(list(y_true), dtype=float).reshape(-1)
    pred = np.asarray(list(y_pred), dtype=float).reshape(-1)
    concordant = 0
    discordant = 0
    tied = 0
    for index in range(len(true) - 1):
        true_delta = true[index + 1 :] - true[index]
        pred_delta = pred[index + 1 :] - pred[index]
        comparable = true_delta != 0
        if not np.any(comparable):
            continue
        true_delta = true_delta[comparable]
        pred_delta = pred_delta[comparable]
        tied += int(np.sum(pred_delta == 0))
        product = true_delta * pred_delta
        concordant += int(np.sum(product > 0))
        discordant += int(np.sum(product < 0))
    denominator = concordant + discordant + tied
    return float((concordant + 0.5 * tied) / denominator) if denominator else 0.5


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * ((start + 1) + stop)
        start = stop
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    true = np.asarray(list(y_true), dtype=float).reshape(-1)
    pred = np.asarray(list(y_pred), dtype=float).reshape(-1)
    if len(true) != len(pred) or len(true) == 0:
        raise ValueError("Metric inputs must have the same non-zero length")
    error = pred - true
    mse = float(np.mean(error ** 2))
    variance = float(np.var(true))
    pearson = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 and np.std(pred) > 0 else 0.0
    true_rank, pred_rank = average_ranks(true), average_ranks(pred)
    spearman = (
        float(np.corrcoef(true_rank, pred_rank)[0, 1])
        if len(true) > 1 and np.std(pred_rank) > 0
        else 0.0
    )
    values = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - mse / variance) if variance > 0 else float("nan"),
        "pearson": pearson,
        "spearman": spearman,
        "ci": concordance_index(true, pred),
    }
    return values


def expected_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(config["params"])
    params.update(config.get("training", {}))
    return params


def compare_hyperparameters(actual: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
    for key, value in expected.items():
        if key not in actual:
            return False, "missing hyperparameter %s" % key
        current = actual[key]
        if isinstance(value, float):
            if not math.isclose(float(current), value, rel_tol=1e-12, abs_tol=1e-15):
                return False, "%s differs: %r != %r" % (key, current, value)
        elif current != value:
            return False, "%s differs: %r != %r" % (key, current, value)
    return True, ""
