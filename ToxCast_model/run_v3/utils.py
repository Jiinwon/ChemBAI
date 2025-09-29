from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from toxcast_pkg.v3_data import (
    SplitData,
    ensure_matching_indices,
    load_split_data,
)


METRIC_KEYS = ("precision", "recall", "f1", "accuracy", "roc_auc")


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common GPU control arguments to ``parser``.

    All run_v3 entrypoints share the same desire for a ``--use-gpu`` flag,
    while still allowing users to opt out explicitly.  The helper keeps the
    behaviour consistent across scripts.
    """

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Prefer CUDA devices when available (default: CPU).",
    )
    group.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Force CPU execution even when CUDA devices are available.",
    )
    parser.set_defaults(use_gpu=False)


def configure_device(use_gpu: bool, logger: logging.Logger | None = None) -> str:
    """Initialise the preferred compute device for deep-learning frameworks.

    The helper attempts to configure PyTorch first and falls back to
    TensorFlow.  Any errors are converted into informative log messages so the
    caller can keep running on CPU.  The selected device identifier is
    returned to help with debugging/logging.
    """

    log = logger or logging.getLogger(__name__)
    if use_gpu:
        try:  # PyTorch configuration (optional dependency)
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
                log.info("Using PyTorch device: %s", device)
                return str(device)
            log.warning("GPU requested but PyTorch reports no CUDA devices. Falling back to CPU.")
        except Exception as exc:  # pragma: no cover - optional dependency guard
            log.warning("GPU requested but PyTorch initialisation failed: %s", exc)

        try:  # TensorFlow configuration (optional dependency)
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    tf.config.set_visible_devices(gpus[0], "GPU")
                except Exception as exc:  # pragma: no cover - TF runtime guard
                    log.warning("Unable to limit TensorFlow to a single GPU: %s", exc)
                try:  # pragma: no cover - best effort memory growth setting
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                except Exception:
                    pass
                log.info("Using TensorFlow GPU device: %s", gpus[0].name)
                return gpus[0].name
            log.warning("GPU requested but TensorFlow did not detect any GPUs. Falling back to CPU.")
        except Exception as exc:  # pragma: no cover - optional dependency guard
            log.warning("TensorFlow GPU initialisation failed: %s", exc)

    log.info("Using CPU execution.")
    return "cpu"


def save_results(result: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def find_best_model(results: dict, metric: str = "f1", metric_agg: str = "mean") -> Tuple[str, dict, float]:
    """Return the best model configuration from a cross validation run."""

    best_model = None
    best_score = -np.inf
    best_model_key = None

    for model_key in results["model"].keys():
        scores = results[metric][model_key]
        if not scores:
            continue
        if metric_agg == "mean":
            agg_score = float(np.mean(scores))
        elif metric_agg == "median":
            agg_score = float(np.median(scores))
        else:
            raise ValueError("metric_agg must be either 'mean' or 'median'")

        if agg_score > best_score:
            best_score = agg_score
            best_model = results["model"][model_key]
            best_model_key = model_key

    if best_model_key is None:
        raise RuntimeError("No valid model configuration identified during CV")

    return best_model_key, best_model, best_score


def apply_smote(x: pd.DataFrame, y: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    positive_count = int(np.sum(y == 1))
    if positive_count > 1:
        k_neighbors = min(5, positive_count - 1)
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        try:
            return sm.fit_resample(x, y)
        except ValueError as exc:
            logging.warning("SMOTE failed: %s", exc)
            return x, y
    logging.warning("Not enough positive samples for SMOTE; skipping oversampling.")
    return x, y


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    try:
        if y_prob is None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def prepare_datasets(
    fingerprint_type: str,
    assay_name: str,
    train_csv: Path,
    val_csv: Path | None,
    test_csv: Path | None,
    train_fp_dir: Path | None = None,
    val_fp_dir: Path | None = None,
    test_fp_dir: Path | None = None,
) -> Tuple[SplitData, SplitData | None, SplitData | None]:
    train = load_split_data(train_csv, fingerprint_type, assay_name, train_fp_dir)
    val = (
        load_split_data(val_csv, fingerprint_type, assay_name, val_fp_dir)
        if val_csv is not None
        else None
    )
    test = (
        load_split_data(test_csv, fingerprint_type, assay_name, test_fp_dir)
        if test_csv is not None
        else None
    )

    ensure_matching_indices(*(d for d in (train, val, test) if d is not None))
    return train, val, test


def initialise_result_container(param_grid: Iterable[dict]) -> dict:
    result = {"model": {}, "precision": {}, "recall": {}, "f1": {}, "accuracy": {}, "roc_auc": {}}
    for idx, params in enumerate(param_grid):
        key = f"model{idx}"
        result["model"][key] = params
        for metric in METRIC_KEYS:
            result[metric][key] = []
    return result


def cross_validate_models(
    param_grid: Iterable[dict],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    random_state: int,
    save_dir: Path,
    result_prefix: str,
    fingerprint_type: str,
    model_builder: Callable[[dict], object],
) -> dict:
    result = initialise_result_container(param_grid)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for idx, params in enumerate(tqdm(list(param_grid), desc="params", leave=False, disable=True)):
        model_key = f"model{idx}"
        for train_idx, val_idx in kf.split(train_x, train_y):
            fold_train_x, fold_val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
            fold_train_y, fold_val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

            fold_train_x, fold_train_y = apply_smote(fold_train_x, fold_train_y, random_state)

            model = model_builder(params)
            model.fit(fold_train_x, fold_train_y)
            val_pred = model.predict(fold_val_x)
            val_prob = model.predict_proba(fold_val_x)[:, 1]

            metrics = evaluate_predictions(fold_val_y, val_pred, val_prob)
            for metric in METRIC_KEYS:
                result[metric][model_key].append(metrics[metric])

        save_results(result, save_dir / f"{result_prefix}_{fingerprint_type}.json")

    return result
