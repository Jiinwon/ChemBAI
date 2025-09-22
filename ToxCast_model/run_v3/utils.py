from __future__ import annotations

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
    train_dir: Path,
    val_dir: Path | None,
    test_dir: Path | None,
) -> Tuple[SplitData, SplitData | None, SplitData | None]:
    train = load_split_data(train_dir, fingerprint_type, assay_name)
    val = load_split_data(val_dir, fingerprint_type, assay_name) if val_dir else None
    test = load_split_data(test_dir, fingerprint_type, assay_name) if test_dir else None

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
