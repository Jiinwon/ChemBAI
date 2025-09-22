from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from toxcast_pkg.common import ParameterGrid
from toxcast_pkg.v3_data import get_assay_names_from_csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_v3.utils import (
    METRIC_KEYS,
    apply_smote,
    cross_validate_models,
    evaluate_predictions,
    find_best_model,
    prepare_datasets,
)

warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest for ToxCast v3")
    parser.add_argument("--fingerprint_type", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--train_fp_dir", required=True)
    parser.add_argument("--val_fp_dir", required=True)
    parser.add_argument("--test_fp_dir", required=True)
    parser.add_argument("--assay_name")
    parser.add_argument("--assay_index", type=int, default=None)
    parser.add_argument("--model_save_path", required=True)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def resolve_assay_name(args: argparse.Namespace) -> str:
    if args.assay_name:
        return args.assay_name
    assay_names = get_assay_names_from_csv(args.train_csv)
    if args.assay_index is None or args.assay_index >= len(assay_names):
        raise ValueError("Assay name not provided and index out of range")
    return assay_names[args.assay_index]


def build_model(params: dict, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_state, n_jobs=-1, **params)


def main() -> None:
    args = parse_args()
    fingerprint_type = args.fingerprint_type
    assay_name = resolve_assay_name(args)

    train_dir = Path(args.train_fp_dir).parent
    val_dir = Path(args.val_fp_dir).parent
    test_dir = Path(args.test_fp_dir).parent

    train, val, test = prepare_datasets(
        fingerprint_type,
        assay_name,
        train_dir,
        val_dir,
        test_dir,
    )

    model_save_path = Path(args.model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)

    params_dict = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
    }
    param_grid = list(ParameterGrid(params_dict))

    result = cross_validate_models(
        param_grid,
        train.features,
        train.labels,
        args.random_state,
        model_save_path,
        "rf_intermediate",
        fingerprint_type,
        lambda params: build_model(params, args.random_state),
    )

    best_key, best_params, best_f1 = find_best_model(result, metric="f1")
    best_metrics = {metric: float(np.mean(result[metric][best_key])) for metric in METRIC_KEYS}

    logging.info("Best Model Parameters: %s", best_params)
    logging.info("Validation F1 Score: %s", best_f1)
    logging.info("Validation Precision: %s", best_metrics["precision"])
    logging.info("Validation Recall: %s", best_metrics["recall"])
    logging.info("Validation Accuracy: %s", best_metrics["accuracy"])
    logging.info("Validation AUC: %s", best_metrics["roc_auc"])

    final_model = build_model(best_params, args.random_state)
    train_x, train_y = apply_smote(train.features, train.labels, args.random_state)
    final_model.fit(train_x, train_y)

    if val and val.labels is not None:
        val_pred = final_model.predict(val.features)
        val_prob = final_model.predict_proba(val.features)[:, 1]
        val_metrics = evaluate_predictions(val.labels, val_pred, val_prob)
        logging.info("Holdout Validation F1 Score: %s", val_metrics["f1"])
        logging.info("Holdout Validation Precision: %s", val_metrics["precision"])
        logging.info("Holdout Validation Recall: %s", val_metrics["recall"])
        logging.info("Holdout Validation Accuracy: %s", val_metrics["accuracy"])
        logging.info("Holdout Validation AUC: %s", val_metrics["roc_auc"])

    if test and test.labels is not None:
        test_pred = final_model.predict(test.features)
        test_prob = final_model.predict_proba(test.features)[:, 1]
        test_metrics = evaluate_predictions(test.labels, test_pred, test_prob)
        logging.info("Test F1 Score: %s", test_metrics["f1"])
        logging.info("Test Precision: %s", test_metrics["precision"])
        logging.info("Test Recall: %s", test_metrics["recall"])
        logging.info("Test Accuracy: %s", test_metrics["accuracy"])
        logging.info("Test AUC: %s", test_metrics["roc_auc"])

    model_filename = model_save_path / f"{assay_name}_best_model_{fingerprint_type}_rf.joblib"
    joblib.dump(final_model, model_filename)
    logging.info("Best model saved as %s", model_filename)


if __name__ == "__main__":
    main()
