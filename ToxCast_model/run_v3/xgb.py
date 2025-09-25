from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "xgboost is required to train the XGB models. Please install it in the environment."
    ) from exc

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
    parser = argparse.ArgumentParser(description="Train XGBoost for ToxCast v3")
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


def build_model(params: dict, random_state: int) -> XGBClassifier:
    return XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss", **params)


def main() -> None:
    args = parse_args()
    fingerprint_type = args.fingerprint_type
    assay_name = resolve_assay_name(args)

    train_csv_path = Path(args.train_csv)
    val_csv_path = Path(args.val_csv) if args.val_csv else None
    test_csv_path = Path(args.test_csv) if args.test_csv else None

    def normalise_fp_dir(csv_path: Path | None, fp_arg: str | None) -> Path | None:
        if fp_arg:
            return Path(fp_arg)
        if csv_path is None:
            return None
        return csv_path.parent / "fingerprints"

    train_fp_dir = normalise_fp_dir(train_csv_path, getattr(args, "train_fp_dir", None))
    val_fp_dir = normalise_fp_dir(val_csv_path, getattr(args, "val_fp_dir", None))
    test_fp_dir = normalise_fp_dir(test_csv_path, getattr(args, "test_fp_dir", None))

    train, val, test = prepare_datasets(
        fingerprint_type,
        assay_name,
        train_csv_path,
        val_csv_path,
        test_csv_path,
        train_fp_dir,
        val_fp_dir,
        test_fp_dir,
    )

    model_save_path = Path(args.model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)

    params_dict = {
        "n_estimators": [200, 400],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    param_grid = list(ParameterGrid(params_dict))

    result = cross_validate_models(
        param_grid,
        train.features,
        train.labels,
        args.random_state,
        model_save_path,
        "xgb_intermediate",
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

    val_metrics = None
    if val and val.labels is not None:
        val_pred = final_model.predict(val.features)
        val_prob = final_model.predict_proba(val.features)[:, 1]
        val_metrics = evaluate_predictions(val.labels, val_pred, val_prob)
        logging.info("Holdout Validation F1 Score: %s", val_metrics["f1"])
        logging.info("Holdout Validation Precision: %s", val_metrics["precision"])
        logging.info("Holdout Validation Recall: %s", val_metrics["recall"])
        logging.info("Holdout Validation Accuracy: %s", val_metrics["accuracy"])
        logging.info("Holdout Validation AUC: %s", val_metrics["roc_auc"])

    test_metrics = None
    if test and test.labels is not None:
        test_pred = final_model.predict(test.features)
        test_prob = final_model.predict_proba(test.features)[:, 1]
        test_metrics = evaluate_predictions(test.labels, test_pred, test_prob)
        logging.info("Test F1 Score: %s", test_metrics["f1"])
        logging.info("Test Precision: %s", test_metrics["precision"])
        logging.info("Test Recall: %s", test_metrics["recall"])
        logging.info("Test Accuracy: %s", test_metrics["accuracy"])
        logging.info("Test AUC: %s", test_metrics["roc_auc"])

    model_filename = model_save_path / "model.joblib"
    joblib.dump(final_model, model_filename)
    logging.info("Best model saved as %s", model_filename)

    report = {
        "assay_name": assay_name,
        "model": "xgb",
        "fingerprint": fingerprint_type,
        "best_params": best_params,
        "cv_metrics": {metric: result[metric][best_key] for metric in METRIC_KEYS},
        "cv_metrics_mean": best_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "random_state": args.random_state,
        "estimator": f"{final_model.__module__}.{final_model.__class__.__name__}",
    }
    save_results(report, model_save_path / "metrics.json")


if __name__ == "__main__":
    main()
