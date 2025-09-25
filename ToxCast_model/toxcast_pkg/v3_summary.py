"""Summarise ToxCast v3 training runs across multiple seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from toxcast_pkg.v3_data import DEFAULT_NON_ASSAY_COLUMNS, load_split_dataframe

COLUMN_ORDER = [
    "AEID",
    "Database",
    "Training",
    "Positive (%)",
    "Model",
    "MF/MD",
    "Algorithm",
    "Test F1",
    "Test Precision",
    "Test Recall",
    "Test AUC",
    "Test Accuracy",
    "Validation F1",
    "Validation Precision",
    "Validation Recall",
    "Validation AUC",
    "Validation Accuracy",
]


def _find_train_csv(seed_dir: Path) -> Path:
    candidates = [
        seed_dir / "train_df.csv",
        seed_dir / "train" / "train_df.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate train_df.csv under {seed_dir}")


def _compute_training_stats(train_csv: Path) -> Dict[str, Dict[str, Optional[float]]]:
    df = load_split_dataframe(train_csv)
    assays = [col for col in df.columns if col not in DEFAULT_NON_ASSAY_COLUMNS]
    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for assay in assays:
        series = df[assay].dropna()
        count = int(series.size)
        positive_pct: Optional[float]
        if count == 0:
            positive_pct = None
        else:
            try:
                positive_pct = float(series.mean() * 100)
            except Exception:
                positive_pct = None
        stats[assay] = {"count": count if count > 0 else None, "positive_pct": positive_pct}
    return stats


def _format_ratio(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def build_seed_summary(seed_dir: Path, results_dir: Path) -> pd.DataFrame:
    train_csv = _find_train_csv(seed_dir)
    stats = _compute_training_stats(train_csv)

    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            record = json.load(f)

        assay = record.get("assay_name", "")
        stat_info = stats.get(assay, {"count": None, "positive_pct": None})

        test_metrics = record.get("test_metrics") or {}
        val_metrics = record.get("validation_metrics") or {}

        row = {
            "AEID": "",
            "Database": assay,
            "Training": stat_info.get("count"),
            "Positive (%)": _format_ratio(stat_info.get("positive_pct")),
            "Model": record.get("model"),
            "MF/MD": record.get("fingerprint"),
            "Algorithm": record.get("estimator"),
            "Test F1": test_metrics.get("f1"),
            "Test Precision": test_metrics.get("precision"),
            "Test Recall": test_metrics.get("recall"),
            "Test AUC": test_metrics.get("roc_auc"),
            "Test Accuracy": test_metrics.get("accuracy"),
            "Validation F1": val_metrics.get("f1"),
            "Validation Precision": val_metrics.get("precision"),
            "Validation Recall": val_metrics.get("recall"),
            "Validation AUC": val_metrics.get("roc_auc"),
            "Validation Accuracy": val_metrics.get("accuracy"),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=COLUMN_ORDER)
    df = df.sort_values(["Database", "Model", "MF/MD"], na_position="last").reset_index(drop=True)
    return df


def aggregate_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        summary_path = seed_dir / "summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df.insert(0, "Seed", seed_dir.name)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["Seed", *COLUMN_ORDER])
    return pd.concat(frames, ignore_index=True)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise ToxCast v3 training outputs")
    parser.add_argument("--seed-dir", type=Path, help="Directory containing seed data (train/val/test CSVs)")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with training outputs")
    parser.add_argument("--output", type=Path, help="Path to write the per-seed summary CSV")
    parser.add_argument("--aggregate", type=Path, help="Path to write the aggregated summary CSV")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.output:
        if not args.seed_dir:
            raise ValueError("--seed-dir must be provided when --output is specified")
        summary_df = build_seed_summary(args.seed_dir, args.results_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.output, index=False)

    if args.aggregate:
        aggregate_df = aggregate_summaries(args.results_dir)
        if aggregate_df.empty:
            return
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        aggregate_df.to_csv(args.aggregate, index=False)


if __name__ == "__main__":
    main()

