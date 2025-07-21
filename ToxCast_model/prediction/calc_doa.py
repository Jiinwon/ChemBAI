#!/usr/bin/env python
"""Calculate DoA values for prediction results and update the Excel file."""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Predict_data import _tanimoto_max

try:
    from config import RESULTS_DIR, PREDICT_FP_PATH
except ImportError as exc:
    raise ImportError("Run from within ToxCast_model directory") from exc

TRAIN_FP_BASE = Path("data/ToxCast_v.4.1_v.2/fingerprints")


def _latest_result() -> Path:
    dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No prediction results found under RESULTS_DIR")
    latest = max(dirs, key=lambda p: p.stat().st_mtime)
    files = list(latest.glob("*_prediction.xlsx"))
    if len(files) != 1:
        raise FileNotFoundError(f"Could not locate prediction excel in {latest}")
    return files[0]


def _load_dropidx(mf: str):
    path = PREDICT_FP_PATH / f"{mf}_dropidx.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path).iloc[:, 0].tolist()
        except pd.errors.EmptyDataError:
            return []
    return []


def _calc_doa_for_mf(mf: str, train_cache: dict):
    if mf not in train_cache:
        fp_path = TRAIN_FP_BASE / f"{mf}.csv"
        drop_path = TRAIN_FP_BASE / f"{mf}_dropidx.csv"
        train_df = pd.read_csv(fp_path)
        if drop_path.exists() and drop_path.stat().st_size > 0:
            try:
                drop_idx = pd.read_csv(drop_path).iloc[:, 0].tolist()
            except pd.errors.EmptyDataError:
                drop_idx = []
            if drop_idx:
                train_df = train_df.drop(index=drop_idx).reset_index(drop=True)
        train_df, _ = train_test_split(train_df, test_size=0.2, shuffle=True, random_state=42)
        train_cache[mf] = train_df.astype(bool).values
    train_fps = train_cache[mf]

    pred_fp = pd.read_csv(PREDICT_FP_PATH / f"{mf}.csv")
    drop_idx = _load_dropidx(mf)
    if drop_idx:
        pred_fp = pred_fp.drop(index=drop_idx).reset_index(drop=True)

    return [
        _tanimoto_max(fp.astype(bool), train_fps)
        for fp in pred_fp.to_numpy()
    ]


def main(result_file: Path, metadata_file: Path) -> None:
    df = pd.read_excel(result_file)
    meta = json.loads(metadata_file.read_text())
    assay_to_mf = {m["ASSAY"]: m["MF"] for m in meta}

    train_cache = {}
    doa_cache = {}

    for assay, mf in assay_to_mf.items():
        if mf not in doa_cache:
            doa_cache[mf] = _calc_doa_for_mf(mf, train_cache)
        doa_values = doa_cache[mf]
        doa_col = f"{assay}_DoA"
        if doa_col not in df.columns:
            idx = df.columns.get_loc(assay) if assay in df.columns else len(df.columns)
            df.insert(idx + 1, doa_col, doa_values)
        else:
            df[doa_col] = doa_values

    df.to_excel(result_file, index=False)
    print(f"DoA results updated in {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DoA for prediction results")
    parser.add_argument("result_file", nargs="?", type=Path, help="Prediction result Excel file")
    parser.add_argument("--metadata-file", type=Path, default=RESULTS_DIR / "metadata.json", help="Metadata JSON file")
    args = parser.parse_args()

    res = args.result_file or _latest_result()
    meta = args.metadata_file
    main(res, meta)
