#!/usr/bin/env python
"""Calculate Domain of Applicability (DoA) for experiment data.

The training split used during model development is reconstructed and used as
reference when computing DoA values for fingerprints under the experiment
fingerprint directory. Results are written to an Excel file containing DTXSID
and SMILES columns from the experiment input along with assay specific DoA, MF
and Model information.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Predict_data import _tanimoto_max

try:
    from config import (
        SMILES_INPUT_PATH,
        FINGERPRINT_DIR,
        RESULTS_DIR,
        REF_FILE_PATH,
    )
except ImportError as exc:  # pragma: no cover - executed during runtime
    raise ImportError("Run from within ToxCast_model directory") from exc


def _load_train_fps(mf: str, assay: str, fp_base: Path, excel_path: Path) -> np.ndarray:
    """Return boolean fingerprint matrix for the reproduced training set."""
    fp_path = fp_base / f"{mf}.csv"
    drop_path = fp_base / f"{mf}_dropidx.csv"
    x = pd.read_csv(fp_path)

    drop_idx: List[int] = []
    if drop_path.exists() and drop_path.stat().st_size > 0:
        try:
            drop_idx = pd.read_csv(drop_path).iloc[:, 0].tolist()
        except pd.errors.EmptyDataError:
            drop_idx = []
    if drop_idx:
        x = x.drop(index=drop_idx).reset_index(drop=True)

    df = pd.read_excel(excel_path, sheet_name="data", header=1)
    y = df[assay]
    if drop_idx:
        y = y.drop(index=drop_idx).reset_index(drop=True)

    na_idx = y[y.isnull()].index
    if len(na_idx) > 0:
        x = x.drop(index=na_idx).reset_index(drop=True)
        y = y.drop(index=na_idx).reset_index(drop=True)

    train_x, _, _, _ = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=42
    )
    return train_x.astype(bool).to_numpy()


def _load_dropidx(path: Path) -> List[int]:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path).iloc[:, 0].tolist()
        except pd.errors.EmptyDataError:
            return []
    return []


def calc_doa(
    experiment_excel: Path,
    train_excel: Path,
    train_fp_base: Path,
    fingerprint_dir: Path,
    results_dir: Path,
) -> Path:
    """Calculate DoA for all assays listed in ``experiment_excel``.

    Returns the path to the generated Excel file.
    """
    assay_df = pd.read_excel(experiment_excel, sheet_name="assay_list")
    data_df = pd.read_excel(experiment_excel, sheet_name="data", header=1)

    first_mf = assay_df.iloc[0]["MF"]
    drop_idx = _load_dropidx(fingerprint_dir / f"{first_mf}_dropidx.csv")

    smiles = [s for i, s in enumerate(data_df["SMILES"]) if i not in drop_idx]
    dtxsid = (
        [sid for i, sid in enumerate(data_df["DTXSID"]) if i not in drop_idx]
        if "DTXSID" in data_df.columns
        else None
    )

    results = pd.DataFrame({"SMILES": smiles})
    if dtxsid is not None:
        results.insert(0, "DTXSID", dtxsid)

    train_cache: Dict[str, np.ndarray] = {}
    for _, row in assay_df.iterrows():
        assay = row["assay_name"]
        mf = row["MF"]
        model_type = row.get("Model", "")

        key = f"{assay}|{mf}"
        if key not in train_cache:
            train_cache[key] = _load_train_fps(mf, assay, train_fp_base, train_excel)
        train_fps = train_cache[key]

        fp_path = fingerprint_dir / f"{mf}.csv"
        pred_fp = pd.read_csv(fp_path)
        drop_idx_exp = _load_dropidx(fingerprint_dir / f"{mf}_dropidx.csv")
        if drop_idx_exp:
            pred_fp = pred_fp.drop(index=drop_idx_exp).reset_index(drop=True)

        doa_vals = [
            _tanimoto_max(fp.astype(bool), train_fps) for fp in pred_fp.to_numpy()
        ]

        doa_col = f"{assay}_DoA"
        mf_col = f"{assay}_MF"
        model_col = f"{assay}_Model"
        results[doa_col] = doa_vals
        results[mf_col] = [mf] * len(doa_vals)
        results[model_col] = [model_type] * len(doa_vals)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{experiment_excel.stem}_DoA.xlsx"
    results.to_excel(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate DoA for experiment data")
    parser.add_argument(
        "--experiment-excel",
        type=Path,
        default=SMILES_INPUT_PATH,
        help="Path to experiment Excel file or directory containing one",
    )
    parser.add_argument(
        "--train-excel",
        type=Path,
        default=REF_FILE_PATH,
        help="Training Excel file used for model development",
    )
    parser.add_argument(
        "--train-fp-base",
        type=Path,
        default=Path("data/ToxCast_v.4.1_v.2/fingerprints"),
        help="Base directory containing training fingerprints",
    )
    args = parser.parse_args()

    exp_excel = args.experiment_excel
    if exp_excel.is_dir():
        from toxcast_pkg.common import find_single_excel_file

        exp_excel = Path(find_single_excel_file(exp_excel))

    out_path = calc_doa(
        exp_excel,
        args.train_excel,
        args.train_fp_base,
        FINGERPRINT_DIR,
        RESULTS_DIR,
    )
    print(f"DoA results saved to {out_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
