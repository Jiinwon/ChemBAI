import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from Predict_data import _tanimoto_max

try:
    from config import TRAIN_FILE_PATH
except ImportError:
    raise ImportError('Run from within ToxCast_model directory')


def load_assay_mf(excel_path: Path, assay_name: str) -> str:
    """Return fingerprint type (MF) for assay_name from assay_list sheet."""
    assay_df = pd.read_excel(excel_path, sheet_name="assay_list")
    row = assay_df[assay_df["assay_name"] == assay_name]
    if row.empty:
        raise KeyError(f"assay_name {assay_name} not found in assay_list")
    return row.iloc[0]["MF"]


def load_training_fps(excel_path: Path, fp_base: Path, assay: str, mf: str) -> np.ndarray:
    fp_path = fp_base / f"{mf}.csv"
    drop_path = fp_base / f"{mf}_dropidx.csv"

    x = pd.read_csv(fp_path)
    drop_idx = []
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

    from sklearn.model_selection import train_test_split

    train_x, _, _, _ = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=42
    )

    return train_x.astype(bool).to_numpy()


def calc_doa_matrix(train_fps: np.ndarray) -> np.ndarray:
    doa_values = []
    for i, fp in enumerate(train_fps):
        ref = np.delete(train_fps, i, axis=0)
        if ref.size == 0:
            doa_values.append(1.0)
        else:
            doa_values.append(_tanimoto_max(fp, ref))
    return np.array(doa_values)


def main(assay_name: str, excel_path: Path, fp_base: Path) -> None:
    mf = load_assay_mf(excel_path, assay_name)
    train_fps = load_training_fps(excel_path, fp_base, assay_name, mf)
    doa = calc_doa_matrix(train_fps)

    out_path = fp_base / f"{assay_name}_{mf}_train_doa.csv"
    pd.DataFrame({"DoA": doa}).to_csv(out_path, index=False)
    print(f"Training DoA saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DoA for training data")
    parser.add_argument("assay", help="Assay name to calculate DoA for")
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=TRAIN_FILE_PATH,
        help="Excel file with data and assay_list sheets",
    )
    parser.add_argument(
        "--fp-base",
        type=Path,
        default=Path("data/ToxCast_v.4.1_v.2/fingerprints"),
        help="Base directory containing training fingerprints",
    )
    args = parser.parse_args()
    main(args.assay, args.excel_path, args.fp_base)
