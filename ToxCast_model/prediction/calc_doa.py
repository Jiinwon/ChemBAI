import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from Predict_data import _tanimoto_max

try:
    from config import (
        RESULTS_DIR,
        PREDICT_FP_PATH,
        PREDICT_SMILES_PATH,
        PREDICT_LIST_PATH,
        get_model_base,
    )
except ImportError:
    raise ImportError("Run from within ToxCast_model directory")


def load_latest_prediction() -> Path:
    files = sorted(Path(RESULTS_DIR).rglob('*_prediction.xlsx'))
    if not files:
        raise FileNotFoundError('No prediction file found in results directory')
    return files[-1]


def main(pred_file: Path) -> None:
    pred_df = pd.read_excel(pred_file)

    excel_path = PREDICT_LIST_PATH
    if Path(excel_path).is_dir():
        from toxcast_pkg.common import find_single_excel_file
        excel_path = find_single_excel_file(excel_path)
    assay_df = pd.read_excel(excel_path, sheet_name='assay_list')
    required_cols = {'assay_name', 'MF', 'Model'}
    if not required_cols.issubset(assay_df.columns):
        missing = required_cols.difference(assay_df.columns)
        raise KeyError(f"assay_list sheet missing columns: {', '.join(missing)}")
    assay_info = {
        row['assay_name']: {'MF': row['MF'], 'model_type': row['Model']}
        for _, row in assay_df.iterrows()
    }

    smiles_df = pd.read_excel(PREDICT_SMILES_PATH, sheet_name='data')
    drop_cache = {}
    train_cache = {}
    test_cache = {}
    model_base = get_model_base()

    for assay, info in assay_info.items():
        mf = info['MF']
        model_type = info['model_type']
        if assay not in train_cache:
            train_fp_path = Path(model_base) / f'{assay}_{mf}_{model_type}' / 'train_fps.csv'
            if not train_fp_path.exists():
                raise FileNotFoundError(f'Training fingerprints not found: {train_fp_path}')
            train_cache[assay] = pd.read_csv(train_fp_path).astype(bool).values
        if mf not in test_cache:
            test_fp = pd.read_csv(Path(PREDICT_FP_PATH) / f'{mf}.csv').astype(bool).values
            drop_idx_path = Path(PREDICT_FP_PATH) / f'{mf}_dropidx.csv'
            if drop_idx_path.exists() and drop_idx_path.stat().st_size > 0:
                drop_idx = pd.read_csv(drop_idx_path).iloc[:,0].tolist()
            else:
                drop_idx = []
            drop_cache[mf] = drop_idx
            if drop_idx:
                test_fp = np.delete(test_fp, drop_idx, axis=0)
            test_cache[mf] = test_fp
        train_fp = train_cache[assay]
        test_fp = test_cache[mf]
        doa_vals = [_tanimoto_max(fp.astype(bool), train_fp) for fp in test_fp]
        pred_df[f'{assay}_DoA'] = doa_vals

    # insert DTXSID if available
    if 'DTXSID' in smiles_df.columns:
        drop_idx = next(iter(drop_cache.values()), [])
        dtxsid_filtered = [sid for i, sid in enumerate(smiles_df['DTXSID']) if i not in drop_idx]
        if 'DTXSID' not in pred_df.columns:
            pred_df.insert(0, 'DTXSID', dtxsid_filtered)

    out_path = pred_file.with_name(pred_file.stem + '_doa.xlsx')
    pred_df.to_excel(out_path, index=False)
    print(f'DoA results saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate DoA for prediction results')
    parser.add_argument('prediction', nargs='?', type=Path, help='Prediction Excel file')
    args = parser.parse_args()
    file_path = args.prediction or load_latest_prediction()
    main(file_path)

