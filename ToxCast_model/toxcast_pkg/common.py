import sklearn
import numpy as np
import pandas as pd
import re
from pathlib import Path

from tqdm import tqdm

from itertools import product
from collections.abc import Iterable

from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)
from sklearn.cross_decomposition import PLSRegression


def data_split(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
    
    for train_idx, test_idx in sss.split(X, y):
        train_x = X.iloc[train_idx].reset_index(drop = True)
        train_y = y.iloc[train_idx].reset_index(drop = True)
        test_x = X.iloc[test_idx].reset_index(drop = True)
        test_y = y.iloc[test_idx].reset_index(drop = True)
    
    return train_x, train_y, test_x, test_y


def ParameterGrid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v))) 
    
    return params_grid


def CV(x, y, model, params, seed):
    skf = StratifiedKFold(n_splits = 5)
    
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    train_precision_ = []
    train_recall_ = []
    train_f1_ = []
    train_accuracy_ = []
    
    val_precision_ = []
    val_recall_ = []
    val_f1_ = []
    val_accuracy_ = []
    
    for train_idx, val_idx in skf.split(x, y):
        train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]
        
        try:
            clf = model(random_state = seed, **params)
        except:
            clf = model(**params)
        
        # clf.fit(train_x, train_y)
        
        if model == sklearn.cross_decomposition._pls.PLSRegression:
            onehot_train_y = pd.get_dummies(train_y)
            
            clf.fit(train_x, onehot_train_y)
            
            train_pred = np.argmax(clf.predict(train_x), axis = 1)
            val_pred = np.argmax(clf.predict(val_x), axis = 1)
            
        else:
            clf.fit(train_x, train_y)
            
            train_pred = clf.predict(train_x)
            val_pred = clf.predict(val_x)
        
        train_precision_.append(precision_score(train_y, train_pred, average = 'binary'))
        train_recall_.append(recall_score(train_y, train_pred, average = 'binary'))
        train_f1_.append(f1_score(train_y, train_pred, average = 'binary'))
        train_accuracy_.append(accuracy_score(train_y, train_pred))

        val_precision_.append(precision_score(val_y, val_pred, average = 'binary'))
        val_recall_.append(recall_score(val_y, val_pred, average = 'binary'))
        val_f1_.append(f1_score(val_y, val_pred, average = 'binary'))
        val_accuracy_.append(accuracy_score(val_y, val_pred))
        
    result = dict(zip(['params'] + train_metrics + val_metrics, 
                      [params] + [np.mean(train_precision_), 
                                  np.mean(train_recall_), 
                                  np.mean(train_f1_), 
                                  np.mean(train_accuracy_), 
                                  np.mean(val_precision_), 
                                  np.mean(val_recall_), 
                                  np.mean(val_f1_), 
                                  np.mean(val_accuracy_)]))
    
    return(result)


def find_single_excel_file(base_dir):
    """Return the single Excel file directly inside base_dir.
    If multiple or none exist, raise an error."""
    import os
    files = [f for f in os.listdir(base_dir) if f.lower().endswith((".xlsx", ".xls"))]
    if len(files) == 0:
        raise FileNotFoundError(f"{base_dir} 디렉토리에 엑셀 파일이 없습니다.")
    if len(files) > 1:
        raise RuntimeError(f"{base_dir} 디렉토리에 예측/훈련용 데이터셋을 하나만 남기세요.")
    return os.path.join(base_dir, files[0])


def check_required_sheets(excel_path, sheets):
    """Raise KeyError if any of ``sheets`` is missing in ``excel_path``."""
    import pandas as pd

    xl = pd.ExcelFile(excel_path)
    missing = [s for s in sheets if s not in xl.sheet_names]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"{excel_path} 파일에 필요한 시트({missing_str})가 없습니다.")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(s):
        s = str(s).replace("\u200b", "").replace("\ufeff", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    df.columns = [_norm(c) for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    for cand in ("smiles","canonical smiles","canonical_smiles","mol_smiles",
                 "structure_smiles","cano_smiles","smile","smiles*"):
        if cand in lower:
            df = df.rename(columns={lower[cand]: "SMILES"})
            break
    for cand in ("dtxsid","dtxs_id"):
        if cand in lower:
            df = df.rename(columns={lower[cand]: "DTXSID"})
            break
    return df


def _detect_header_row(xlsx: Path, sheet: str="data") -> int:
    for h in (0, 1):
        try:
            df = pd.read_excel(xlsx, sheet_name=sheet, header=h, nrows=2)
            cols = [str(c).strip().lower() for c in df.columns]
            if ("smiles" in cols) or ("dtxsid" in cols) or sum(("_" in c) or (" " in c) for c in cols) >= 2:
                return h
        except Exception:
            pass
    return 1


def read_data_with_smiles(xlsx_path: str | Path, sheet: str="data") -> pd.DataFrame:
    xlsx = Path(xlsx_path)
    h = _detect_header_row(xlsx, sheet=sheet)
    df = pd.read_excel(xlsx, sheet_name=sheet, header=h)
    df = _standardize_columns(df)
    if "SMILES" not in df.columns:
        cands = [c for c in df.columns if re.search(r"smile", c, re.IGNORECASE)]
        if cands:
            df = df.rename(columns={cands[0]: "SMILES"})
    if "SMILES" not in df.columns:
        raise KeyError(f"'SMILES' 컬럼을 찾지 못했습니다. 파일={xlsx_path}, 시트={sheet}, header={h}, columns={list(df.columns)}")
    return df

