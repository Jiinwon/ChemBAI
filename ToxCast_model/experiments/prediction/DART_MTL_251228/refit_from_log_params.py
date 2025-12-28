#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump, load as joblib_load

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb  # noqa
    HAS_XGB = True
except Exception:
    HAS_XGB = False


MACCS_BITS = 167


def log(msg: str):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def safe_joblib_dump(obj, final_path: Path, compress=3, retries=3, sleep_sec=2.0) -> Path:
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    last_err = None
    for k in range(retries):
        try:
            if tmp_path.exists():
                tmp_path.unlink()

            joblib_dump(obj, tmp_path, compress=compress)

            if (not tmp_path.exists()) or tmp_path.stat().st_size == 0:
                raise OSError(f"tmp missing/empty: {tmp_path}")

            os.replace(tmp_path, final_path)

            if (not final_path.exists()) or final_path.stat().st_size == 0:
                raise OSError(f"final missing/empty: {final_path}")

            return final_path

        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * (k + 1))

    raise RuntimeError(f"save failed after retries: {final_path} | last={last_err}")


def parse_best_params_for_seed(log_path: Path, seed: int) -> Dict:
    """
    로그에서
      START seed_{seed}:...
      ... Best Model Parameters: {...}
    를 찾아 dict로 반환
    """
    text = log_path.read_text(errors="ignore")
    # seed 블록 시작점
    start_pat = re.compile(rf"START\s+seed_{seed}\s*:", re.IGNORECASE)
    m = start_pat.search(text)
    if not m:
        raise ValueError(f"START seed_{seed}: not found in {log_path}")

    # 시작점 이후로 Best Model Parameters 찾기
    tail = text[m.end():]
    bp_pat = re.compile(r"Best Model Parameters:\s*(\{.*?\})")
    m2 = bp_pat.search(tail)
    if not m2:
        raise ValueError(f"Best Model Parameters not found for seed_{seed} in {log_path}")

    d = ast.literal_eval(m2.group(1))
    if not isinstance(d, dict):
        raise ValueError("parsed params is not dict")
    return d


def smiles_to_mol(smiles: str):
    if smiles is None:
        return None
    s = str(smiles).strip()
    if s == "" or s.lower() == "nan":
        return None
    return Chem.MolFromSmiles(s)


def bitvect_to_uint8(fp, dim: int) -> np.ndarray:
    arr = np.zeros((dim,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.uint8, copy=False)


def compute_fp_matrix(smiles_list, fp_name: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(smiles_list)
    X = np.zeros((n, dim), dtype=np.uint8)
    valid = np.zeros((n,), dtype=bool)

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue

        if fp_name == "MACCS":
            fp = MACCSkeys.GenMACCSKeys(mol)
            if fp.GetNumBits() != MACCS_BITS:
                continue
            X[i, :] = bitvect_to_uint8(fp, MACCS_BITS)
            valid[i] = True
            continue

        if fp_name == "Morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=dim)
        elif fp_name == "RDKit":
            fp = Chem.RDKFingerprint(mol, fpSize=dim)
        elif fp_name == "Pattern":
            fp = Chem.PatternFingerprint(mol, fpSize=dim)
        elif fp_name == "Layered":
            fp = Chem.LayeredFingerprint(mol, fpSize=dim)
        else:
            raise ValueError(f"Unknown fp: {fp_name}")

        if fp.GetNumBits() != dim:
            continue
        X[i, :] = bitvect_to_uint8(fp, dim)
        valid[i] = True

    return X, valid


def infer_dim_from_any_existing_model(results_root: Path, seed: int, model_name: str, fp: str) -> int:
    """
    같은 seed/model/fp 조합의 다른 assay 모델 중 하나를 찾아 n_features_in_로 dim 추정.
    없으면 fp 기준 기본값.
    """
    if fp == "MACCS":
        return MACCS_BITS

    seed_dir = results_root / f"seed_{seed}"
    if not seed_dir.exists():
        return 2048

    candidates = sorted(seed_dir.glob(f"*_{model_name}_{fp}/model.joblib"))
    for p in candidates:
        try:
            m = joblib_load(p)
            if hasattr(m, "n_features_in_"):
                return int(m.n_features_in_)
        except Exception:
            continue

    return 2048


def build_model(model_name: str, best_params: Dict, seed: int):
    if model_name == "dt":
        m = DecisionTreeClassifier(random_state=seed)
    elif model_name == "rf":
        m = RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif model_name == "gbt":
        m = GradientBoostingClassifier(random_state=seed)
    elif model_name == "logistic":
        m = LogisticRegression(random_state=seed, max_iter=5000, n_jobs=-1)
    elif model_name == "xgb":
        if not HAS_XGB:
            raise RuntimeError("xgboost not available in this env")
        m = xgb.XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # best_params 중에서 해당 모델이 받을 수 있는 키만 적용
    allowed = set(m.get_params().keys())
    filtered = {k: v for k, v in best_params.items() if k in allowed}
    m.set_params(**filtered)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--assay", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=["dt", "rf", "xgb", "gbt", "logistic"])
    ap.add_argument("--fp", type=str, required=True, choices=["MACCS", "Morgan", "RDKit", "Pattern", "Layered"])
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--excel", type=str, required=True)
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--sheet", type=str, default="data")
    args = ap.parse_args()

    seed = args.seed
    assay = args.assay
    model_name = args.model
    fp = args.fp

    log_path = Path(args.log)
    out_path = Path(args.out)
    excel_path = Path(args.excel)
    results_root = Path(args.results_root)

    log(f"seed={seed} assay={assay} model={model_name} fp={fp}")
    log(f"log_path={log_path}")
    log(f"excel_path={excel_path}")
    log(f"out_path={out_path}")

    if not log_path.exists():
        raise FileNotFoundError(f"log missing: {log_path}")
    if not excel_path.exists():
        raise FileNotFoundError(f"excel missing: {excel_path}")

    # 1) best params 파싱
    best_params = parse_best_params_for_seed(log_path, seed)
    log(f"best_params={best_params}")

    # 2) 데이터 로드
    df = pd.read_excel(excel_path, sheet_name=args.sheet, engine="openpyxl")
    smiles_col = None
    for c in df.columns:
        if str(c).strip().lower() == "smiles":
            smiles_col = c
            break
    if smiles_col is None:
        raise ValueError("SMILES column not found")

    if assay not in df.columns:
        raise ValueError(f"assay column not found in excel: {assay}")

    y = pd.to_numeric(df[assay], errors="coerce")
    mask = y.isin([0, 1]).to_numpy()
    if mask.sum() == 0:
        raise ValueError(f"no labeled rows for assay={assay}")

    smiles_list = df.loc[mask, smiles_col].tolist()
    y_train = y.to_numpy()[mask].astype(int)

    # 3) dim 추정 + FP 생성
    dim = infer_dim_from_any_existing_model(results_root, seed, model_name, fp)
    log(f"fingerprint_dim={dim}")

    X_u8, valid = compute_fp_matrix(smiles_list, fp, MACCS_BITS if fp == "MACCS" else dim)
    valid_mask = valid
    if valid_mask.sum() == 0:
        raise ValueError("all SMILES invalid after parsing")

    X = X_u8[valid_mask].astype(np.float32)  # sklearn 안정성
    y_fit = y_train[valid_mask]

    # 4) 모델 생성 + fit
    model = build_model(model_name, best_params, seed=seed)
    log(f"fitting: n={X.shape[0]} dim={X.shape[1]}")
    model.fit(X, y_fit)

    # 5) 안전 저장
    saved = safe_joblib_dump(model, out_path, compress=3, retries=4, sleep_sec=2.0)
    log(f"saved: {saved}")

if __name__ == "__main__":
    main()
