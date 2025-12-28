#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refit_from_log_params_precomputed_fp.py

- 로그 파일에서 seed별 "Best Model Parameters" 파싱
- seed별 train_df.csv + fingerprints/train/{FP}.csv 로드(재생성 없음)
- 라벨(0/1) 있는 행만 학습하여 모델 refit
- model.joblib을 backup 경로에 안전 저장(tmp -> os.replace)

참고(공식 문서):
- os.replace (atomic rename 성격): https://docs.python.org/3/library/os.html#os.replace
- joblib.dump/load: https://joblib.readthedocs.io/
"""

import argparse
import ast
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb  # noqa
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def safe_joblib_dump(obj: Any, final_path: Path, compress: int = 3,
                     retries: int = 4, sleep_sec: float = 2.0) -> Path:
    """
    임시파일로 저장 후 os.replace로 교체(가능한 범위에서 원자적).
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    last_err = None
    for k in range(retries):
        try:
            if tmp_path.exists():
                tmp_path.unlink()

            joblib.dump(obj, tmp_path, compress=compress)

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

    start_pat = re.compile(rf"START\s+seed_{seed}\s*:", re.IGNORECASE)
    m = start_pat.search(text)
    if not m:
        raise ValueError(f"START seed_{seed}: not found in {log_path}")

    tail = text[m.end():]
    bp_pat = re.compile(r"Best Model Parameters:\s*(\{.*?\})")
    m2 = bp_pat.search(tail)
    if not m2:
        raise ValueError(f"Best Model Parameters not found for seed_{seed} in {log_path}")

    d = ast.literal_eval(m2.group(1))
    if not isinstance(d, dict):
        raise ValueError("parsed params is not dict")
    return d


def build_model(model_name: str, best_params: Dict, seed: int):
    if model_name == "dt":
        m = DecisionTreeClassifier(random_state=seed)
    elif model_name == "rf":
        m = RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif model_name == "gbt":
        m = GradientBoostingClassifier(random_state=seed)
    elif model_name == "logistic":
        # max_iter는 넉넉히, best_params에 있으면 override됨
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

    allowed = set(m.get_params().keys())
    filtered = {k: v for k, v in best_params.items() if k in allowed}
    m.set_params(**filtered)
    return m


def load_fp_matrix_csv(fp_dir: Path, fp_name: str) -> np.ndarray:
    """
    fingerprints/train 에 저장된 {FP}.csv 로드
    - 헤더/인덱스 컬럼(Unnamed: 0 등) 제거
    - 전부 numeric 변환(비수치 -> 0)
    """
    fp_dir = Path(fp_dir)
    if not fp_dir.exists():
        raise FileNotFoundError(f"fp_dir missing: {fp_dir}")

    # 1) 가장 흔한 정규 경로
    direct = fp_dir / f"{fp_name}.csv"
    if direct.exists():
        path = direct
    else:
        # 2) 혹시 파일명이 살짝 다를 경우 대비(대소문자/접두어/접미어)
        fp_low = fp_name.lower()
        cands = [p for p in fp_dir.rglob("*.csv") if fp_low in p.name.lower()]
        if not cands:
            raise FileNotFoundError(f"fingerprint file not found for fp={fp_name} in {fp_dir}")
        # 크기 큰 파일 우선(보통 전체 행렬이 가장 큼)
        cands.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        path = cands[0]

    log(f"loading fp csv: {path}")
    df_fp = pd.read_csv(path)

    # 흔한 인덱스 컬럼 제거
    for col in ["Unnamed: 0", "index", "Index"]:
        if col in df_fp.columns:
            df_fp = df_fp.drop(columns=[col])

    # 전부 수치로 변환
    df_fp = df_fp.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = df_fp.to_numpy()

    # dtype 정리(용량/속도)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--assay", type=str, required=True)
    ap.add_argument("--model", type=str, required=True,
                    choices=["dt", "rf", "xgb", "gbt", "logistic"])
    ap.add_argument("--fp", type=str, required=True,
                    choices=["MACCS", "Morgan", "RDKit", "Pattern", "Layered"])
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--fp_dir", type=str, required=True)
    args = ap.parse_args()

    seed = args.seed
    assay = args.assay
    model_name = args.model
    fp = args.fp

    log_path = Path(args.log)
    out_path = Path(args.out)
    train_csv = Path(args.train_csv)
    fp_dir = Path(args.fp_dir)

    log(f"seed={seed} assay={assay} model={model_name} fp={fp}")
    log(f"log_path={log_path}")
    log(f"train_csv={train_csv}")
    log(f"fp_dir={fp_dir}")
    log(f"out_path={out_path}")

    if not log_path.exists():
        raise FileNotFoundError(f"log missing: {log_path}")
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv missing: {train_csv}")

    # 1) params
    best_params = parse_best_params_for_seed(log_path, seed)
    log(f"best_params={best_params}")

    # 2) train df + label mask
    df = pd.read_csv(train_csv)
    if assay not in df.columns:
        raise ValueError(f"assay column not found in train_df: {assay}")

    y = pd.to_numeric(df[assay], errors="coerce")
    mask = y.isin([0, 1]).to_numpy()
    n_lab = int(mask.sum())
    if n_lab == 0:
        raise ValueError(f"no labeled rows for assay={assay}")

    y_fit = y.to_numpy()[mask].astype(int)

    # 3) precomputed FP matrix (CSV)
    X_all = load_fp_matrix_csv(fp_dir, fp)

    # 행 정합성 체크
    if X_all.shape[0] != df.shape[0]:
        raise ValueError(f"row mismatch: fp_rows={X_all.shape[0]} vs df_rows={df.shape[0]}")

    X = X_all[mask]

    # 4) build + fit
    model = build_model(model_name, best_params, seed=seed)
    log(f"fitting: n={X.shape[0]} dim={X.shape[1]}")
    model.fit(X, y_fit)

    # 5) save
    saved = safe_joblib_dump(model, out_path, compress=3, retries=4, sleep_sec=2.0)
    log(f"saved: {saved}")


if __name__ == "__main__":
    main()
