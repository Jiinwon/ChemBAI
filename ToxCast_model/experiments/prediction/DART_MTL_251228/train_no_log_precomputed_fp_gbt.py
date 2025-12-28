#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def safe_joblib_dump(obj: Any, final_path: Path, compress: int = 3,
                     retries: int = 4, sleep_sec: float = 2.0) -> Path:
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


def load_fp_matrix_csv(fp_dir: Path, fp_name: str) -> np.ndarray:
    fp_dir = Path(fp_dir)
    if not fp_dir.exists():
        raise FileNotFoundError(f"fp_dir missing: {fp_dir}")

    # 정규 경로 우선
    direct = fp_dir / f"{fp_name}.csv"
    if direct.exists():
        path = direct
    else:
        # 혹시 파일명 변형 대비(대소문자 등)
        fp_low = fp_name.lower()
        cands = [p for p in fp_dir.rglob("*.csv") if fp_low in p.name.lower()]
        if not cands:
            raise FileNotFoundError(f"fingerprint file not found for fp={fp_name} in {fp_dir}")
        cands.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
        path = cands[0]

    log(f"loading fp csv: {path}")
    df_fp = pd.read_csv(path)

    # 흔한 인덱스 컬럼 제거
    for col in ["Unnamed: 0", "index", "Index"]:
        if col in df_fp.columns:
            df_fp = df_fp.drop(columns=[col])

    df_fp = df_fp.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = df_fp.to_numpy()

    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def build_param_grid(grid_mode: str) -> Dict[str, list]:
    # fast: 너무 오래 안 걸리게(기본값)
    if grid_mode == "fast":
        return {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "min_samples_leaf": [1, 3],
            # 아래는 고정(탐색 폭 줄이기)
            "min_samples_split": [2],
            "subsample": [1.0],
        }
    # full: 조금 더 넓게
    if grid_mode == "full":
        return {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 5],
            "min_samples_leaf": [1, 3, 5],
            "min_samples_split": [2, 5],
            "subsample": [0.8, 1.0],
        }
    raise ValueError(f"unknown grid_mode: {grid_mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--assay", type=str, required=True)
    ap.add_argument("--fp", type=str, required=True, choices=["Layered", "RDKit", "Pattern"])
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--fp_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--grid", type=str, default="fast", choices=["fast", "full"])
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--scoring", type=str, default="f1")  # binary f1
    args = ap.parse_args()

    seed = args.seed
    assay = args.assay
    fp = args.fp

    train_csv = Path(args.train_csv)
    fp_dir = Path(args.fp_dir)
    out_path = Path(args.out)

    log(f"seed={seed} assay={assay} model=gbt fp={fp}")
    log(f"train_csv={train_csv}")
    log(f"fp_dir={fp_dir}")
    log(f"out_path={out_path}")
    log(f"grid={args.grid} cv={args.cv} n_jobs={args.n_jobs} scoring={args.scoring}")

    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv missing: {train_csv}")
    if not fp_dir.exists():
        raise FileNotFoundError(f"fp_dir missing: {fp_dir}")

    # 1) data
    df = pd.read_csv(train_csv)
    if assay not in df.columns:
        raise ValueError(f"assay column not found in train_df: {assay}")

    y = pd.to_numeric(df[assay], errors="coerce")
    mask = y.isin([0, 1]).to_numpy()
    n_lab = int(mask.sum())
    if n_lab == 0:
        raise ValueError(f"no labeled rows for assay={assay}")

    y_fit = y.to_numpy()[mask].astype(int)

    # 클래스가 한쪽으로만 있으면 학습 불가
    if np.unique(y_fit).size < 2:
        raise ValueError(f"only one class in labeled rows for assay={assay} (n_lab={n_lab})")

    # 2) FP
    X_all = load_fp_matrix_csv(fp_dir, fp)
    if X_all.shape[0] != df.shape[0]:
        raise ValueError(f"row mismatch: fp_rows={X_all.shape[0]} vs df_rows={df.shape[0]}")

    X = X_all[mask]
    log(f"X shape={X.shape}, y shape={y_fit.shape}, labeled={n_lab}")

    # 3) model + grid search
    base = GradientBoostingClassifier(random_state=seed)
    param_grid = build_param_grid(args.grid)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=seed)

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True,
        verbose=1,
    )

    gs.fit(X, y_fit)

    best_model = gs.best_estimator_
    best_params = gs.best_params_
    best_score = float(gs.best_score_)

    # 4) sanity: train f1(참고용)
    yhat = best_model.predict(X)
    train_f1 = float(f1_score(y_fit, yhat, zero_division=0))

    # 5) save
    saved = safe_joblib_dump(best_model, out_path, compress=3, retries=4, sleep_sec=2.0)

    report = {
        "seed": seed,
        "assay": assay,
        "model": "gbt",
        "fp": fp,
        "n_labeled": n_lab,
        "best_params": best_params,
        "best_cv_score_f1": best_score,
        "train_f1_on_labeled": train_f1,
        "train_csv": str(train_csv),
        "fp_dir": str(fp_dir),
        "model_path": str(saved),
        "grid_mode": args.grid,
        "cv_folds": args.cv,
        "scoring": args.scoring,
    }
    report_path = out_path.parent / "train_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    log(f"saved model: {saved}")
    log(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
