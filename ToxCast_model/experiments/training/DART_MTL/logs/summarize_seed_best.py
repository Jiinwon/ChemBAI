#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ====== 설정 ======
LOG_DIR = Path("/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model/experiments/training/DART_MTL/logs/seed_multi")
OUT_XLSX = LOG_DIR.parent / "seed_multi_best_summary.xlsx"
# 사용 여부 X: 이전 코드 호환용으로 남김
SELECT_BY = os.getenv("SELECT_BY", "validation").strip().lower()

# 시드 열 구성(필요 시 확장/수정)
SEED_LIST = [0, 1, 2]
DECIMALS = 4  # 출력 소수점 자릿수

ALGO_MAP = {
    "dt": "DecisionTree",
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "gbt": "GradientBoosting",
    "lr": "LogisticRegression",
    "svm": "SVM",
    "knn": "KNN",
    "mlp": "MLP",
    "gnn": "GraphNeuralNet",
    "gat": "GraphAttention",
    "gcn": "GraphConvNet",
    "gin": "GraphIsomorphism",
    "logistic": "LogisticRegression",
}

def parse_name_parts(log_path: Path):
    """
    파일명: {assay_name}_{model}_{mf}.log
    오른쪽부터 mf, model, 나머지는 assay_name
    assay_name 앞의 'multi_' 접두어는 제거.
    """
    stem = log_path.stem  # .log 제외
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, "", "", stem
    mf = parts[-1]
    model = parts[-2]
    assay_name = "_".join(parts[:-2])

    if assay_name.startswith("multi_"):
        assay_name = assay_name[len("multi_"):]

    algo = ALGO_MAP.get(model, model)
    return assay_name, model, mf, algo

def float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_log_file(log_path: Path):
    """
    한 로그에서 seed별 Validation/*, Test/* 값을 추출해 dict 반환.
    반환: dict(seed -> metrics dict)
    """
    seed_blocks = {}
    current_seed = None

    re_start = re.compile(r"\bSTART\s+seed_(\d+):", re.I)
    re_holdout = re.compile(r"\bHoldout\s+Validation\b", re.I)
    NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"
    re_val = re.compile(rf"\bValidation\s+(F1 Score|Precision|Recall|Accuracy|AUC):\s+{NUM}", re.I)
    re_test = re.compile(rf"\bTest\s+(F1 Score|Precision|Recall|Accuracy|AUC):\s+{NUM}", re.I)

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if re_holdout.search(line):
                continue

            ms = re_start.search(line)
            if ms:
                current_seed = int(ms.group(1))
                seed_blocks.setdefault(current_seed, {})
                continue

            if current_seed is None:
                # seed 표기가 누락된 로그라면 임시 seed -1로 묶기
                current_seed = -1
                seed_blocks.setdefault(current_seed, {})

            mv = re_val.search(line)
            if mv:
                metric = mv.group(1)
                value = float_or_none(mv.group(2))
                seed_blocks[current_seed][f"Validation {metric}"] = value
                continue

            mt = re_test.search(line)
            if mt:
                metric = mt.group(1)
                value = float_or_none(mt.group(2))
                seed_blocks[current_seed][f"Test {metric}"] = value
                continue

    # 성능이 하나도 없는 seed는 제거
    cleaned = {
        s: m for s, m in seed_blocks.items()
        if any(k.startswith(("Validation", "Test")) and m.get(k) is not None for k in m)
    }
    return cleaned

def format_mean_std(values, decimals=DECIMALS):
    """NaN 제외 평균/표준편차를 'm (s)' 형식으로 반환. 표본표준편차(ddof=1)."""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if len(vals) == 0:
        return ""
    if len(vals) == 1:
        m = float(vals[0])
        s = 0.0
    else:
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1))
    return f"{m:.{decimals}f} ({s:.{decimals}f})"

def main():
    rows = []
    log_files = sorted(LOG_DIR.rglob("*.log"))

    for log_path in log_files:
        assay_name, model, mf, algo = parse_name_parts(log_path)
        seed_rows = parse_log_file(log_path)
        if not seed_rows:
            continue

        # 각 시드의 Test F1 점수 수집
        seed_f1_map = {}
        for s in SEED_LIST:
            metrics = seed_rows.get(s, {})
            seed_f1_map[s] = metrics.get("Test F1 Score")

        # 모든 시드가 None이면 스킵
        if all((v is None or pd.isna(v)) for v in seed_f1_map.values()):
            continue

        # 평균(표준편차) 문자열 생성
        mean_std_str = format_mean_std(list(seed_f1_map.values()), decimals=DECIMALS)

        # 결과 행 구성
        row = {
            "Database": assay_name,
            "Model": model,
            "MF/MD": mf,
            "Algorithm": algo,
        }
        # seed 열 추가
        for s in SEED_LIST:
            row[f"seed_{s}"] = seed_f1_map[s]
        row["Mean(±std)"] = mean_std_str

        rows.append(row)

    if not rows:
        print("[INFO] 요약할 결과가 없습니다(시드별 Test F1 값을 찾지 못함).")
        return

    # 컬럼 순서 정의
    base_cols = ["Database", "Model", "MF/MD", "Algorithm"]
    seed_cols = [f"seed_{s}" for s in SEED_LIST]
    out_cols = base_cols + seed_cols + ["Mean(±std)"]

    df = pd.DataFrame(rows, columns=out_cols)
    df.sort_values(by=["Database", "Model", "MF/MD"], inplace=True, kind="mergesort")

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="seed_f1_summary")
    print(f"[OK] 저장 완료: {OUT_XLSX}")

if __name__ == "__main__":
    main()
