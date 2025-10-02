#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
import pandas as pd

# ====== 설정 ======
LOG_DIR = Path("/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model/experiments/training/DART_MTL/logs/seed_multi")
OUT_XLSX = LOG_DIR.parent / "seed_multi_best_summary.xlsx"
SELECT_BY = os.getenv("SELECT_BY", "validation").strip().lower()  # validation | test

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
    (예: multi_ATG_Ahr_CIS_rf_Morgan.log)
    오른쪽부터 mf, 그 앞이 model, 나머지 전부 assay_name
    assay_name 앞의 'multi_' 접두어는 제거한다.
    """
    stem = log_path.stem  # .log 제외
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, "", "", stem
    mf = parts[-1]
    model = parts[-2]
    assay_name = "_".join(parts[:-2])

    # 접두어 'multi_' 제거
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
    # 숫자: 소수/지수표기 허용
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
                # (필요 시 주석 처리 가능)
                current_seed = -1
                seed_blocks.setdefault(current_seed, {})

            mv = re_val.search(line)
            if mv:
                metric = mv.group(1)  # F1 Score | Precision | Recall | Accuracy | AUC
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

def pick_best_seed(seed_rows, select_by="validation"):
    """
    select_by: 'validation' | 'test'
    """
    key = "Validation F1 Score" if select_by == "validation" else "Test F1 Score"
    best_seed, best_score = None, None
    for s, m in seed_rows.items():
        if key in m and m[key] is not None:
            if (best_score is None) or (m[key] > best_score):
                best_seed, best_score = s, m[key]
    return best_seed

def main():
    rows = []
    log_files = sorted(LOG_DIR.rglob("*.log"))
    for log_path in log_files:
        assay_name, model, mf, algo = parse_name_parts(log_path)
        seed_rows = parse_log_file(log_path)
        if not seed_rows:
            continue

        best_seed = pick_best_seed(seed_rows, SELECT_BY)
        if best_seed is None:
            # 선택 기준(Test/Validation F1 Score)이 아예 없는 로그는 패스
            continue

        m = seed_rows[best_seed]
        row = {
            "Database": assay_name,
            "Model": model,
            "MF/MD": mf,
            "Algorithm": algo,
            "Test F1": m.get("Test F1 Score"),
            "Test Precision": m.get("Test Precision"),
            "Test Recall": m.get("Test Recall"),
            "Test AUC": m.get("Test AUC"),
            "Test Accuracy": m.get("Test Accuracy"),
            "Validation F1": m.get("Validation F1 Score"),
            "Validation Precision": m.get("Validation Precision"),
            "Validation Recall": m.get("Validation Recall"),
            "Validation AUC": m.get("Validation AUC"),
            "Validation Accuracy": m.get("Validation Accuracy"),
        }
        rows.append(row)

    if not rows:
        print("[INFO] 요약할 결과가 없습니다(선택 기준 지표가 없는 로그만 있었을 가능성).")
        return

    df = pd.DataFrame(rows, columns=[
        "Database","Model","MF/MD","Algorithm",
        "Test F1","Test Precision","Test Recall","Test AUC","Test Accuracy",
        "Validation F1","Validation Precision","Validation Recall","Validation AUC","Validation Accuracy"
    ])
    df.sort_values(by=["Database", "Model", "MF/MD"], inplace=True, kind="mergesort")

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="seed_best")
    print(f"[OK] 저장 완료: {OUT_XLSX}")

if __name__ == "__main__":
    main()
