#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate Domain of Applicability (DoA) for experiment data.

- config.py만 사용 (터미널 인자 없음)
- REF_FILE_PATH: 훈련 시 사용한 fingerprint 디렉토리 (예: .../fingerprints/)
  - 여기서 {MF}.csv, 선택적으로 {MF}_dropidx.csv를 읽어 '훈련 지문 집합'으로 사용
- experiments/.../fingerprints/{MF}.csv: 실험 대상 화학물질의 fingerprint
- DoA = 각 실험 FP 행과 훈련 FP 행들 간 Tanimoto 유사도의 최대값

출력: experiments/<object>/<PROJECT_NAME>/results/<experiment_stem>_DoA.xlsx
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from Predict_data import _tanimoto_max

# --- config 가져오기 ---
try:
    from config import (
        SMILES_INPUT_PATH,   # experiments/<object>/<PROJECT_NAME> (엑셀 1개만)
        FINGERPRINT_DIR,     # 실험 FP 디렉토리
        RESULTS_DIR,         # 출력 디렉토리
        REF_FILE_PATH,       # 훈련 FP 디렉토리 (여기엔 엑셀이 없음)
        validate_paths,      # (data, assay_list) 시트 존재 검증
    )
except ImportError as exc:
    raise ImportError("ToxCast_model 디렉토리 내에서 실행하세요.") from exc

# 단일 엑셀 자동 탐색 유틸 및 공용 함수
try:
    from toxcast_pkg.common import find_single_excel_file, read_data_with_smiles, _standardize_columns
except Exception:
    raise ImportError("toxcast_pkg.common 관련 함수를 import할 수 없습니다. PYTHONPATH를 확인하세요.")


# =========================
# 헬퍼
# =========================
def _load_dropidx(path: Path) -> List[int]:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path).iloc[:, 0].tolist()
        except pd.errors.EmptyDataError:
            return []
    return []


# =========================
# 훈련 FP 로딩: REF_FILE_PATH(디렉토리)에서 바로 읽음
# =========================
def _load_train_fps(mf: str, train_fp_base: Path) -> np.ndarray:
    """
    훈련 fingerprint 행렬(boolean)을 반환.
    - train_fp_base / "{mf}.csv"
    - 선택적으로 train_fp_base / "{mf}_dropidx.csv" 적용
    """
    fp_path = train_fp_base / f"{mf}.csv"
    if not fp_path.exists():
        raise FileNotFoundError(f"훈련 FP 파일이 없습니다: {fp_path}")

    x = pd.read_csv(fp_path)
    drop_path = train_fp_base / f"{mf}_dropidx.csv"
    drop_idx = _load_dropidx(drop_path)
    if drop_idx:
        x = x.drop(index=drop_idx).reset_index(drop=True)

    return x.astype(bool).to_numpy()


# =========================
# 경로 해석
# =========================
def resolve_inputs() -> Tuple[Path, Path, Path, Path]:
    """
    config 기반 경로 확정.
    반환: (experiment_excel, train_fp_base, fingerprint_dir, results_dir)
    """
    validate_paths()

    # 1) experiment excel
    p = Path(SMILES_INPUT_PATH)
    experiment_excel = Path(find_single_excel_file(p)) if p.is_dir() else p

    # 2) train fingerprint base (REF_FILE_PATH를 디렉토리로 사용)
    train_fp_base = Path(REF_FILE_PATH)
    if not train_fp_base.exists() or not train_fp_base.is_dir():
        raise NotADirectoryError(f"REF_FILE_PATH가 올바른 디렉토리가 아님: {train_fp_base}")

    # 3) 실험 FP/결과 디렉토리
    fingerprint_dir = Path(FINGERPRINT_DIR)
    results_dir = Path(RESULTS_DIR)

    print("[config] experiment_excel :", experiment_excel)
    print("[config] train_fp_base    :", train_fp_base)
    print("[config] fingerprint_dir  :", fingerprint_dir)
    print("[config] results_dir      :", results_dir)

    fingerprint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return experiment_excel, train_fp_base, fingerprint_dir, results_dir


# =========================
# 본 계산
# =========================
def calc_doa(
    experiment_excel: Path,
    train_fp_base: Path,
    fingerprint_dir: Path,
    results_dir: Path,
) -> Path:
    """
    assay_list에 기재된 모든 assay에 대해 DoA 계산.
    결과: [DTXSID?, SMILES, <assay>_DoA, <assay>_MF, <assay>_Model] 열로 구성된 Excel 저장.
    """
    # 입력 로드
    assay_df = pd.read_excel(experiment_excel, sheet_name="assay_list")
    assay_df = _standardize_columns(assay_df)
    data_df = read_data_with_smiles(experiment_excel, sheet="data")

    # 드랍 인덱스(실험 FP 기준) — 첫 MF 기준으로 추출(필요시 assay별로 별도 적용)
    first_mf = assay_df.iloc[0]["MF"]
    drop_idx_exp_global = _load_dropidx(fingerprint_dir / f"{first_mf}_dropidx.csv")

    # ID/SMILES
    smiles_full = data_df["SMILES"].tolist()
    dtxsid_full = data_df["DTXSID"].tolist() if "DTXSID" in data_df.columns else None

    # dropidx 적용
    def _mask_drop(seq, drop_idx):
        return [v for i, v in enumerate(seq) if i not in drop_idx] if drop_idx else seq

    smiles = _mask_drop(smiles_full, drop_idx_exp_global)
    dtxsid = _mask_drop(dtxsid_full, drop_idx_exp_global) if dtxsid_full is not None else None

    # 결과 테이블 초기화
    results = pd.DataFrame({"SMILES": smiles})
    if dtxsid is not None:
        results.insert(0, "DTXSID", dtxsid)

    # 캐시: 훈련 FP
    train_cache: Dict[str, np.ndarray] = {}

    # 각 assay에 대해 DoA 계산
    for _, row in assay_df.iterrows():
        assay = row["assay_name"]
        mf = row["MF"]
        model_type = row.get("Model", "")

        # --- 훈련 FP 불러오기 (REF_FILE_PATH 디렉토리) ---
        if mf not in train_cache:
            train_cache[mf] = _load_train_fps(mf, train_fp_base)
        train_fps = train_cache[mf]

        # --- 실험 FP 불러오기 (experiments/.../fingerprints/{mf}.csv) ---
        exp_fp_path = fingerprint_dir / f"{mf}.csv"
        if not exp_fp_path.exists():
            raise FileNotFoundError(f"실험 FP 파일이 없습니다: {exp_fp_path}")

        pred_fp = pd.read_csv(exp_fp_path)

        # 실험 측 dropidx (MF별로 있을 수 있으니 우선 동일 규칙 적용)
        drop_idx_exp = _load_dropidx(fingerprint_dir / f"{mf}_dropidx.csv")
        if drop_idx_exp:
            pred_fp = pred_fp.drop(index=drop_idx_exp).reset_index(drop=True)

        # --- 길이 일치 점검 ---
        if len(pred_fp) != len(results):
            raise ValueError(
                f"[길이 불일치] MF={mf}의 실험 FP 행수({len(pred_fp)}) "
                f"!= 결과 테이블 행수({len(results)}). "
                f"data 시트 / dropidx / FP 생성 순서를 확인하세요."
            )

        # --- DoA 계산: 각 실험 행에 대해 훈련 집합과의 Tanimoto 최대값 ---
        doa_vals = [_tanimoto_max(fp.astype(bool), train_fps) for fp in pred_fp.to_numpy()]

        # --- 결과 컬럼 추가 ---
        results[f"{assay}_DoA"] = doa_vals
        results[f"{assay}_MF"] = [mf] * len(doa_vals)
        results[f"{assay}_Model"] = [model_type] * len(doa_vals)

    # 출력
    out_path = results_dir / f"{experiment_excel.stem}_DoA.xlsx"
    results.to_excel(out_path, index=False)
    return out_path


def main() -> None:
    experiment_excel, train_fp_base, fingerprint_dir, results_dir = resolve_inputs()
    out_path = calc_doa(
        experiment_excel=experiment_excel,
        train_fp_base=train_fp_base,
        fingerprint_dir=fingerprint_dir,
        results_dir=results_dir,
    )
    print(f"[done] DoA results saved to: {out_path}")


if __name__ == "__main__":
    main()
