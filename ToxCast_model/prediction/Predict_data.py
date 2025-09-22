import argparse
import pandas as pd
import joblib
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np

from toxcast_pkg.common import read_data_with_smiles, _standardize_columns


def _tanimoto_max(test_fp: np.ndarray, train_fps: np.ndarray) -> float:
    """Return max Tanimoto similarity of test_fp against train_fps."""
    inter = np.logical_and(train_fps, test_fp).sum(axis=1)
    union = np.logical_or(train_fps, test_fp).sum(axis=1)
    # avoid divide by zero
    sim = inter / np.where(union == 0, 1, union)
    return float(sim.max())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prediction step")
    parser.add_argument("--skip-doa", action="store_true", help="Skip DoA calculation")
    args = parser.parse_args()
    skip_doa = args.skip_doa

    # config.py에 정의된 경로 사용
    try:
        from config import (
            PREDICT_LIST_PATH,
            MODEL_SELECTION,
            MODEL_PATH_BASE_0,
            MODEL_PATH_BASE_1,
            get_model_base,

            PREDICT_FP_PATH,
            PREDICT_SMILES_PATH,
            RESULTS_DIR,
            REF_FILE_PATH,
        )
    except ImportError:
        raise ImportError("config.py 파일을 찾을 수 없습니다. 'ToxCast_model' 디렉토리에서 실행해 주세요.")

    input_path = Path(PREDICT_LIST_PATH)

    model_path_base = get_model_base()
    input_fp_path_base = Path(PREDICT_FP_PATH)
    SMILES_path = Path(PREDICT_SMILES_PATH)
    if input_path.is_dir():
        from toxcast_pkg.common import find_single_excel_file
        input_path = Path(find_single_excel_file(input_path))
    if SMILES_path.is_dir():
        from toxcast_pkg.common import find_single_excel_file
        SMILES_path = Path(find_single_excel_file(SMILES_path))

    df_data = read_data_with_smiles(SMILES_path, sheet="data")

    if input_path.suffix.lower() in {".csv", ".tsv"}:
        if MODEL_SELECTION == 1:
            config_candidate = input_path.parent / "assay_list.csv"
            if config_candidate.exists():
                data = pd.read_csv(config_candidate)
                data = _standardize_columns(data)
            else:
                raise FileNotFoundError(
                    "MODEL_SELECTION=1 requires an assay_list.csv file when using CSV inputs."
                )
        else:
            skip_cols = {"DTXSID", "SMILES"}
            assay_names = [col for col in df_data.columns if col not in skip_cols]
            data = pd.DataFrame({"assay_name": assay_names})
    else:
        data = pd.read_excel(input_path, sheet_name="assay_list")
        data = _standardize_columns(data)
    SMILES = df_data["SMILES"].astype(str)
    has_dtxsid = "DTXSID" in df_data.columns


    # 필요한 열 추출
    if MODEL_SELECTION == 0:
        required_columns = ["assay_name"]
    else:
        required_columns = ["assay_name", "Model", "MF"]
    if not all(col in data.columns for col in required_columns):
        raise KeyError(f"필요한 열 {required_columns}이(가) 설정 파일에 없습니다.")

    # 전체 결과를 저장할 데이터프레임 초기화
    all_results = pd.DataFrame()
    metadata_records = []
    # cache training fingerprints for DoA calculation
    train_fp_base = Path(REF_FILE_PATH)
    train_fp_cache = {}
    first_mf_type = None

    # 반복문으로 각 모델에 대해 처리
    for _, row in data.iterrows():
        assay_name = row["assay_name"]

        if MODEL_SELECTION == 0:
            # locate model automatically from best F1 directory
            pattern = f"{model_path_base}/{assay_name}_*/{assay_name}_best_model_*.joblib"
            matches = list(Path(model_path_base).glob(f"{assay_name}_*/{assay_name}_best_model_*.joblib"))
            if len(matches) != 1:
                msg = f"모델 파일을 찾지 못했습니다: {pattern}"
                print(msg)
                metadata_records.append({"ASSAY": assay_name, "error": msg})

                continue
            model_path = str(matches[0])
            filename = os.path.basename(model_path)
            prefix = f"{assay_name}_best_model_"
            mf_model = filename[len(prefix):-len(".joblib")]
            try:
                mf_type, model_type = mf_model.split("_", 1)
            except ValueError:
                msg = f"모델 파일 이름에서 MF와 모델 타입을 파싱할 수 없습니다: {filename}"
                print(msg)
                metadata_records.append({"ASSAY": assay_name, "error": msg})

                continue
        else:
            model_type = row["Model"]
            mf_type = row["MF"]
            model_path = f"{model_path_base}/{assay_name}_{mf_type}_{model_type}/{assay_name}_best_model_{mf_type}_{model_type}.joblib"
        
        if not os.path.exists(model_path):
            msg = f"모델 파일이 존재하지 않습니다: {model_path}"
            print(msg)
            metadata_records.append({"ASSAY": assay_name, "error": msg})
            continue

        # 모델 로드
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        if first_mf_type is None:
            first_mf_type = mf_type

        # 입력 데이터 경로 설정
        input_csv_path = input_fp_path_base / f"{mf_type}.csv"
        input_drop_csv_path = input_fp_path_base / f"{mf_type}_dropidx.csv"

        if not input_csv_path.exists():
            raise FileNotFoundError(f"실험 FP 파일이 없습니다: {input_csv_path}")

        # 입력 데이터 로드
        input_data = pd.read_csv(input_csv_path)

        # 예측 수행
        print(f"Performing prediction for assay: {assay_name}...")
        predictions = model.predict(input_data)

        if not skip_doa:
            # DoA 계산을 위한 학습 fingerprint 로드
            if mf_type not in train_fp_cache:
                train_fp_path = train_fp_base / f"{mf_type}.csv"
                dropidx_path = train_fp_base / f"{mf_type}_dropidx.csv"
                if not train_fp_path.exists():
                    raise FileNotFoundError(f"훈련 fingerprint 파일이 없습니다: {train_fp_path}")
                train_df = pd.read_csv(train_fp_path)
                if dropidx_path.exists() and dropidx_path.stat().st_size > 0:
                    try:
                        drop_idx = pd.read_csv(dropidx_path).iloc[:, 0].tolist()
                    except pd.errors.EmptyDataError:
                        drop_idx = []
                    if drop_idx:
                        train_df = train_df.drop(index=drop_idx).reset_index(drop=True)
                train_fp_cache[mf_type] = train_df.astype(bool).values

            train_fps = train_fp_cache[mf_type]
            doa_values = [
                _tanimoto_max(fp.astype(bool), train_fps)
                for fp in input_data.to_numpy()
            ]
        else:
            doa_values = None

        # assay_name별 열에 예측 결과 추가
        doa_col = f"{assay_name}_DoA"
        if assay_name not in all_results:
            all_results[assay_name] = predictions
            if not skip_doa:
                insert_idx = all_results.columns.get_loc(assay_name)
                all_results.insert(insert_idx + 1, doa_col, doa_values)
        else:
            all_results[assay_name] = predictions
            if not skip_doa:
                if doa_col not in all_results:
                    idx = all_results.columns.get_loc(assay_name)
                    all_results.insert(idx + 1, doa_col, doa_values)
                else:
                    all_results[doa_col] = doa_values

        metadata_records.append({
            "model": os.path.basename(model_path),
            "ASSAY": assay_name,
            "model_type": model_type,
            "MF": mf_type,
            "prediction_count": int(len(predictions)),
        })

    # dropidx 파일은 첫 번째 MF 기준으로만 적용
    if first_mf_type is None:
        dropidx = []
    else:
        drop_path = input_fp_path_base / f"{first_mf_type}_dropidx.csv"
        if drop_path.exists() and drop_path.stat().st_size > 0:
            try:
                dropidx = pd.read_csv(drop_path).iloc[:, 0].tolist()
            except pd.errors.EmptyDataError:
                dropidx = []
        else:
            dropidx = []

    # 기존 SMILES 리스트에서 dropidx에 해당하는 인덱스의 항목 제거
    filtered_smiles = [sm for i, sm in enumerate(SMILES) if i not in dropidx]
    if has_dtxsid:
        filtered_dtxsid = [sid for i, sid in enumerate(df_data["DTXSID"]) if i not in dropidx]
        all_results.insert(0, "DTXSID", filtered_dtxsid)

    # SMILES 열 추가 및 채우기
    all_results.insert(1, "SMILES", filtered_smiles)

    # replace prediction outputs so 0 -> 2, 1 -> 3
    assay_names = data["assay_name"].tolist()
    for col in assay_names:
        if col in all_results.columns:
            all_results[col] = all_results[col].replace({0: 2, 1: 3})

    # 최종 결과 저장 - create timestamped file under the experiment results dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_excel_path = output_dir / f"{Path(SMILES_path).stem}_prediction.xlsx"
    all_results.to_excel(output_excel_path, index=False)
    print(f"All predictions saved to {output_excel_path}")

    # 메타데이터 저장
    metadata_path = RESULTS_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to {metadata_path}")

    print("모든 예측 작업이 완료되었습니다.")
