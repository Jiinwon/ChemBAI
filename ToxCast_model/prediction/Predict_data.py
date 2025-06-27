import pandas as pd
import joblib
import os
from pathlib import Path
from datetime import datetime
import json

if __name__ == "__main__":
    # config.py에 정의된 경로 사용
    try:
        from config import (
            PREDICT_LIST_PATH,
            MODEL_SELECTION,
            MODEL_PATH_BASE_0,
            MODEL_PATH_BASE_1,
            PREDICT_FP_PATH,
            PREDICT_SMILES_PATH,
            RESULTS_DIR,
        )
    except ImportError:
        raise ImportError("config.py 파일을 찾을 수 없습니다. 'ToxCast_model' 디렉토리에서 실행해 주세요.")

    input_excel_path = PREDICT_LIST_PATH
    model_path_base = MODEL_PATH_BASE_0 if MODEL_SELECTION == 0 else MODEL_PATH_BASE_1
    input_fp_path_base = PREDICT_FP_PATH
    SMILES_path = PREDICT_SMILES_PATH
    if os.path.isdir(input_excel_path):
        from toxcast_pkg.common import find_single_excel_file
        input_excel_path = find_single_excel_file(input_excel_path)
    if os.path.isdir(SMILES_path):
        from toxcast_pkg.common import find_single_excel_file
        SMILES_path = find_single_excel_file(SMILES_path)

    # 입력 데이터 읽기
    data = pd.read_excel(input_excel_path, sheet_name="assay_list")
    SMILES_df = pd.read_excel(SMILES_path, sheet_name="data")
    SMILES = SMILES_df['SMILES']

    # 필요한 열 추출
    if MODEL_SELECTION == 0:
        required_columns = ["assay_name"]
    else:
        required_columns = ["assay_name", "Model", "MF"]
    if not all(col in data.columns for col in required_columns):
        raise KeyError(f"필요한 열 {required_columns}이(가) 엑셀 파일에 없습니다.")

    # 전체 결과를 저장할 데이터프레임 초기화
    all_results = pd.DataFrame()
    metadata_records = []

    # 반복문으로 각 모델에 대해 처리
    for _, row in data.iterrows():
        assay_name = row["assay_name"]

        if MODEL_SELECTION == 0:
            # locate model automatically from best F1 directory
            pattern = f"{model_path_base}/{assay_name}_*/{assay_name}_best_model_*.joblib"
            matches = list(Path(model_path_base).glob(f"{assay_name}_*/{assay_name}_best_model_*.joblib"))
            if len(matches) != 1:
                print(f"모델 파일을 찾지 못했습니다: {pattern}")
                continue
            model_path = str(matches[0])
            filename = os.path.basename(model_path)
            prefix = f"{assay_name}_best_model_"
            mf_model = filename[len(prefix):-len(".joblib")]
            try:
                mf_type, model_type = mf_model.split("_", 1)
            except ValueError:
                print(f"모델 파일 이름에서 MF와 모델 타입을 파싱할 수 없습니다: {filename}")
                continue
        else:
            model_type = row["Model"]
            mf_type = row["MF"]
            model_path = f"{model_path_base}/{assay_name}_{mf_type}_{model_type}/{assay_name}_best_model_{mf_type}_{model_type}.joblib"
        
        if not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}")
            continue

        # 모델 로드
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        # 입력 데이터 경로 설정
        input_csv_path = f"{input_fp_path_base}/{mf_type}.csv"
        input_drop_csv_path = f"{input_fp_path_base}/{mf_type}_dropidx.csv"

        if not os.path.exists(input_csv_path):
            print(f"입력 데이터 파일이 존재하지 않습니다: {input_csv_path}")
            continue

        # 입력 데이터 로드
        input_data = pd.read_csv(input_csv_path)

        # 예측 수행
        print(f"Performing prediction for assay: {assay_name}...")
        predictions = model.predict(input_data)

        # assay_name별 열에 예측 결과 추가
        if assay_name not in all_results:
            all_results[assay_name] = [None] * len(input_data)

        # 예측 결과 삽입
        all_results[assay_name] = predictions

        metadata_records.append({
            "model": os.path.basename(model_path),
            "ASSAY": assay_name,
            "model_type": model_type,
            "MF": mf_type,
            "prediction_count": int(len(predictions)),
        })

    # dropidx 파일이 존재하고 크기가 0보다 큰지 확인
    if os.path.exists(input_drop_csv_path) and os.stat(input_drop_csv_path).st_size > 0:
        try:
            dropidx_df = pd.read_csv(input_drop_csv_path)
            # 첫 번째 열에 제거할 행 인덱스가 있다고 가정하고 리스트로 변환
            dropidx = dropidx_df.iloc[:, 0].tolist()
        except pd.errors.EmptyDataError:
            print("dropidx 파일이 비어있습니다. 건너뜁니다.")
            dropidx = []
    else:
        print("dropidx 파일이 없거나 비어있습니다. 건너뜁니다.")
        dropidx = []

    # 기존 SMILES 리스트에서 dropidx에 해당하는 인덱스의 항목 제거
    filtered_smiles = [sm for i, sm in enumerate(SMILES) if i not in dropidx]

    # SMILES 열 추가 및 채우기
    all_results.insert(0, "SMILES", filtered_smiles)

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
