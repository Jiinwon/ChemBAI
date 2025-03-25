import pandas as pd
import joblib
import os

if __name__ == "__main__":
    # 입력 엑셀 파일과 시트명
    input_excel_path = "./prediction/example_prediction_DBPs/example_assay_list_ER.xlsx" #수정
    model_path_base = "./results/example_DBPs_ER/2025-03-25/model_save_path"
    input_fp_path_base = "./data/example_data_DBPs/for_predict/fingerprints"
    SMILES_path = "./data/example_data_DBPs/for_predict/example_DBPs_for_pred.xlsx"

    # 입력 데이터 읽기
    data = pd.read_excel(input_excel_path)
    SMILES_df = pd.read_excel(SMILES_path)
    SMILES = SMILES_df['SMILES']

    # 필요한 열 추출
    required_columns = ["assay_name", "Model", "MF"]
    if not all(col in data.columns for col in required_columns):
        raise KeyError(f"필요한 열 {required_columns}이(가) 엑셀 파일에 없습니다.")

    # 전체 결과를 저장할 데이터프레임 초기화
    all_results = pd.DataFrame()

    # 반복문으로 각 모델에 대해 처리
    for _, row in data.iterrows():
        assay_name = row["assay_name"]
        model_type = row["Model"]
        mf_type = row["MF"]
        print(assay_name, model_type, mf_type)

        model_path = f"{model_path_base}/{assay_name}/{assay_name}_{mf_type}_{model_type}/{assay_name}_best_model_{mf_type}_{model_type}.joblib"

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

    # 최종 결과 저장
    output_excel_path = "./prediction/example_prediction_DBPs/example_predict_DBPs_ER.xlsx" #수정
    all_results.to_excel(output_excel_path, index=False)
    print(f"All predictions saved to {output_excel_path}")

    print("모든 예측 작업이 완료되었습니다.")
