import pandas as pd
import joblib
import os

def parse_model_info(model_path):
    """모델 경로에서 assay_name, MF, Algorithm을 추출"""
    filename = os.path.basename(model_path).replace(".joblib", "")
    assay_name, rest = filename.split("_best_model_")
    mf_type, algorithm = rest.rsplit("_", 1)
    return assay_name, mf_type, algorithm

if __name__ == "__main__":
    # SMILES 파일 위치 및 fingerprint 기본 경로
    SMILES_path = "/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/prediction/250513/prediction.xlsx"
    input_fp_path_base = "/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/data/250513/fingerprints"

    # 사용할 모델들 (파일 경로만 명시)
    model_paths = [
        "/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/prediction/250513/Zebrafish_Reproduction_best_model_Layered_rf.joblib",
        "/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/prediction/250513/Japanese_medaka_Reproduction_best_model_MACCS_gbt.joblib",
        "/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/prediction/250513/Fathead_minnow_Reproduction_best_model_MACCS_logistic.joblib"
    ]

    # SMILES 불러오기
    SMILES_df = pd.read_excel(SMILES_path)
    SMILES = SMILES_df["SMILES"]

    # 전체 결과 초기화
    all_results = pd.DataFrame()

    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
            continue

        # assay_name, mf_type, algorithm 추출
        assay_name, mf_type, algorithm = parse_model_info(model_path)
        print(f"🔎 모델 정보 추출 → assay: {assay_name}, MF: {mf_type}, Algo: {algorithm}")

        # fingerprint 경로 구성
        input_csv_path = f"{input_fp_path_base}/{mf_type}.csv"
        input_drop_csv_path = f"{input_fp_path_base}/{mf_type}_dropidx.csv"

        if not os.path.exists(input_csv_path):
            print(f"❌ fingerprint 파일이 존재하지 않습니다: {input_csv_path}")
            continue

        # 입력 데이터 로드
        input_data = pd.read_csv(input_csv_path)

        # dropidx 불러오기
        dropidx = []
        if os.path.exists(input_drop_csv_path) and os.stat(input_drop_csv_path).st_size > 0:
            try:
                dropidx_df = pd.read_csv(input_drop_csv_path)
                dropidx = dropidx_df.iloc[:, 0].tolist()
            except pd.errors.EmptyDataError:
                print(f"⚠️ {input_drop_csv_path} 비어 있음. 건너뜀.")
        else:
            print(f"ℹ️ dropidx 없음 또는 비어 있음: {input_drop_csv_path}")

        # drop된 SMILES 생성
        filtered_smiles = [sm for i, sm in enumerate(SMILES) if i not in dropidx]

        # 모델 로드 및 예측 수행
        print(f"📦 모델 로딩 중: {model_path}")
        model = joblib.load(model_path)

        print(f"🚀 예측 수행 중: {assay_name}")
        predictions = model.predict(input_data)

        # 결과 병합
        if "SMILES" not in all_results.columns:
            all_results["SMILES"] = filtered_smiles
        all_results[assay_name] = predictions

    # 최종 결과 저장
    all_results.to_excel(SMILES_path, index=False)
    print(f"\n✅ 예측 결과 저장 완료: {SMILES_path}")