import pandas as pd
import os
import shutil

# 엑셀 파일 경로 설정
excel_file = '/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/Final_model_save/ToxCast_model_dir.xlsx'
df = pd.read_excel(excel_file)

# 복사될 대상 폴더 베이스 경로 설정
dest_base = '/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/Final_model_save/ToxCast_v.4.2_model_total'
# 모델이 저장된 원본 폴더 베이스 경로 설정
source_base = '/home1/won0316/_RESEARCH/0817_Genotoxicity/tg471/results'

# 사용할 MF, Algorithm 리스트
MF_list = ['MACCS', 'Pattern', 'Layered', 'RDKit', 'Morgan']
Algorithm_list = ['dt', 'xgb', 'gbt', 'logistic', 'rf']

for index, row in df.iterrows():
    assay_name = row['assay_name']
    dir_num = row['dir']
    num = row['num']

    for MF in MF_list:
        for Algorithm in Algorithm_list:
            source_folder = os.path.join(source_base, str(dir_num), 'model_save_path', str(num), f"{num}_{MF}_{Algorithm}")
            dest_folder_name = f"{assay_name}_{MF}_{Algorithm}"
            dest_folder = os.path.join(dest_base, dest_folder_name)

            if not os.path.exists(source_folder):
                print(f"[건너뜀] 원본 폴더 없음: {source_folder}")
                continue

            try:
                os.makedirs(dest_folder, exist_ok=True)

                # dir_num이 1118인 경우 joblib 파일명 조정
                joblib_num = int(num) - 10000 if int(dir_num) == 1118 else int(num)

                # 복사 대상 파일 경로 구성
                source_joblib = os.path.join(source_folder, f"{joblib_num}_best_model_{MF}_{Algorithm}.joblib")
                dest_joblib = os.path.join(dest_folder, f"{assay_name}_best_model_{MF}_{Algorithm}.joblib")

                source_json = os.path.join(source_folder, f"{Algorithm}_intermediate_{MF}.json")
                dest_json = os.path.join(dest_folder, f"{assay_name}_best_model_{MF}_{Algorithm}.json")

                if os.path.exists(source_joblib):
                    shutil.copy2(source_joblib, dest_joblib)
                else:
                    print(f"[joblib 누락] {source_joblib}")

                if os.path.exists(source_json):
                    shutil.copy2(source_json, dest_json)
                else:
                    print(f"[json 누락] {source_json}")

            except Exception as e:
                print(f"[에러] {source_folder} -> {dest_folder} 복사 중 오류 발생: {e}")