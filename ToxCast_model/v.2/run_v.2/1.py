#10.10
#시간 단축을 위해 load_data를 미리 돌려서 얻어놓은 fp파일을 바로 불러와서 사용하는것으로 바꿈
#여러 assay 사용하기 위해서 assay_num 새로운 변수 사용
import sys
import os
from datetime import datetime

# 현재 파일의 디렉토리 경로를 기준으로 상위 두 단계 디렉토리로 이동하여 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import warnings
import joblib
import logging
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from toxcast_pkg.common import ParameterGrid
from rdkit import RDLogger
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def main(fingerprint_type, assay_num, file_path,  fp_path):

    if fingerprint_type == 'MACCS':
        file_path_fp = f'{fp_path}/MACCS.csv'
    if fingerprint_type == 'Layered':
        file_path_fp = f'{fp_path}/Layered.csv'
    if fingerprint_type == 'Morgan':
        file_path_fp = f'{fp_path}/Morgan.csv'
    if fingerprint_type == 'Pattern':
        file_path_fp = f'{fp_path}/Pattern.csv'
    if fingerprint_type == 'RDKit':
        file_path_fp = f'{fp_path}/RDKit.csv'
   
    # fingerprint, hitcall값 x, y로 불러오기
    x = pd.read_csv(file_path_fp)

    dropidx_file = f'{fp_path}/{fingerprint_type}_dropidx.csv'
    if os.path.exists(dropidx_file) and os.path.getsize(dropidx_file) > 0:
        try:
            df_drop_idx = pd.read_csv(dropidx_file)
            drop_idx = df_drop_idx[f'{fingerprint_type}'].tolist() if not df_drop_idx.empty else []
        except pd.errors.EmptyDataError:
            drop_idx = []
    else:
        drop_idx = []

    # 이후 drop_idx가 빈 리스트일 경우 drop 작업이 수행되지 않습니다.
    x = pd.read_csv(file_path_fp)
    df = pd.read_excel(file_path)
    assay_name = df.columns[assay_num+1]
    if drop_idx:
        y = df.iloc[:, assay_num+1].drop(drop_idx).reset_index(drop=True)
    else:
        y = df.iloc[:, assay_num+1].reset_index(drop=True)

    na_idx = y[y.isnull()].index
    y = y.drop(index=na_idx).reset_index(drop=True)
    x = x.drop(index=na_idx).reset_index(drop=True)

############################################################################04.04 위 수정 x

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True) #04.04 randomstate=42 제거
    # 저장 디렉토리 설정
    # 오늘 날짜 가져오기
    today = datetime.today().strftime("%Y%m%d")
    save_dir = f'./v.2/data_v.2/train_test_split/{today}'
    os.makedirs(save_dir, exist_ok=True)


    # iteration number 설정 (예시로 i=1)
    i = 1
    # 파일 이름 포맷
    prefix = f"{i}"

    # 각각 저장
    x_train.to_csv(os.path.join(save_dir, f"{prefix}_x_train.csv"), index=False)
    x_test.to_csv(os.path.join(save_dir, f"{prefix}_x_test.csv"), index=False)
    y_train.to_csv(os.path.join(save_dir, f"{prefix}_y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_dir, f"{prefix}_y_test.csv"), index=False)

    print("train_test_split 파일 저장 완료.")

if __name__ == "__main__":
    main(
        fingerprint_type='MACCS',
        fp_path='./data/ToxCast_v.4.1_v.2/fingerprints',
        assay_num = 1,
        file_path = "/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI_ToxCast/ToxCast_model/data/ToxCast_v.4.1_v.2/ToxCast_v.4.1_v.2.xlsx"
    )