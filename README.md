# ChemBAI
Data version : ToxCast_v.4.1
==========================


사용된 분자지문
-------------
- MACCS
- Morgan
- RDKit
- Pattern
- Layered
(총 다섯 가지)

사용된 알고리즘
-------------
- Decision Tree (dt.py)
- Logistic Regression (logistic.py)
- Gradient Boost Tree (gbt.py)
- XGBoost (xgb.py)
- Random Forest (rf.py)
(총 다섯 가지)

분자지문
----------------
모델 훈련 시마다 분자지문으로 변환하는 과정이 중복 실행되지 않도록, 최초에 한 번만 분자지문으로 변환하여 저장한 파일을 불러와 사용합니다.

데이터 전처리
-------------
ToxCast_v.4.1_v.2 데이터는 KNIME을 통한 염 제거와 무기물질 제거, 그리고 Hitcall의 개수가 5개 미만인 데이터를 제거한 결과물입니다.

데이터 처리 과정 (예시)
------------------------
run 디렉토리에 있는 scikit-learn 기반 머신러닝 코드에서는 아래와 같이 데이터를 처리합니다.

  1. SMILES로부터 분자지문(mol 형식)으로 변환되지 않는 화학물질의 인덱스는 dropidx.csv 파일에 저장합니다.
     (예시 코드에서 사용하는 변수: drop_idx)
     
  2. Hitcall 데이터에서 결측값이 있는 인덱스는 na_idx 변수로 받아 해당 행을 제거합니다.

예시 코드 (dt.py, 70~77라인)
----------------------------
    x = pd.read_csv(file_path_fp)
    df_drop_idx = pd.read_csv(f'{fp_path}/{fingerprint_type}_dropidx.csv')
    drop_idx = df_drop_idx[f'{fingerprint_type}'].tolist()
    df = pd.read_excel(file_path)
    y = df.iloc[:, assay_num+1].drop(drop_idx).reset_index(drop=True)
    na_idx = y[y.isnull()].index
    y = y.drop(index=na_idx).reset_index(drop=True)
    x = x.drop(index=na_idx).reset_index(drop=True)

참고 사항
----------
모델 코드 내에 사용된 assay_num 변수는 ToxCast 내 다양한 assay 데이터를 반복문으로 처리하기 위해 설정된 변수입니다.