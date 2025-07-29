import pandas as pd
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# 1. 모델 로드 (필요 시; 이 예제에서는 fingerprint 계산에 직접 쓰진 않음)
model_path = '/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/Final_model_save/ToxCast_model(F1)/ATG_PPARa_TRANS_Morgan_xgb/ATG_PPARa_TRANS_best_model_Morgan_xgb.joblib'
model = load(model_path)

# 2. Excel 파일 읽기
excel_path = '/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model/experiments/XAI/XAI_pd_form.xlsx'
df = pd.read_excel(excel_path, engine='openpyxl')

# 3. 무작위 10개 샘플 추출 (DTXSID, SMILES 컬럼이 존재한다고 가정)
df_sample = df[['DTXSID', 'SMILES']].dropna().sample(n=10, random_state=123).reset_index(drop=True)

# 4. Morgan fingerprint 계산 (radius=2, nBits=2048)
fps = []
for smi in df_sample['SMILES']:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fps.append(fp)

# 5. pairwise Tanimoto similarity 계산
n = len(fps)
sim_matrix = []
for i in range(n):
    row = []
    for j in range(n):
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        row.append(sim)
    sim_matrix.append(row)

# 6. 결과를 DataFrame으로 정리
sim_df = pd.DataFrame(sim_matrix,
                      index=df_sample['DTXSID'],
                      columns=df_sample['DTXSID'])

# 7. 출력
print("샘플 DTXSID 및 SMILES:")
print(df_sample.to_string(index=False))
print("\nPairwise Tanimoto Similarity Matrix:")
print(sim_df)