import os
import pandas as pd
import numpy as np

try: 
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
    
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint


def Smiles2Fing(smiles, fingerprint_type='MACCS'):
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] is None]
    
    ms = list(filter(None, ms_tmp))
    
    if fingerprint_type == 'MACCS':
        fingerprints = [np.array(MACCSkeys.GenMACCSKeys(i), dtype=int) for i in ms]
    elif fingerprint_type == 'Morgan':
        fingerprints = [np.array(AllChem.GetMorganFingerprintAsBitVect(i, 2, nBits=1024), dtype=int) for i in ms]
    elif fingerprint_type == 'RDKit':
        fingerprints = [np.array(RDKFingerprint(i), dtype=int) for i in ms]
    elif fingerprint_type == 'Layered':
        fingerprints = [np.array(AllChem.LayeredFingerprint(i), dtype=int) for i in ms]
    elif fingerprint_type == 'Pattern':
        fingerprints = [np.array(AllChem.PatternFingerprint(i), dtype=int) for i in ms]
    else:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
    
    fingerprints_df = pd.DataFrame(fingerprints)
    
    # 컬럼명 생성 (예: maccs_1, maccs_2, ..., maccs_n)
    colname = [f'{fingerprint_type.lower()}_{i+1}' for i in range(fingerprints_df.shape[1])]
    fingerprints_df.columns = colname
    fingerprints_df = fingerprints_df.reset_index(drop=True)
    
    return ms_none_idx, fingerprints_df


if __name__ == "__main__":
    # 입력 파일 경로 설정
    input_excel_path = "./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx"  # 입력 엑셀 파일 경로 : 훈련 or 예측에 사용할 데이터
    output_dir = "./data/example_data_DBPs/for_train/fingerprints"  # DBPs/for_predict/example_DBPs 디렉토리

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 엑셀 파일 읽기
    data = pd.read_excel(input_excel_path)

    # SMILES 열 추출
    if 'SMILES' not in data.columns:
        raise KeyError("엑셀 파일에 'SMILES' 열이 없습니다.")
    smiles = data['SMILES']

    # Fingerprint 유형 리스트
    fps = ['MACCS', 'Morgan', 'RDKit', 'Layered', 'Pattern']

    # 각 fingerprint_type에 대해 처리 및 저장
    for fingerprint_type in fps:
        print(f"Processing {fingerprint_type} fingerprints...")
        ms_none_idx, fingerprints_df = Smiles2Fing(smiles, fingerprint_type)

        # None 값이 있는 SMILES 처리 (필요에 따라 로그 저장 가능)
        if ms_none_idx:
            print(f"Warning: {len(ms_none_idx)}개의 SMILES이 None 처리되었습니다.")

        # Fingerprint 결과 저장 (CSV 파일)
        output_csv_path = os.path.join(output_dir, f"{fingerprint_type}.csv")
        fingerprints_df.to_csv(output_csv_path, index=False)
        print(f"Saved {fingerprint_type} fingerprints to {output_csv_path}")
        
        # ms_none_idx 저장 (drop된 index 정보를 CSV 파일로 저장)
        dropidx_df = pd.DataFrame(ms_none_idx)
        dropidx_csv_path = os.path.join(output_dir, f"{fingerprint_type}_dropidx.csv")
        dropidx_df.to_csv(dropidx_csv_path, index=False)
        print(f"Saved {fingerprint_type} drop indices to {dropidx_csv_path}")

    print("모든 fingerprint 유형에 대한 처리가 완료되었습니다.")