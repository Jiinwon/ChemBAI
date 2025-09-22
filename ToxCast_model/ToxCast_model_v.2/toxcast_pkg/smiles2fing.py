import os
import pandas as pd
import numpy as np

from toxcast_pkg.common import read_data_with_smiles

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
    RDLogger.DisableLog('rdApp.*')
except Exception:
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
    # config.py에 정의된 경로 사용
    try:
        from config import SMILES_INPUT_PATH, FINGERPRINT_DIR
    except ImportError:
        raise ImportError("config.py 파일을 찾을 수 없습니다. 'ToxCast_model' 디렉토리에서 실행해 주세요.")

    input_excel_path = SMILES_INPUT_PATH  # 입력 엑셀 파일 경로 또는 디렉토리
    if os.path.isdir(input_excel_path):
        from toxcast_pkg.common import find_single_excel_file
        input_excel_path = find_single_excel_file(input_excel_path)
    output_dir = FINGERPRINT_DIR  # fingerprints를 저장할 디렉토리

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)


    # 엑셀 파일 읽기 - "data" 시트에서 SMILES 자동 탐지
    df = read_data_with_smiles(input_excel_path, sheet="data")
    smiles = df["SMILES"].astype(str)


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