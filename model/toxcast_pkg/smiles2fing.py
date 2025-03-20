import pickle
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
    
    # 컬럼명 생성 (예: maccs_1, maccs_2, ..., maccs_167)
    colname = [f'{fingerprint_type.lower()}_{i+1}' for i in range(fingerprints_df.shape[1])]
    fingerprints_df.columns = colname
    fingerprints_df = fingerprints_df.reset_index(drop=True)
    
    return ms_none_idx, fingerprints_df


if __name__ == '__main__':
    file_path = "/home1/won0316/_RESEARCH/0817_Genotoxicity/tg471/241213_new_data/tc_241213.xlsx"
    df = pd.read_excel(file_path)
    fps=['MACCS', 'Morgan', 'RDKit', 'Layered']

    # Molecular Fingerprints 생성
    for fingerprint_type in fps:
        drop_idx, fingerprints = Smiles2Fing(df.SMILES, fingerprint_type=fingerprint_type)

    # 계산한 결과를 파일로 저장
    # 저장 경로 설정
    save_path = '/home1/won0316/_RESEARCH/0817_Genotoxicity/tg471/data/FPS_pickle'
    
    # 파일 경로 생성
    file_path = os.path.join(save_path, f'{fingerprint_type}.pkl')
    
    # 결과 저장
    with open(file_path, 'wb') as f:
        pickle.dump({'drop_idx': drop_idx, 'fingerprints': fingerprints}, f)

    print("작업이 완료되었습니다.")
