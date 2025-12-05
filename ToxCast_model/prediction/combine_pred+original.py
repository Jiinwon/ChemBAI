import pandas as pd
from pathlib import Path

# 경로 설정
ref_path = Path("/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/"
                "ToxCast_model/data/ToxCast_v.4.1_v.2/ToxCast_v4.1_hitcall_v.2.xlsx")

pred_path = Path("/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/"
                 "ToxCast_model/experiments/prediction/DART_MTL/results/"
                 "2025-12-05_11-52-20/DART_Multi+DL_AddData_251204_prediction.xlsx")

out_path = Path("/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/"
                "ToxCast_model/experiments/prediction/DART_MTL/results/"
                "2025-12-05_11-52-20/DART_Multi+DL_AddData_251204_merged.xlsx")

# 1. 엑셀 읽기
ref = pd.read_excel(ref_path)   # 원본 hitcall
pred = pd.read_excel(pred_path) # 예측 결과 (기본 틀)

# 2. DTXSID 컬럼 체크
if "DTXSID" not in ref.columns or "DTXSID" not in pred.columns:
    raise ValueError("두 파일 모두에 'DTXSID' 컬럼이 있어야 합니다.")

# 3. DTXSID 인덱스로
ref_idx = ref.set_index("DTXSID")
pred_idx = pred.set_index("DTXSID")

# 4. 기준은 prediction에 있는 DTXSID들만 사용
#    → 원본 ref를 prediction DTXSID 순서에 맞춰 재배열
ref_aligned = ref_idx.reindex(pred_idx.index)

# 5. 컬럼(assay) 세트 정의
#    기본 틀은 prediction의 컬럼을 그대로 쓰고,
#    그 중에서 ref에도 존재하는 컬럼들만 "원본 우선" 규칙 적용
pred_cols = pred_idx.columns
ref_cols = ref_aligned.columns
common_cols = pred_cols.intersection(ref_cols)

# 6. 병합: 기본 틀은 prediction, common 컬럼에서만 ref 우선 규칙 적용
merged_idx = pred_idx.copy()

# 원본(ref_aligned) 값이 있으면 그 값, 없으면(pred_idx) 값 사용
merged_idx[common_cols] = ref_aligned[common_cols].combine_first(pred_idx[common_cols])

# 7. 다시 DTXSID를 컬럼으로 되돌리고 엑셀로 저장
merged = merged_idx.reset_index()
merged.to_excel(out_path, index=False)

print("완료: 병합된 파일이 저장되었습니다.")
print(out_path)
