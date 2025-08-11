python - <<'PY'
import pandas as pd, re
from pathlib import Path

xlsx = Path("ToxCast_model/experiments/prediction/NGRA/NGRA_input_template.xlsx")

def cols(header):
    df = pd.read_excel(xlsx, sheet_name="data", header=header, nrows=1)
    # 컬럼의 숨은 문자/공백을 그대로 보이게
    print(f"\n[header={header}] raw columns repr:")
    print([repr(c) for c in df.columns])

for h in (1, 0):
    cols(h)

# 'SMILES'가 들어있는 열 후보를 셀 내부 검색으로도 확인
df_any = pd.read_excel(xlsx, sheet_name="data", header=None)
hits = []
for j in range(df_any.shape[1]):
    col = df_any.iloc[:, j].astype(str).str.strip().str.replace("\u200b", "").str.replace("\ufeff", "")
    if col.str.fullmatch(r"(?i)\s*smiles\s*").any():  # 대소문자 무시
        hits.append(j)
print("\nheader=None에서 'SMILES' 텍스트가 발견된 컬럼 인덱스:", hits)
PY
