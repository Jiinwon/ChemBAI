import json
import sys
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook

project_dir = Path(sys.argv[1])
input_excel = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else project_dir / "training_input_template.xlsx"
)
metadata_file = (
    Path(sys.argv[3]) if len(sys.argv) > 3 else project_dir / "metadata.json"
)
results_excel = project_dir / "results" / "training_results_template.xlsx"

# load metadata
with open(metadata_file, "r", encoding="utf-8") as f:
    records = json.load(f)
if not records:
    sys.exit()

# group by assay
by_assay = {}
for r in records:
    by_assay.setdefault(r["assay_name"], []).append(r)

# compute training stats using pandas
# first row holds AEID numbers, second row holds assay names
aeid_row = pd.read_excel(input_excel, sheet_name="data", nrows=1, header=None).iloc[
    0, 1:
]
train_df = pd.read_excel(input_excel, sheet_name="data", header=1)
aeid_map = dict(zip(train_df.columns, aeid_row))

# open workbook for update
wb = load_workbook(results_excel)
f1_ws = wb["F1"]
auc_ws = wb["AUC"]

# ensure AEID column exists in results template
if f1_ws["B1"].value != "AEID":
    f1_ws.insert_cols(2)
    f1_ws["B1"] = "AEID"
if auc_ws["B1"].value != "AEID":
    auc_ws.insert_cols(2)
    auc_ws["B1"] = "AEID"

for assay in by_assay:
    series = train_df[assay].dropna()
    train_count = len(series)
    pos_ratio = series.mean() * 100 if len(series) > 0 else 0
    best_f1 = max(by_assay[assay], key=lambda r: r.get("F1", 0))
    best_auc = max(by_assay[assay], key=lambda r: r.get("AUC", 0))
    aeid = aeid_map.get(assay, "")
    row_f1 = [
        assay,
        aeid,
        train_count,
        f"{pos_ratio:.2f}",
        best_f1["MF"],
        best_f1["Model"],
        best_f1.get("F1"),
        best_f1.get("Precision"),
        best_f1.get("Recall"),
        best_f1.get("AUC"),
        best_f1.get("Accuracy"),
        best_f1.get("valF1"),
        "",
        "",
        best_f1.get("valAUC"),
        "",
    ]
    f1_ws.append(row_f1)
    row_auc = [
        assay,
        aeid,
        train_count,
        f"{pos_ratio:.2f}",
        best_auc["MF"],
        best_auc["Model"],
        best_auc.get("F1"),
        best_auc.get("Precision"),
        best_auc.get("Recall"),
        best_auc.get("AUC"),
        best_auc.get("Accuracy"),
        best_auc.get("valF1"),
        "",
        "",
        best_auc.get("valAUC"),
        "",
    ]
    auc_ws.append(row_auc)

wb.save(results_excel)
