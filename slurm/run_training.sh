#!/bin/bash

# Convenience script to train models for a project directory
set -e

# Load required modules for GPU execution
module purge
module load cuda/12.1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate toxcast_env

# Resolve script directory both when run directly and within a Slurm job
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    script_dir="$SLURM_SUBMIT_DIR"
else
    script_dir="$(cd "$(dirname "$0")" && pwd)"
fi

# Determine model directory based on config.VERSION
version=$(python - "$script_dir" <<'PY'
import sys, pathlib
script_dir = pathlib.Path(sys.argv[1])
sys.path.append(str(script_dir.parent / "ToxCast_model"))
import config
print(getattr(config, "VERSION", 1))
PY
)

if [ "$version" = "2" ]; then
    model_dir="$(cd "$script_dir/../ToxCast_model/ToxCast_model_v.2" && pwd)"
    run_subdir="run_v.2"
else
    model_dir="$(cd "$script_dir/../ToxCast_model" && pwd)"
    run_subdir="run"
fi

# Determine default project directory and log location
default_project_dir=$(MODEL_DIR="$model_dir" PYTHONPATH="$model_dir" python - <<'PY'
import os, config
from pathlib import Path
print(Path(os.environ['MODEL_DIR']) / config.BASE_DIR)
PY
)

project_dir="${1:-$default_project_dir}"
project_name="$(basename "$project_dir")"
slurm_out="$project_dir/${project_name}_training.out"
mkdir -p "$project_dir"

# Submit via Slurm if not already launched
if [ -z "$SLURM_LAUNCHED" ]; then
    PARTITIONS=(gpu1 gpu2 gpu3 gpu4 gpu5 gpu6)
    GRES="gpu"
    CPUS_PER_TASK=8
    MEM_PER_TASK="16G"

    if [ -z "$1" ]; then
        echo "Using project directory from config: $project_dir"
    fi

    for p in "${PARTITIONS[@]}"; do
        JOBID=$(sbatch --parsable --partition="$p" --gres="$GRES" \
            --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
            --job-name="${project_name}_training" --output="$slurm_out" \
            --wrap="SLURM_LAUNCHED=1 SLURM_SUBMIT_DIR=\"$PWD\" bash \"$script_dir/run_training.sh\" \"$project_dir\"")
        sleep 2
        info=$(squeue -j "$JOBID" -h -o '%T %R')
        state=$(echo "$info" | awk '{print $1}')
        if [ "$state" != "PD" ]; then
            echo "Job $JOBID running on $(echo "$info" | awk '{print $2}')"
            exit 0
        fi
        scancel "$JOBID"
        echo "Partition $p busy, trying next..."
        sleep 10
    done

    LAST_PART=${PARTITIONS[$(( ${#PARTITIONS[@]} - 1 ))]}
    sbatch --partition="$LAST_PART" --gres="$GRES" \
        --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
        --job-name="${project_name}_training" --output="$slurm_out" \
        --wrap="SLURM_LAUNCHED=1 SLURM_SUBMIT_DIR=\"$PWD\" bash \"$script_dir/run_training.sh\" \"$project_dir\""
    exit 0
fi

# Prepare directories
input_excel=("$project_dir"/*.xlsx)
results_dir="$project_dir/results"
logs_dir="$project_dir/logs"
fp_dir="$project_dir/fingerprints"
metadata_file="$project_dir/metadata.json"
mkdir -p "$results_dir" "$logs_dir" "$fp_dir"
: > "$metadata_file"

# Generate fingerprints only if missing
if [ -z "$(ls -A "$fp_dir" 2>/dev/null)" ]; then
    PYTHONPATH="$model_dir" python -m toxcast_pkg.smiles2fing
fi

# Extract assay names from Excel
mapfile -t assays < <(python - "$input_excel" <<'PY'
import sys, zipfile, xml.etree.ElementTree as ET
xlsx=sys.argv[1]
with zipfile.ZipFile(xlsx) as z:
    shared=z.read('xl/sharedStrings.xml').decode()
    sheet=z.read('xl/worksheets/sheet1.xml').decode()
ss=ET.fromstring(shared)
strings=[s.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t').text for s in ss]
ns={'a':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
root=ET.fromstring(sheet)
row=root.find('.//a:row[@r="2"]', ns)
vals=[]
for c in row.findall('a:c', ns):
    v=c.find('a:v', ns)
    val = '' if v is None else v.text
    if c.get('t')=='s':
        val=strings[int(val)]
    vals.append(val)
print('\n'.join(vals[2:]))
PY
)

# Define fingerprints and models
fingerprints=(MACCS Morgan Layered Pattern RDKit)
models=(rf logistic xgb gbt dt)

assay_index=0
for assay in "${assays[@]}"; do
    for fp in "${fingerprints[@]}"; do
        for model in "${models[@]}"; do
            save_dir="$results_dir/${assay}_${fp}_${model}"
            mkdir -p "$save_dir"
            log_file="$logs_dir/${assay}/${assay}_${fp}_${model}.log"
            mkdir -p "$(dirname "$log_file")"   # <<< fix: ensure assay log folder exists
            echo "[$(date '+%F %T')] START ${assay}_${fp}_${model}" >> "$slurm_out"
            start_time=$(date +%s)
            set +e
            PYTHONPATH="$model_dir" python "$model_dir/$run_subdir/${model}.py" \
                --fingerprint_type "$fp" \
                --file_path "$input_excel" \
                --model_save_path "$save_dir" \
                --assay_num "$assay_index" \
                --fp_path "$fp_dir" \
                >"$log_file" 2>&1
            exit_code=$?
            set -e
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo "[$(date '+%F %T')] END ${assay}_${fp}_${model}" >> "$slurm_out"

            error_msg=""
            if [ $exit_code -ne 0 ]; then
                error_msg=$(grep -m1 'ValueError' "$log_file" || true)
            fi

            test_f1=$(grep -o "Test F1 Score: [0-9.e+-]*" "$log_file" | awk '{print $4}' | tail -n1)
            val_f1=$(grep -o "Validation F1 Score: [0-9.e+-]*" "$log_file" | awk '{print $4}' | tail -n1)
            test_auc=$(grep -o "Test AUC: [0-9.e+-]*" "$log_file" | awk '{print $3}' | tail -n1)
            val_auc=$(grep -o "Validation AUC: [0-9.e+-]*" "$log_file" | awk '{print $3}' | tail -n1)
            precision=$(grep -o "Test Precision: [0-9.e+-]*" "$log_file" | awk '{print $4}' | tail -n1)
            recall=$(grep -o "Test Recall: [0-9.e+-]*" "$log_file" | awk '{print $4}' | tail -n1)
            accuracy=$(grep -o "Test Accuracy: [0-9.e+-]*" "$log_file" | awk '{print $4}' | tail -n1)

            python - "$metadata_file" <<PY
import json,sys
f=sys.argv[1]
try:
    data=json.load(open(f))
except Exception:
    data=[]
rec={"assay_name":"$assay","MF":"$fp","Model":"$model","duration":$duration}
error="$error_msg"
if error:
    rec["Error"] = error
else:
    rec.update({
        "F1":float("${test_f1:-0}"),
        "valF1":float("${val_f1:-0}"),
        "AUC":float("${test_auc:-0}"),
        "valAUC":float("${val_auc:-0}"),
        "Precision":float("${precision:-0}"),
        "Recall":float("${recall:-0}"),
        "Accuracy":float("${accuracy:-0}")
    })
data.append(rec)
json.dump(data,open(f,'w'),indent=2)
PY
        done
    done
    assay_index=$((assay_index+1))
done

python "$model_dir/update_training_results.py" "$project_dir" "$input_excel" "$metadata_file"
echo "[$(date '+%F %T')] Training complete" >> "$slurm_out"
