#!/bin/bash
# Convenience script to train models for a project directory
set -e

script_dir="$(cd "$(dirname "$0")" && pwd)"

# When not launched by a slurm job, submit this script via sbatch similarly to
# prediction/run_doa_slurm.sh.  This checks GPU partitions in the order
# gpu6->gpu1->gpu2->gpu3->gpu4->gpu5 and submits the job to the first partition
# that is not pending.  If all partitions remain pending, the job is submitted
# to gpu1 and left pending.
if [ -z "$SLURM_LAUNCHED" ]; then
    PARTITIONS=(gpu6 gpu1 gpu2 gpu3 gpu4 gpu5)
    GRES="gpu"
    CPUS_PER_TASK=8
    MEM_PER_TASK="16G"

    project_dir="$1"
    if [ -z "$project_dir" ]; then
        echo "Usage: $0 <PROJECT_DIR>" >&2
        exit 1
    fi

    for p in "${PARTITIONS[@]}"; do
        JOBID=$(sbatch --parsable --partition="$p" --gres="$GRES" \
            --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
            --wrap="SLURM_LAUNCHED=1 bash \"$script_dir/run_training.sh\" \"$project_dir\"")
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
        --wrap="SLURM_LAUNCHED=1 bash \"$script_dir/run_training.sh\" \"$project_dir\""
    exit 0
fi

model_dir="$script_dir/ToxCast_model"
project_dir="$1"
if [ -z "$project_dir" ]; then
    echo "Usage: $0 <PROJECT_DIR>" >&2
    exit 1
fi

input_excel="$project_dir/training_input_template.xlsx"
results_dir="$project_dir/results"
logs_dir="$project_dir/logs"
fp_dir="$project_dir/fingerprints"
metadata_file="$project_dir/metadata.json"
mkdir -p "$results_dir" "$logs_dir" "$fp_dir"
: > "$metadata_file"

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
    if v is None:
        val=''
    else:
        val=v.text
    if c.get('t')=='s':
        val=strings[int(val)]
    vals.append(val)
print('\n'.join(vals[1:]))
PY
)

fingerprints=(MACCS Morgan Layered Pattern RDKit)
models=(rf logistic xgb gbt dt)

assay_index=0
for assay in "${assays[@]}"; do
    for fp in "${fingerprints[@]}"; do
        for model in "${models[@]}"; do
            save_dir="$results_dir/${assay}_${fp}_${model}"
            mkdir -p "$save_dir"
            log_file="$logs_dir/${assay}_${fp}_${model}.log"
            start_time=$(date +%s)
            PYTHONPATH="$model_dir" python "$model_dir/run/${model}.py" \
                --fingerprint_type "$fp" \
                --file_path "$input_excel" \
                --model_save_path "$save_dir" \
                --assay_num "$assay_index" \
                --fp_path "$fp_dir" \
                >"$log_file" 2>&1
            end_time=$(date +%s)
            duration=$((end_time - start_time))
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
rec={"assay_name":"$assay","MF":"$fp","Model":"$model","duration":$duration,
     "F1":float("${test_f1:-0}"),"valF1":float("${val_f1:-0}"),
     "AUC":float("${test_auc:-0}"),"valAUC":float("${val_auc:-0}"),
     "Precision":float("${precision:-0}"),"Recall":float("${recall:-0}"),"Accuracy":float("${accuracy:-0}")}
data.append(rec)
json.dump(data,open(f,'w'),indent=2)
PY
        done
    done
    assay_index=$((assay_index+1))
done

python "$model_dir/update_training_results.py" "$project_dir" "$input_excel" "$metadata_file"
