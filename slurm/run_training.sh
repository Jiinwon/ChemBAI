#!/bin/bash

# Convenience script to train models for a project directory
set -e

job_mode="${SLURM_JOB_MODE:-controller}"

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
base_model_dir="$(cd "$script_dir/../ToxCast_model" && pwd)"
version=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(getattr(config, "VERSION", 1))
PY
)

default_project_dir=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(config.BASE_DIR)
PY
)

project_dir="${1:-$default_project_dir}"
project_name="$(basename "$project_dir")"
project_logs_dir="$project_dir/logs"
mkdir -p "$project_dir" "$project_logs_dir"
slurm_out="$project_logs_dir/logs_run_training.out"
slurm_err="$project_logs_dir/logs_run_training.err"

results_dir="$project_dir/results"
logs_dir="$project_logs_dir"
mkdir -p "$results_dir" "$logs_dir"

fingerprints=(MACCS Morgan Layered Pattern RDKit)
models=(rf logistic xgb gbt dt)

if [ "$version" = "2" ]; then
    model_dir="$base_model_dir/ToxCast_model_v.2"
    run_subdir="run_v.2"
elif [ "$version" = "3" ]; then
    model_dir="$base_model_dir"
    run_subdir="run_v3"

    if [ "$job_mode" = "worker" ]; then
        if [ -z "${TASK_FILE:-}" ] || [ -z "${PROJECT_DIR:-}" ]; then
            echo "TASK_FILE and PROJECT_DIR must be provided for worker mode" >&2
            exit 1
        fi
        if [ -z "${MODEL_DIR:-}" ] || [ -z "${RUN_SUBDIR:-}" ]; then
            echo "MODEL_DIR and RUN_SUBDIR must be provided for worker mode" >&2
            exit 1
        fi
        if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
            echo "SLURM_ARRAY_TASK_ID is not set for worker mode" >&2
            exit 1
        fi

                task_index=$((SLURM_ARRAY_TASK_ID + 1))
        task_line=$(sed -n "${task_index}p" "$TASK_FILE")
        if [ -z "$task_line" ]; then
            echo "No task entry for index $SLURM_ARRAY_TASK_ID in $TASK_FILE" >&2
            exit 1
        fi

        IFS='|' read -r seed_dir seed_name assay model fp <<< "$task_line"

        if [ -z "$seed_dir" ] || [ -z "$assay" ] || [ -z "$model" ] || [ -z "$fp" ]; then
            echo "Malformed task entry: $task_line" >&2
            exit 1
        fi

        results_dir="$PROJECT_DIR/results"
        logs_dir="${PROJECT_LOGS_DIR:-$PROJECT_DIR/logs}"
        seed_results_dir="$results_dir/$seed_name"
        seed_logs_dir="$logs_dir/$seed_name"
        log_dir="$seed_logs_dir/$assay"
        save_dir="$seed_results_dir/${assay}_${model}_${fp}"
        log_file="$log_dir/${assay}_${model}_${fp}.log"

        mkdir -p "$save_dir" "$log_dir"

        resolve_seed_paths "$seed_dir"
        if [ ! -f "$train_csv" ]; then
            echo "Missing train_df.csv for $seed_dir" >&2
            exit 1
        fi

        project_name_worker="$(basename "$PROJECT_DIR")"
        job_label="${project_name_worker}_${assay}_${model}_${fp}"
        sanitized_label=$(echo "$job_label" | tr -c 'A-Za-z0-9_-' '_')
        if command -v scontrol >/dev/null 2>&1; then
            scontrol update JobId="$SLURM_JOB_ID" JobName="$sanitized_label" >/dev/null 2>&1 || true
        fi

        start_timestamp="$(date '+%F %T')"
        start_time=$(date +%s)

        : > "$log_file"
        echo "[$start_timestamp] START ${seed_name}:${assay}_${model}_${fp}" | tee -a "$log_file" >&2

        set +e
        PYTHONPATH="$MODEL_DIR" python "$MODEL_DIR/$RUN_SUBDIR/${model}.py" \
            --fingerprint_type "$fp" \
            --train_csv "$train_csv" \
            --val_csv "$val_csv" \
            --test_csv "$test_csv" \
            --train_fp_dir "$train_fp_dir" \
            --val_fp_dir "$val_fp_dir" \
            --test_fp_dir "$test_fp_dir" \
            --assay_name "$assay" \
            --model_save_path "$save_dir" \
            2>&1 | tee -a "$log_file" >&2
        exit_code=${PIPESTATUS[0]}
        set -e

        end_time=$(date +%s)
        duration=$(( end_time - start_time ))
        end_timestamp="$(date '+%F %T')"
        echo "[$end_timestamp] END ${seed_name}:${assay}_${model}_${fp} (status=${exit_code}, duration=${duration}s)" | tee -a "$log_file" >&2

        exit $exit_code
    elif [ "$job_mode" = "summary" ]; then
        if [ -z "${PROJECT_DIR:-}" ]; then
            echo "PROJECT_DIR must be provided for summary mode" >&2
            exit 1
        fi

        data_dir="$PROJECT_DIR/data"
        if [ ! -d "$data_dir" ]; then
            echo "Data directory not found for VERSION=3: $data_dir" >&2
            exit 1
        fi

        mapfile -t seed_dirs < <(find "$data_dir" -maxdepth 1 -mindepth 1 -type d -name 'seed_*' | sort)
        results_dir="$PROJECT_DIR/results"

        for seed_dir in "${seed_dirs[@]}"; do
            seed_name="$(basename "$seed_dir")"
            seed_results_dir="$results_dir/$seed_name"
            if [ -d "$seed_results_dir" ]; then
                PYTHONPATH="$MODEL_DIR" python -m toxcast_pkg.v3_summary \
                    --seed-dir "$seed_dir" \
                    --results-dir "$seed_results_dir" \
                    --output "$seed_results_dir/summary.csv"
            fi
        done

        if [ -d "$results_dir" ]; then
            PYTHONPATH="$MODEL_DIR" python -m toxcast_pkg.v3_summary \
                --results-dir "$results_dir" \
                --aggregate "$results_dir/summary_all_seeds.csv"
        fi

        echo "[$(date '+%F %T')] Summary aggregation complete for $(basename "$PROJECT_DIR")" >&2
        exit 0
    fi

    if [ "$job_mode" != "controller" ]; then
        echo "Unknown job mode '$job_mode' for VERSION=3" >&2
        exit 1
    fi

    if [ -z "$1" ] && [ -z "${PROJECT_DIR:-}" ]; then
        echo "Using project directory from config: $project_dir"
    fi

        data_dir="$project_dir/data"
    if [ ! -d "$data_dir" ]; then
        echo "Data directory not found for VERSION=3: $data_dir" >&2
        exit 1
    fi

    mapfile -t seed_dirs < <(find "$data_dir" -maxdepth 1 -mindepth 1 -type d -name 'seed_*' | sort)
    if [ ${#seed_dirs[@]} -eq 0 ]; then
        echo "No seed directories detected under $data_dir" >&2
        exit 1
    fi

    tasks_file="$project_logs_dir/training_tasks.txt"
    : > "$tasks_file"
    task_count=0

    for seed_dir in "${seed_dirs[@]}"; do
        seed_name="$(basename "$seed_dir")"
        seed_results_dir="$results_dir/$seed_name"
                seed_logs_dir="${PROJECT_LOGS_DIR:-$PROJECT_DIR/logs}/$seed_name"
        mkdir -p "$seed_results_dir" "$seed_logs_dir"

        resolve_seed_paths "$seed_dir"
        if [ ! -f "$train_csv" ]; then
            echo "Missing train_df.csv for $seed_dir" >&2
            continue
        fi

        for split in train val test; do
            csv_var="${split}_csv"
            fp_var="${split}_fp_dir"
            csv_path="${!csv_var}"
            fp_path="${!fp_var}"
            if [ -n "$csv_path" ] && [ -f "$csv_path" ]; then
                mkdir -p "$fp_path"
                PYTHONPATH="$model_dir" python - "$csv_path" "$fp_path" "${fingerprints[@]}" <<'PYGEN'
import sys
from pathlib import Path
from toxcast_pkg.v3_data import load_split_data

csv_path = Path(sys.argv[1])
fp_dir = Path(sys.argv[2])
fingerprints = sys.argv[3:]

for fp in fingerprints:
    load_split_data(csv_path, fp, None, fp_dir)
PYGEN
            fi
        done

        mapfile -t assays < <(PYTHONPATH="$base_model_dir" python - "$train_csv" <<'PY'
import sys
from toxcast_pkg.v3_data import get_assay_names_from_csv
print("\n".join(get_assay_names_from_csv(sys.argv[1])))
PY
)

        if [ ${#assays[@]} -eq 0 ]; then
            echo "No assays detected for $seed_dir" >&2
            continue
        fi

        for assay in "${assays[@]}"; do
                    for model in "${models[@]}"; do
                for fp in "${fingerprints[@]}"; do
                    printf '%s|%s|%s|%s|%s\n' "$seed_dir" "$seed_name" "$assay" "$model" "$fp" >> "$tasks_file"
                    ((task_count++))
                done
            done
        done
            done

    if [ $task_count -eq 0 ]; then
        echo "No training tasks to submit for $project_name" >&2
        exit 0
    fi

    PARTITIONS=(gpu1 gpu2 gpu3 gpu4 gpu5 gpu6)
    GRES="gpu"
    CPUS_PER_TASK=8
    MEM_PER_TASK="16G"

    array_spec="0-$((task_count - 1))%20"
    output_pattern="$project_logs_dir/${project_name}_%A_%a.out"
    error_pattern="$project_logs_dir/${project_name}_%A_%a.err"
    export_args="ALL,SLURM_JOB_MODE=worker,PROJECT_DIR=$project_dir,MODEL_DIR=$model_dir,RUN_SUBDIR=$run_subdir,BASE_MODEL_DIR=$base_model_dir,TASK_FILE=$tasks_file,PROJECT_LOGS_DIR=$project_logs_dir"

    chosen_partition=""
    array_jobid=""

    for p in "${PARTITIONS[@]}"; do
        JOBID=$(sbatch --parsable --partition="$p" --gres="$GRES" \
            --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
            --job-name="${project_name}_training" --array="$array_spec" \
            --output="$output_pattern" --error="$error_pattern" \
            --export="$export_args" "$script_dir/run_training.sh" "$project_dir")
        if [ -z "$JOBID" ]; then
            echo "Failed to submit job array on partition $p" >&2
            continue
        fi
        sleep 2
        info=$(squeue -j "$JOBID" -h -o '%T %R')
        state=$(echo "$info" | awk '{print $1}')
        if [ "$state" != "PD" ]; then
            chosen_partition="$p"
            array_jobid="$JOBID"
            break
        fi
        scancel "$JOBID" >/dev/null 2>&1 || true
        echo "Partition $p busy, trying next..." >&2
        sleep 10
    done

        if [ -z "$array_jobid" ]; then
        last_index=$(( ${#PARTITIONS[@]} - 1 ))
        LAST_PART=${PARTITIONS[$last_index]}
        array_jobid=$(sbatch --parsable --partition="$LAST_PART" --gres="$GRES" \
            --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
            --job-name="${project_name}_training" --array="$array_spec" \
            --output="$output_pattern" --error="$error_pattern" \
            --export="$export_args" "$script_dir/run_training.sh" "$project_dir")
        chosen_partition="$LAST_PART"
    fi

    if [ -z "$array_jobid" ]; then
        echo "Failed to submit training job array" >&2
        exit 1
    fi

    summary_export="ALL,SLURM_JOB_MODE=summary,PROJECT_DIR=$project_dir,MODEL_DIR=$model_dir,RUN_SUBDIR=$run_subdir,BASE_MODEL_DIR=$base_model_dir,PROJECT_LOGS_DIR=$project_logs_dir"
    summary_out="$project_logs_dir/${project_name}_summary_%j.out"
    summary_err="$project_logs_dir/${project_name}_summary_%j.err"
    summary_jobid=$(sbatch --parsable --partition="$chosen_partition" --gres="$GRES" \
        --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
        --job-name="${project_name}_summary" --dependency="afterany:${array_jobid}" \
        --output="$summary_out" --error="$summary_err" \
        --export="$summary_export" "$script_dir/run_training.sh" "$project_dir")

    echo "Submitted ${task_count} training tasks as array job ${array_jobid} on ${chosen_partition}" >&2
    if [ -n "$summary_jobid" ]; then
        echo "Summary job scheduled with JobID ${summary_jobid}" >&2
    fi
    exit 0
else
    model_dir="$base_model_dir"
    run_subdir="run"
fi
input_excel=("$project_dir"/*.xlsx)
fp_dir="$project_dir/fingerprints"
metadata_file="$project_dir/metadata.json"
mkdir -p "$fp_dir"
: > "$metadata_file"

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