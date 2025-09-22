#!/bin/bash

# Run fingerprint generation and either training or prediction based on config.py

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

# Base directory containing shared config and experiments
base_model_dir="$(cd "$script_dir/../ToxCast_model" && pwd)"

# Determine model directory based on config.VERSION
version=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(getattr(config, "VERSION", 1))
PY
)

if [ "$version" = "2" ]; then
    model_dir="$base_model_dir/ToxCast_model_v.2"
    training_script="ToxCast_model_training_v.2.sh"
    prediction_script="prediction_v.2/Predict_data_v.2.py"
else
    model_dir="$base_model_dir"
    training_script="ToxCast_model_training.sh"
    prediction_script="prediction/Predict_data.py"
fi

# Run DOA depending on slurm or local
if [ -n "$SLURM_LAUNCHED" ]; then
    run_doa() { bash prediction/run_doa_slurm.sh "$1" "$2"; }
else
    run_doa() { PYTHONPATH=".:$base_model_dir" python prediction/calc_doa.py "$1" --metadata-file "$2"; }
fi

# Resolve project directory from config
project_dir=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(config.BASE_DIR)
PY
)
project_name="$(basename "$project_dir")"

# Resolve mode from config
mode=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(config.OBJECTS[config.OBJECT])
PY
)

job_name="${project_name}_${mode}"
slurm_out="$project_dir/${job_name}.out"
mkdir -p "$project_dir"

# Slurm submission if not launched
if [ -z "$SLURM_LAUNCHED" ]; then
    arg_str=""
    for a in "$@"; do
        arg_str="$arg_str \"$a\""
    done

    # 여기서 gpu1에 고정 → 원하면 다른 파티션 fallback 로직도 넣을 수 있음
    sbatch --partition=gpu1 --gres=gpu:rtx3090:1 \
        --cpus-per-task=16 --mem=32G --time=03:00:00 \
        --job-name="$job_name" --output="$slurm_out" \
        --wrap="SLURM_LAUNCHED=1 SLURM_SUBMIT_DIR=\"$PWD\" bash \"$script_dir/run_prediction.sh\"$arg_str"
    exit 0
fi

echo "[$(date '+%F %T')] run_prediction start" >> "$slurm_out"
step="${1:-predict}"

# doa mode
if [ "$step" = "doa" ]; then
    cd "$model_dir" || exit 1
    result_file="$2"
    metadata_file="$3"
    if [ -z "$result_file" ]; then
        results_dir=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(config.RESULTS_DIR)
PY
)
        latest_dir=$(ls -dt "$results_dir"/* | head -n 1)
        result_file=$(ls "$latest_dir"/*_prediction.xlsx | head -n 1)
        metadata_file="$results_dir/metadata.json"
    fi
    run_doa "$result_file" "$metadata_file"
    exit $?
fi

# train_doa mode
if [ "$step" = "train_doa" ]; then
    cd "$model_dir" || exit 1
    PYTHONPATH=".:$base_model_dir" python prediction/calc_train_doa.py "$2"
    exit $?
fi

cd "$model_dir" || exit 1

# validate experiment setup
PYTHONPATH="$base_model_dir" python - <<'PY'
import config
config.validate_paths()
PY

# generate fingerprints only if not already present
fp_dir=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(config.FINGERPRINT_OUTPUT_DIR)
PY
)
mkdir -p "$fp_dir"
if [ -z "$(ls -A "$fp_dir" 2>/dev/null)" ]; then
    PYTHONPATH="$model_dir:$base_model_dir" python -m toxcast_pkg.smiles2fing
fi

# mode already determined earlier
echo "Running mode: $mode"

case "$mode" in
    training)
        bash "$training_script"
        ;;
    prediction)
        if [ "$version" = "2" ]; then
            PYTHONPATH=".:$base_model_dir" python "$prediction_script"
        else
            PYTHONPATH=".:$base_model_dir" python "$prediction_script" --skip-doa
            results_dir=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(config.RESULTS_DIR)
PY
)
            latest_dir=$(ls -dt "$results_dir"/*/ | head -n1)
            result_file=$(ls "$latest_dir"/*_prediction.xlsx | head -n1)
            metadata_file="$results_dir/metadata.json"
            run_doa "$result_file" "$metadata_file"
        fi
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

echo "[$(date '+%F %T')] run_prediction end" >> "$slurm_out"
