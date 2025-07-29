#!/bin/bash
# Run fingerprint generation and either training or prediction based on config.py

# resolve script directory and determine action
script_dir="$(cd "$(dirname "$0")" && pwd)"
model_dir="$(cd "$script_dir/../ToxCast_model" && pwd)"

if [ -n "$SLURM_LAUNCHED" ]; then
    run_doa() { bash prediction/run_doa_slurm.sh "$1" "$2"; }
else
    run_doa() { PYTHONPATH=. python prediction/calc_doa.py "$1" --metadata-file "$2"; }
fi

current_date=$(date +%Y-%m-%d)
current_time=$(date +%H-%M-%S)
default_log_dir="$script_dir/../slurm_logs/$current_date/$current_time"
slurm_log_dir="${SLURM_LOG_DIR:-$default_log_dir}"
mkdir -p "$slurm_log_dir"
slurm_out="$slurm_log_dir/slurm.out"
echo "[$(date '+%F %T')] run_prediction start" >> "$slurm_out"
step="${1:-predict}"

if [ "$step" = "doa" ]; then
    cd "$model_dir" || exit 1
    result_file="$2"
    metadata_file="$3"
    if [ -z "$result_file" ]; then
        results_dir=$(python - <<'PY'
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

if [ "$step" = "train_doa" ]; then
    cd "$model_dir" || exit 1
    PYTHONPATH=. python prediction/calc_train_doa.py "$2"
    exit $?
fi

cd "$model_dir" || exit 1

# validate experiment setup
python - <<'PY'
import config
config.validate_paths()
PY

# generate fingerprints
#python -m toxcast_pkg.smiles2fing

# determine mode from config
mode=$(python - <<'PY'
import config
print(config.OBJECTS[config.OBJECT])
PY
)

echo "Running mode: $mode"

case "$mode" in
    training)
        bash ToxCast_model_training.sh
        ;;
    prediction)
        PYTHONPATH=. python prediction/Predict_data.py --skip-doa
        results_dir=$(python - <<'PY'
import config
print(config.RESULTS_DIR)
PY
)
        latest_dir=$(ls -dt "$results_dir"/*/ | head -n1)  
        result_file=$(ls "$latest_dir"/*_prediction.xlsx | head -n1)
        metadata_file="$results_dir/metadata.json"
        run_doa "$result_file" "$metadata_file"
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

echo "[$(date '+%F %T')] run_prediction end" >> "$slurm_out"

