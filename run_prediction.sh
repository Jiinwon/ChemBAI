#!/bin/bash
# Run fingerprint generation and either training or prediction based on config.py

# resolve script directory and determine action
script_dir="$(cd "$(dirname "$0")" && pwd)"
model_dir="$script_dir/ToxCast_model"
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
    bash prediction/run_doa_slurm.sh "$result_file" "$metadata_file"
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
python -m toxcast_pkg.smiles2fing

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
        latest_dir=$(ls -dt "$results_dir"/* | head -n 1)
        result_file=$(ls "$results_dir"/*_prediction.xlsx | head -n 1)
        metadata_file="$results_dir/metadata.json"
        bash prediction/run_doa_slurm.sh "$result_file" "$metadata_file"
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

