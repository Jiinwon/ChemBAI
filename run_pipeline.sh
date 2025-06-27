#!/bin/bash
# Run fingerprint generation and either training or prediction based on config.py

# resolve script directory and move into model directory
script_dir="$(cd "$(dirname "$0")" && pwd)"
model_dir="$script_dir/ToxCast_model"
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
        python prediction/Predict_data.py
        ;;
    both)
        bash ToxCast_model_training.sh
        python prediction/Predict_data.py
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

