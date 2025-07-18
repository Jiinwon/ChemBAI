#!/bin/bash
# Run fingerprint generation and either training or prediction based on config.py

# resolve script directory and determine action
script_dir="$(cd "$(dirname "$0")" && pwd)"
model_dir="$script_dir/ToxCast_model"
step="${1:-predict}"

if [ "$step" = "doa" ]; then
    cd "$model_dir" || exit 1
    PYTHONPATH=. python prediction/calc_doa.py
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
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

