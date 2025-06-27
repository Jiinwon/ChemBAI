#!/bin/bash
# Run fingerprint generation and either training or prediction based on config.py

# ensure script runs from its directory
cd "$(dirname "$0")"

# generate fingerprints
python -m ./toxcast_pkg.smiles2fing

# determine mode
mode=$(python - <<'PY'
import config
print(config.OBJECTS[config.OBJECT])
PY
)

echo "Running mode: $mode"

if [ "$mode" = "training" ]; then
    bash ./ToxCast_model_training.sh
elif [ "$mode" = "prediction" ]; then
    python ./prediction/Predict_data.py
else
    echo "Unknown mode: $mode"
    exit 1
fi
