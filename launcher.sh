#!/bin/bash
# Unified launcher for training or prediction

script_dir="$(cd "$(dirname "$0")" && pwd)"
mode=$(python - <<'PY'
import config
print(config.OBJECTS[config.OBJECT])
PY
)

case "$mode" in
    training)
        bash "$script_dir/run_training.sh" "$@"
        ;;
    prediction)
        bash "$script_dir/run_prediction.sh" "$@"
        ;;
    *)
        echo "Unknown OBJECT mode: $mode" >&2
        exit 1
        ;;
esac
