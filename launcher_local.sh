#!/bin/bash
# Launcher that runs without submitting to Slurm
script_dir="$(cd "$(dirname "$0")" && pwd)"
mode=$(python - <<'PY'
import config
print(config.OBJECTS[config.OBJECT])
PY
)
case "$mode" in
    training)
        SLURM_LAUNCHED=1 bash "$script_dir/slurm/run_training.sh" "$@"
        ;;
    prediction)
        SLURM_LAUNCHED=1 bash "$script_dir/slurm/run_prediction.sh" "$@"
        ;;
    *)
        echo "Unknown OBJECT mode: $mode" >&2
        exit 1
        ;;
esac
