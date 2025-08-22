#!/bin/bash
# Unified launcher for training or prediction

# Load required modules for GPU execution
module purge
module load cuda/12.1
module load python/3.11.2

script_dir="$(cd "$(dirname "$0")" && pwd)"
mode=$(python - <<'PY'
import config
print(config.OBJECTS[config.OBJECT])
PY
)

case "$mode" in
    training)
        sbatch "$script_dir/slurm/run_training.sh" "$@"
        ;;
    prediction)
        sbatch "$script_dir/slurm/run_prediction.sh" "$@"
        ;;
    *)
        echo "Unknown OBJECT mode: $mode" >&2
        exit 1
        ;;
esac
