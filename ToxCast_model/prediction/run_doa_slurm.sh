#!/usr/bin/env bash

#SBATCH --job-name=GNN_launcher
#SBATCH --partition=gpu1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

PARTITIONS=(gpu1 gpu6 gpu2 gpu3 gpu4 gpu5)
GRES="gpu"
CPUS_PER_TASK=8
MEM_PER_TASK="16G"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EXPERIMENT_EXCEL="$1"
TRAIN_EXCEL="$2"
TRAIN_FP_BASE="$3"

CMD="cd \"$SCRIPT_DIR/..\" && PYTHONPATH=. python prediction/calc_train_doa.py"
if [ -n "$EXPERIMENT_EXCEL" ]; then
    CMD="$CMD --experiment-excel \"$EXPERIMENT_EXCEL\""
fi
if [ -n "$TRAIN_EXCEL" ]; then
    CMD="$CMD --train-excel \"$TRAIN_EXCEL\""
fi
if [ -n "$TRAIN_FP_BASE" ]; then
    CMD="$CMD --train-fp-base \"$TRAIN_FP_BASE\""
fi

JOBID=""
for p in "${PARTITIONS[@]}"; do
    JOBID=$(sbatch --parsable --partition="$p" --gres="$GRES" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" --wrap="$CMD")
    sleep 2
    info=$(squeue -j "$JOBID" -h -o '%T %R')
    state=$(echo "$info" | awk '{print $1}')
    if [ "$state" != "PD" ]; then
        echo "Job $JOBID running on $(echo "$info" | awk '{print $2}')"
        exit 0
    fi
    scancel "$JOBID"
    echo "Partition $p busy, trying next..."
    sleep 10
done

LAST_PART=${PARTITIONS[$(( ${#PARTITIONS[@]} - 1 ))]}
sbatch --partition="$LAST_PART" --gres="$GRES" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" --wrap="$CMD"
