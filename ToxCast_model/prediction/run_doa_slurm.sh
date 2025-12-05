#!/usr/bin/env bash

#SBATCH --job-name=GNN_launcher
#SBATCH --partition=gpu1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

PARTITIONS=(gpu1 gpu6 gpu2 gpu3 gpu4 gpu5)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CMD="cd \"$SCRIPT_DIR/..\" && PYTHONPATH=. python prediction/calc_doa.py"

JOBID=""
for p in "${PARTITIONS[@]}"; do
    JOBID=$(sbatch --parsable --partition="$p" --gres=gpu --cpus-per-task=8 --mem=16G --wrap="$CMD")
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
sbatch --partition="$LAST_PART" --gres=gpu --cpus-per-task=8 --mem=16G --wrap="$CMD"
