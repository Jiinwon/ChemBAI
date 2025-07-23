#!/usr/bin/env bash

#SBATCH --job-name=GNN_launcher
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

PARTITIONS=(gpu6 gpu1 gpu2 gpu3 gpu4 gpu5)
GRES="gpu"
CPUS_PER_TASK=8
MEM_PER_TASK="16G"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RESULT_FILE="$1"
METADATA_FILE="$2"

JOBID=""
for p in "${PARTITIONS[@]}"; do
    JOBID=$(sbatch --parsable --partition="$p" --gres="$GRES" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
      --wrap="cd \"$SCRIPT_DIR/..\" && PYTHONPATH=. python prediction/calc_doa.py \"$RESULT_FILE\" --metadata-file \"$METADATA_FILE\"")
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
sbatch --partition="$LAST_PART" --gres="$GRES" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
  --wrap="cd \"$SCRIPT_DIR/..\" && PYTHONPATH=. python prediction/calc_doa.py \"$RESULT_FILE\" --metadata-file \"$METADATA_FILE\""
