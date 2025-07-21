#!/usr/bin/env bash

#SBATCH --job-name=GNN_launcher
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -e

MAX_JOBS=20
PARTITIONS=(gpu1 gpu2 gpu3 gpu4 gpu5 gpu6)
DEFAULT_PART="gpu1"
MAX_RUNNING_PER_PART=10
GRES="gpu:rtx3090:1"
CPUS_PER_TASK=8
MEM_PER_TASK="16G"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

current_job_count() {
    squeue -u "$USER" -h | wc -l
}

running_in_partition() {
    local p="$1"
    squeue -u "$USER" -p "$p" -t R -h | wc -l
}

has_idle_nodes() {
    local p="$1"
    sinfo -h -p "$p" -o "%D %t" \
      | awk '$2=="idle"||$2=="mix"{sum+=$1}END{print sum}' \
      | grep -q '[1-9]'
}

wait_for_slot() {
    while [ "$(current_job_count)" -ge "$MAX_JOBS" ]; do
        echo "▶ waiting for submit slot ($(current_job_count)/$MAX_JOBS)…"
        sleep 30
    done
}

RESULT_FILE="$1"
METADATA_FILE="$2"

for p in "${PARTITIONS[@]}"; do
    if [ "$(running_in_partition "$p")" -lt "$MAX_RUNNING_PER_PART" ] && has_idle_nodes "$p"; then
        PART="$p"
        break
    fi
done
PART=${PART:-$DEFAULT_PART}

wait_for_slot

sbatch --partition="$PART" --gres="$GRES" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM_PER_TASK" \
  --wrap="cd \"$SCRIPT_DIR/..\" && PYTHONPATH=. python prediction/calc_doa.py \"$RESULT_FILE\" --metadata-file \"$METADATA_FILE\""
