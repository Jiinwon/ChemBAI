#!/bin/bash

# Wrapper around run_training.sh that captures GPU utilisation metrics
# alongside the normal training workflow. The script mirrors the
# environment/bootstrap logic of run_training.sh so it can be submitted
# through the same Slurm pipeline.

set -euo pipefail
trap 'echo "[ERR] line:$LINENO cmd:${BASH_COMMAND}" >&2' ERR
shopt -s inherit_errexit 2>/dev/null || true

usage() {
    cat <<'USAGE'
Usage: run_training_with_gpu_monitor.sh [OPTIONS] [PROJECT_DIR]

Options:
  --interval SECONDS   Sampling interval for nvidia-smi (default: 5).
  --gpu-log PATH       Write raw utilisation samples to PATH.
  --summary PATH       Optional JSON summary file with basic statistics.
  -h, --help           Show this message and exit.

All remaining arguments are forwarded to run_training.sh. When PROJECT_DIR is
omitted the default from ToxCast_model/config.py is used, matching
run_training.sh behaviour.
USAGE
}

strip_carriage_returns() { printf '%s' "${1//$'\r'/}"; }

module load cuda/12.1.1 || {
    echo "Failed to load cuda module" >&2
    exit 1
}
module load gnu12/12.3.0 || {
    echo "Failed to load gnu12 module" >&2
    exit 1
}


interval=5
raw_log_path=""
summary_path=""
declare -a forward_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)
            if [[ $# -lt 2 ]]; then
                echo "--interval requires a numeric argument" >&2
                exit 1
            fi
            interval="$2"
            shift 2
            ;;
        --gpu-log)
            if [[ $# -lt 2 ]]; then
                echo "--gpu-log requires a path" >&2
                exit 1
            fi
            raw_log_path="$2"
            shift 2
            ;;
        --summary)
            if [[ $# -lt 2 ]]; then
                echo "--summary requires a path" >&2
                exit 1
            fi
            summary_path="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                forward_args+=("$1")
                shift
            done
            break
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_model_dir="$(cd "$script_dir/../ToxCast_model" && pwd)"

default_project_dir=$(PYTHONPATH="$base_model_dir" python - <<'PY'
import config
print(config.BASE_DIR)
PY
)
default_project_dir=$(strip_carriage_returns "$default_project_dir")

project_dir=""
if [[ ${#forward_args[@]} -gt 0 ]]; then
    project_dir_candidate="${forward_args[0]}"
    if [[ -n "$project_dir_candidate" ]]; then
        project_dir="$project_dir_candidate"
    fi
fi

if [[ -z "$project_dir" ]]; then
    project_dir="$default_project_dir"
fi
project_dir=$(strip_carriage_returns "$project_dir")

if [[ -z "$raw_log_path" ]]; then
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    raw_log_path="$project_dir/logs/gpu_utilization_${timestamp}.csv"
fi
raw_log_dir="$(dirname "$raw_log_path")"
mkdir -p "$raw_log_dir"

if [[ -n "$summary_path" ]]; then
    summary_dir="$(dirname "$summary_path")"
    mkdir -p "$summary_dir"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found in PATH. Ensure the NVIDIA tools are available." >&2
    exit 1
fi

if ! [[ "$interval" =~ ^[0-9]+$ ]] || [[ "$interval" -le 0 ]]; then
    echo "--interval must be a positive integer (got '$interval')" >&2
    exit 1
fi

query_fields="timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free"

# Capture header with units for readability
nvidia-smi --query-gpu="$query_fields" --format=csv > "$raw_log_path"

nvidia-smi --query-gpu="$query_fields" --format=csv,noheader -l "$interval" >> "$raw_log_path" &
monitor_pid=$!

cleanup_monitor() {
    if [[ -n "${monitor_pid:-}" ]] && kill -0 "$monitor_pid" >/dev/null 2>&1; then
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap 'cleanup_monitor' EXIT

training_status=0
"$script_dir/run_training.sh" "${forward_args[@]}" || training_status=$?
cleanup_monitor
trap - EXIT

if [[ $training_status -ne 0 ]]; then
    echo "Training exited with status $training_status" >&2
fi

echo "GPU utilisation samples saved to $raw_log_path" >&2

if [[ -n "$summary_path" ]]; then
    python - "$raw_log_path" "$summary_path" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
with log_path.open(newline='') as fh:
    reader = csv.DictReader(fh)
    util_keys = [
        key
        for key in reader.fieldnames or []
        if key.lower().startswith('utilization.gpu')
    ]
    if not util_keys:
        summary = {
            'samples': 0,
            'error': 'GPU utilisation column not found in log file',
        }
        out_path.write_text(json.dumps(summary, indent=2))
        sys.exit(0)
    util_key = util_keys[0]
    for row in reader:
        try:
            util_value = row[util_key].replace('%', '').strip()
            rows.append(
                {
                    'gpu': int(row['index']),
                    'util_gpu': float(util_value),
                }
            )
        except (ValueError, KeyError):
            # Skip malformed entries (e.g. during driver resets)
            continue

if not rows:
    summary = {'samples': 0}
else:
    util_by_gpu = {}
    for r in rows:
        util_by_gpu.setdefault(r['gpu'], []).append(r['util_gpu'])
    summary = {
        'samples': len(rows),
        'gpus': {
            gpu: {
                'mean_util': statistics.mean(values),
                'max_util': max(values),
                'min_util': min(values),
            }
            for gpu, values in util_by_gpu.items()
        },
    }

out_path.write_text(json.dumps(summary, indent=2))
PY
    echo "GPU utilisation summary saved to $summary_path" >&2
fi

exit "$training_status"