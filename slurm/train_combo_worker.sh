#!/bin/bash

set -euo pipefail

cleanup_temp_seed_file() {
    if [ -n "${__TEMP_SEED_FILE:-}" ] && [ -f "${__TEMP_SEED_FILE}" ]; then
        rm -f "${__TEMP_SEED_FILE}"
    fi
}

__TEMP_SEED_FILE=""

trap_err() {
    local line_no="$1"
    local cmd="$2"
    cleanup_temp_seed_file
    echo "[ERR] line:$line_no cmd:$cmd" >&2
}

trap 'trap_err $LINENO "$BASH_COMMAND"' ERR
trap cleanup_temp_seed_file EXIT
shopt -s inherit_errexit 2>/dev/null || true

DEBUG="${DEBUG:-0}"
case "$DEBUG" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
        DEBUG=1
        ;;
    *)
        DEBUG=0
        ;;
esac
dbg() { [ "$DEBUG" -eq 1 ] && echo "DEBUG: $*" >&2; }

strip_cr() { printf '%s' "${1//$'\r'/}"; }

resolve_seed_paths() {
    if [ $# -lt 1 ] || [ -z "$1" ]; then
        echo "resolve_seed_paths: seed directory argument required" >&2
        return 1
    fi

    local seed_dir="$1"

    train_csv="$seed_dir/train_df.csv"
    val_csv="$seed_dir/val_df.csv"
    test_csv="$seed_dir/test_df.csv"
    if [ ! -f "$train_csv" ]; then train_csv="$seed_dir/train/train_df.csv"; fi
    if [ ! -f "$val_csv" ]; then val_csv="$seed_dir/val/val_df.csv"; fi
    if [ ! -f "$test_csv" ]; then test_csv="$seed_dir/test/test_df.csv"; fi

    train_fp_dir="$seed_dir/fingerprints/train"
    val_fp_dir="$seed_dir/fingerprints/val"
    test_fp_dir="$seed_dir/fingerprints/test"
    if [ -d "$seed_dir/train/fingerprints" ]; then train_fp_dir="$seed_dir/train/fingerprints"; fi
    if [ -d "$seed_dir/val/fingerprints" ]; then val_fp_dir="$seed_dir/val/fingerprints"; fi
    if [ -d "$seed_dir/test/fingerprints" ]; then test_fp_dir="$seed_dir/test/fingerprints"; fi
}

require_var() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: $name" >&2
        exit 1
    fi
}

require_var PROJECT_DIR
require_var MODEL_DIR
require_var RUN_SUBDIR
require_var ASSAY_NAME
require_var MODEL_NAME
require_var FINGERPRINT_NAME

PROJECT_DIR=$(strip_cr "$PROJECT_DIR")
MODEL_DIR=$(strip_cr "$MODEL_DIR")
RUN_SUBDIR=$(strip_cr "$RUN_SUBDIR")
ASSAY_NAME=$(strip_cr "$ASSAY_NAME")
MODEL_NAME=$(strip_cr "$MODEL_NAME")
FINGERPRINT_NAME=$(strip_cr "$FINGERPRINT_NAME")

LOGS_DIR=$(strip_cr "${LOGS_DIR:-$PROJECT_DIR/logs}")
RESULTS_DIR=$(strip_cr "${RESULTS_DIR:-$PROJECT_DIR/results}")
PYTHON_BIN=$(strip_cr "${PYTHON_BIN:-python}")

if [ -z "${SEED_FILE:-}" ]; then
    tmp_seed_file="$(mktemp "${TMPDIR:-/tmp}/seed_list.XXXXXX")"
    if ! "$PYTHON_BIN" - "$PROJECT_DIR" "$MODEL_DIR" "$tmp_seed_file" <<'PY'
import sys
from pathlib import Path

project_dir = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
if len(sys.argv) <= 3 or not sys.argv[3]:
    print("Missing output path for seed list", file=sys.stderr)
    raise SystemExit(1)
out_path = Path(sys.argv[3])

if model_dir:
    sys.path.insert(0, str(model_dir))

try:
    import config  # type: ignore
except Exception as exc:  # pragma: no cover - runtime safeguard
    print(f"Unable to import config.py from {model_dir}: {exc}", file=sys.stderr)
    raise SystemExit(1)

base_dir = Path(project_dir) if project_dir else Path(getattr(config, "BASE_DIR"))
data_dir = Path(getattr(config, "DATA_DIR", base_dir / "data"))

if not data_dir.exists():
    print(f"Data directory not found: {data_dir}", file=sys.stderr)
    raise SystemExit(1)

seed_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("seed_"))
if not seed_dirs:
    print(f"No seed directories found under {data_dir}", file=sys.stderr)
    raise SystemExit(1)

with out_path.open("w", encoding="utf-8") as fh:
    for seed in seed_dirs:
        fh.write(f"{seed}|{seed.name}\n")
PY
    then
        echo "Failed to auto-discover seed directories under $PROJECT_DIR" >&2
        rm -f "$tmp_seed_file"
        exit 1
    fi
    SEED_FILE="$tmp_seed_file"
    __TEMP_SEED_FILE="$tmp_seed_file"
fi

SEED_FILE=$(strip_cr "$SEED_FILE")
PYTHONPATH_BASE=$(strip_cr "${PYTHONPATH_BASE:-$MODEL_DIR}")
RANDOM_STATE=$(strip_cr "${RANDOM_STATE:-}")
ENV_MODULE_INIT=$(strip_cr "${MODULE_INIT:-}")
ENV_MODULE_PURGE=$(strip_cr "${MODULE_PURGE:-1}")
ENV_MODULES=$(strip_cr "${ENV_MODULES:-}")
CONDA_SETUP=$(strip_cr "${CONDA_SETUP:-}")
CONDA_ENV=$(strip_cr "${CONDA_ENV:-}")
JOB_LABEL=$(strip_cr "${JOB_LABEL:-${ASSAY_NAME}_${MODEL_NAME}_${FINGERPRINT_NAME}}")

if [ ! -f "$SEED_FILE" ]; then
    echo "Seed file not found: $SEED_FILE" >&2
    exit 1
fi

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

if [ -n "$ENV_MODULE_INIT" ] && [ -f "$ENV_MODULE_INIT" ]; then
    # shellcheck disable=SC1090
    source "$ENV_MODULE_INIT"
fi

if command -v module >/dev/null 2>&1; then
    if [ "$ENV_MODULE_PURGE" = "1" ] || [ "$ENV_MODULE_PURGE" = "true" ]; then
        module purge || true
    fi
    IFS=',' read -r -a __module_list <<< "$ENV_MODULES"
    for __m in "${__module_list[@]}"; do
        if [ -n "$__m" ]; then
            module load "$__m"
        fi
    done
fi

if [ -n "$CONDA_SETUP" ]; then
    if [ ! -f "$CONDA_SETUP" ]; then
        echo "Conda setup script not found: $CONDA_SETUP" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONDA_SETUP"
    if [ -n "$CONDA_ENV" ]; then
        conda activate "$CONDA_ENV"
    fi
fi

if [ -n "$PYTHONPATH_BASE" ]; then
    export PYTHONPATH="$PYTHONPATH_BASE:${PYTHONPATH:-}"
fi

job_start="$(date '+%F %T')"
echo "[$job_start] Job $JOB_LABEL starting on $(hostname)" >&2

failures=0
line_no=0
while IFS='|' read -r seed_dir seed_name; do
    seed_dir=$(strip_cr "$seed_dir")
    seed_name=$(strip_cr "$seed_name")
    if [ -z "$seed_dir" ]; then
        continue
    fi
    ((++line_no))

    if [ ! -d "$seed_dir" ]; then
        echo "Seed directory not found: $seed_dir" >&2
        failures=1
        continue
    fi

    seed_results_dir="$RESULTS_DIR/$seed_name"
    seed_logs_dir="$LOGS_DIR/$seed_name"
    log_dir="$seed_logs_dir/$ASSAY_NAME"
    log_file="$log_dir/${ASSAY_NAME}_${MODEL_NAME}_${FINGERPRINT_NAME}.log"
    mkdir -p "$seed_results_dir" "$log_dir"

    resolve_seed_paths "$seed_dir"
    if [ ! -f "$train_csv" ]; then
        echo "Missing train_df.csv for $seed_dir" | tee -a "$log_file" >&2
        failures=1
        continue
    fi

    save_dir="$seed_results_dir/${ASSAY_NAME}_${MODEL_NAME}_${FINGERPRINT_NAME}"
    mkdir -p "$save_dir"

    start_ts="$(date '+%F %T')"
    echo "[$start_ts] START ${seed_name}:${ASSAY_NAME}_${MODEL_NAME}_${FINGERPRINT_NAME}" | tee -a "$log_file"
    dbg "Running seed_dir=$seed_dir"

    cmd=("$PYTHON_BIN" "$MODEL_DIR/$RUN_SUBDIR/${MODEL_NAME}.py"
         --fingerprint_type "$FINGERPRINT_NAME"
         --train_csv "$train_csv" --val_csv "$val_csv" --test_csv "$test_csv"
         --train_fp_dir "$train_fp_dir" --val_fp_dir "$val_fp_dir" --test_fp_dir "$test_fp_dir"
         --assay_name "$ASSAY_NAME" --model_save_path "$save_dir")
    if [ -n "$RANDOM_STATE" ]; then
        cmd+=(--random_state "$RANDOM_STATE")
    fi

    set +e
    "${cmd[@]}" 2>&1 | tee -a "$log_file"
    exit_code=${PIPESTATUS[0]}
    set -e

    end_ts="$(date '+%F %T')"
    if [ "$exit_code" -ne 0 ]; then
        echo "[$end_ts] END ${seed_name}:${ASSAY_NAME}_${MODEL_NAME}_${FINGERPRINT_NAME} status=$exit_code" | tee -a "$log_file"
        failures=1
    else
        echo "[$end_ts] END ${seed_name}:${ASSAY_NAME}_${MODEL_NAME}_${FINGERPRINT_NAME} status=0" | tee -a "$log_file"
    fi

done < "$SEED_FILE"

job_end="$(date '+%F %T')"
if [ "$failures" -ne 0 ]; then
    echo "[$job_end] Job $JOB_LABEL completed with failures" >&2
    exit 1
fi

echo "[$job_end] Job $JOB_LABEL completed successfully" >&2
