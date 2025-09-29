#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
JOB_TEMPLATE_PATH="${SCRIPT_DIR}/job_gpu.sbatch"
LOG_DIR="${REPO_ROOT}/slurm_logs"
LOG_FILE="${LOG_DIR}/submitter.log"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"

cleanup_files=()

cleanup() {
    local status=$?
    for path in "${cleanup_files[@]:-}"; do
        if [[ -n "${path}" && -f "${path}" ]]; then
            rm -f "${path}"
        fi
    done
    if [[ -n "${command_script:-}" && "${command_script_keep}" != true && -f "${command_script}" ]]; then
        rm -f "${command_script}"
    fi
    return ${status}
}

log_error() {
    local line="$1"
    local cmd="$2"
    log_event "[ERROR] line:${line} cmd:${cmd}"
}

trap 'log_error $LINENO "$BASH_COMMAND"' ERR
trap cleanup EXIT

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

require_cmd sbatch
require_cmd squeue
require_cmd scancel
require_cmd date
require_cmd id

log_event() {
    local message="$1"
    local timestamp
    timestamp="$(date '+%F %T')"
    printf '[%s] %s\n' "${timestamp}" "${message}" | tee -a "${LOG_FILE}"
}

usage() {
    cat <<USAGE
Usage: $0 [options] -- <command ...>

Options:
  --job-name NAME           Slurm job name (default: run_v3_gpu)
  --gpu-count N             Number of GPUs to request (default: 1)
  --cpus N                  CPUs per task (default: 8)
  --mem SIZE                Memory request, e.g. 32G (default: 32G)
  --time LIMIT              Time limit in HH:MM:SS (default: 04:00:00)
  --seed ID                 Seed identifier for performance logging
  --assay NAME              Assay name for performance logging
  --model NAME              Model name for performance logging
  --mf NAME                 Molecular fingerprint label for performance logging
  --label TEXT              Custom label recorded in performance log
  --workdir PATH            Working directory for the job (default: repository root)
  --compiler-module NAME    Compiler module to load (default: gnu12/12.3.0)
  --cuda-module NAME        CUDA module to load (default: cuda/12.1.1)
  --module NAME             Additional module to load (can be repeated)
  --poll-interval SEC       Polling interval in seconds (default: 2)
  --poll-timeout SEC        Allocation timeout per partition (default: 3)
  --max-jobs N              Maximum concurrent jobs allowed before aborting (default: 19)
  --user NAME               User name for queue counting (default: current user)
  -h, --help                Show this help message

The command following ``--`` is executed inside the job script. Use
``--use-gpu`` or ``--no-gpu`` flags within that command to control
run_v3 entrypoints.
USAGE
}

job_name="run_v3_gpu"
gpu_count=1
cpus=8
memory="32G"
time_limit="04:00:00"
seed_id=""
assay_name=""
model_name=""
mf_name=""
command_label=""
work_dir="${REPO_ROOT}"
compiler_module="gnu12/12.3.0"
cuda_module="cuda/12.1.1"
extra_modules=()
poll_interval=2
poll_timeout=3
max_jobs=19
user_name=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-name)
            job_name="$2"; shift 2 ;;
        --gpu-count)
            gpu_count="$2"; shift 2 ;;
        --cpus)
            cpus="$2"; shift 2 ;;
        --mem)
            memory="$2"; shift 2 ;;
        --time)
            time_limit="$2"; shift 2 ;;
        --seed)
            seed_id="$2"; shift 2 ;;
        --assay)
            assay_name="$2"; shift 2 ;;
        --model)
            model_name="$2"; shift 2 ;;
        --mf)
            mf_name="$2"; shift 2 ;;
        --label)
            command_label="$2"; shift 2 ;;
        --workdir)
            work_dir="$2"; shift 2 ;;
        --compiler-module)
            compiler_module="$2"; shift 2 ;;
        --cuda-module)
            cuda_module="$2"; shift 2 ;;
        --module)
            extra_modules+=("$2"); shift 2 ;;
        --poll-interval)
            poll_interval="$2"; shift 2 ;;
        --poll-timeout)
            poll_timeout="$2"; shift 2 ;;
        --max-jobs)
            max_jobs="$2"; shift 2 ;;
        --user)
            user_name="$2"; shift 2 ;;
        -h|--help)
            usage
            exit 0 ;;
        --)
            shift
            break ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "No command specified." >&2
    usage
    exit 1
fi

if [[ ! -f "${JOB_TEMPLATE_PATH}" ]]; then
    echo "Template not found: ${JOB_TEMPLATE_PATH}" >&2
    exit 1
fi

if [[ "${job_name}" =~ [[:space:]] ]]; then
    echo "Job name must not contain whitespace." >&2
    exit 1
fi

if [[ -z "${user_name}" ]]; then
    user_name="${USER:-}"
    if [[ -z "${user_name}" ]]; then
        user_name=$(id -un)
    fi
fi

if [[ -z "${user_name}" ]]; then
    echo "Unable to determine user name for queue counting." >&2
    exit 1
fi

if [[ ! "${max_jobs}" =~ ^[0-9]+$ ]]; then
    echo "--max-jobs must be a positive integer." >&2
    exit 1
fi

max_jobs=$((max_jobs))

if (( max_jobs <= 0 )); then
    echo "--max-jobs must be a positive integer." >&2
    exit 1
fi

command_script=""
command_script_keep=false

normalise_export_value() {
    local name="$1"
    local value="$2"
    local sanitised="${value//[[:space:]]/_}"
    if [[ "${sanitised}" != "${value}" && -n "${value}" ]]; then
        log_event "Normalised ${name} value '${value}' to '${sanitised}' for environment export"
    fi
    printf '%s' "${sanitised}"
}

seed_id=$(normalise_export_value "seed" "${seed_id}")
assay_name=$(normalise_export_value "assay" "${assay_name}")
model_name=$(normalise_export_value "model" "${model_name}")
mf_name=$(normalise_export_value "mf" "${mf_name}")
command_label=$(normalise_export_value "label" "${command_label}")

if [[ "${work_dir}" =~ [[:space:]] ]]; then
    echo "Working directory must not contain whitespace: ${work_dir}" >&2
    exit 1
fi

command_script=$(mktemp "${LOG_DIR}/cmd.XXXXXX.sh")
{
    printf '#!/usr/bin/env bash\n'
    printf 'set -euo pipefail\n'
    printf '%s\n' "$(printf '%q ' "$@" | sed 's/[[:space:]]*$//')"
} > "${command_script}"
chmod 700 "${command_script}"

log_event "Command script created at ${command_script}"

get_current_job_count() {
    local output
    if ! output=$(squeue -u "${user_name}" -h -o "%i" 2>/dev/null); then
        log_event "Failed to query current job count for ${user_name}."
        return 1
    fi
    if [[ -z "${output}" ]]; then
        printf '0'
        return 0
    fi
    awk 'NF { ++count } END { print count + 0 }' <<< "${output}"
}

ensure_job_limit() {
    local context="${1:-}"
    local job_count
    if ! job_count=$(get_current_job_count); then
        log_event "Unable to determine queue utilisation; aborting submission."
        exit 1
    fi
    local message="Current jobs for ${user_name}: ${job_count}/${max_jobs}"
    if [[ -n "${context}" ]]; then
        message+=" (${context})"
    fi
    log_event "${message}"
    if (( job_count >= max_jobs )); then
        log_event "Job limit of ${max_jobs} reached. Aborting submission."
        exit 1
    fi
}

prepare_job_script() {
    local partition="$1"
    local output_path="$2"
    TEMPLATE_PATH="${JOB_TEMPLATE_PATH}" \
    REPL_PARTITION="${partition}" \
    REPL_GPU_COUNT="${gpu_count}" \
    REPL_CPUS="${cpus}" \
    REPL_MEMORY="${memory}" \
    REPL_TIME_LIMIT="${time_limit}" \
    REPL_JOB_NAME="${job_name}" \
    REPL_COMPILER_MODULE="${compiler_module}" \
    REPL_CUDA_MODULE="${cuda_module}" \
    REPL_WORK_DIR="${work_dir}" \
    python - "${output_path}" <<'PY'
import os
import sys
from pathlib import Path

template = Path(os.environ["TEMPLATE_PATH"]).read_text()
replacements = {
    "__PARTITION__": os.environ["REPL_PARTITION"],
    "__GPU_COUNT__": os.environ["REPL_GPU_COUNT"],
    "__CPUS__": os.environ["REPL_CPUS"],
    "__MEMORY__": os.environ["REPL_MEMORY"],
    "__TIME_LIMIT__": os.environ["REPL_TIME_LIMIT"],
    "__JOB_NAME__": os.environ["REPL_JOB_NAME"],
    "__COMPILER_MODULE__": os.environ["REPL_COMPILER_MODULE"],
    "__CUDA_MODULE__": os.environ["REPL_CUDA_MODULE"],
    "__WORK_DIR__": os.environ["REPL_WORK_DIR"],
}
content = template
for key, value in replacements.items():
    content = content.replace(key, value)
Path(sys.argv[1]).write_text(content)
PY
}

poll_job() {
    local job_id="$1"
    local partition="$2"
    local timeout="$3"
    local start_ts
    start_ts=$(date +%s)
    while true; do
        local status_line
        status_line=$(squeue -j "${job_id}" -h -o "%T|%R" 2>/dev/null || true)
        if [[ -z "${status_line}" ]]; then
            log_event "Job ${job_id} no longer appears in squeue."
            return 2
        fi
        local state reason
        IFS='|' read -r state reason <<< "${status_line}"
        log_event "Job ${job_id} poll: state=${state} reason=${reason}"
        if [[ -n "${reason}" && "${reason}" != \(* ]]; then
            log_event "Job ${job_id} allocated on ${reason}"
            return 0
        fi
        if (( timeout > 0 )); then
            local now elapsed
            now=$(date +%s)
            elapsed=$(( now - start_ts ))
            if (( elapsed >= timeout )); then
                log_event "Job ${job_id} timed out on partition ${partition}"
                return 1
            fi
        fi
        sleep "${poll_interval}"
    done
}

submit_to_partition() {
    local partition="$1"
    local timeout="$2"
<<<<<<< Updated upstream
    local leave_queued="${3:-false}"
=======
    local wait_mode="${3:-poll}"
    ensure_job_limit "before submitting to ${partition}"
>>>>>>> Stashed changes
    local job_script
    job_script=$(mktemp "${REPO_ROOT}/gpu_job_logs/.gpu_job.XXXXXX")
    cleanup_files+=("${job_script}")
    prepare_job_script "${partition}" "${job_script}"

    local extra_modules_joined=""
    if [[ ${#extra_modules[@]} -gt 0 ]]; then
        extra_modules_joined=$(IFS=','; echo "${extra_modules[*]}")
    fi

    local export_vars="ALL,COMMAND_FILE=${command_script},WORK_DIR=${work_dir},SEED_ID=${seed_id},ASSAY_NAME=${assay_name},MODEL_NAME=${model_name},MF_NAME=${mf_name},PERF_LOG_LABEL=${command_label},EXTRA_MODULES=${extra_modules_joined}"

    if [[ "${leave_queued}" == true ]]; then
        log_event "Submitting job to partition ${partition} without polling (leaving queued)"
        local job_id
        if ! job_id=$(sbatch --parsable --export="${export_vars}" "${job_script}"); then
            log_event "Failed to submit job to ${partition} when leaving queued"
            return 1
        fi
        log_event "Submitted job ${job_id} to ${partition} (queued)"
        echo "${job_id}"
        return 0
    fi

    log_event "Submitting job to partition ${partition}"
    local job_id
    if ! job_id=$(sbatch --parsable --export="${export_vars}" "${job_script}"); then
        log_event "Failed to submit job to ${partition}"
        return 1
    fi
    log_event "Submitted job ${job_id} to ${partition}"

    if [[ "${wait_mode}" == "no-wait" ]]; then
        echo "${job_id}"
        return 0
    fi

    local poll_result
    poll_job "${job_id}" "${partition}" "${timeout}"
    poll_result=$?
    case "${poll_result}" in
        0)
            log_event "Job ${job_id} successfully allocated on ${partition}"
            echo "${job_id}"
            return 0 ;;
        1)
            log_event "Cancelling job ${job_id} on ${partition} after timeout"
            scancel "${job_id}" >/dev/null 2>&1 || true
            return 1 ;;
        *)
            log_event "Job ${job_id} ended before allocation on ${partition}"
            return 1 ;;
    esac
}

partitions=(gpu1 gpu6 gpu2 gpu3 gpu4 gpu5)
job_allocated=""

for partition in "${partitions[@]}"; do
    if job_id=$(submit_to_partition "${partition}" "${poll_timeout}"); then
        job_allocated="${job_id}"
        command_script_keep=true
        break
    fi
    log_event "Partition ${partition} attempt finished without allocation."
    sleep 1
done

if [[ -z "${job_allocated}" ]]; then
    log_event "No allocation after full rotation. Submitting to gpu1 and leaving job queued."
<<<<<<< Updated upstream
    if job_id=$(submit_to_partition "gpu1" 0 true); then
=======
    if job_id=$(submit_to_partition "gpu1" "${poll_timeout}" "no-wait"); then
>>>>>>> Stashed changes
        job_allocated="${job_id}"
        command_script_keep=true
    else
        log_event "Submission to gpu1 failed when leaving job queued."
        exit 1
    fi
fi

log_event "Final submitted job ID: ${job_allocated}"
exit 0
