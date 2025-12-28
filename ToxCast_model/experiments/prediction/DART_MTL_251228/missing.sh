#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 설정
###############################################################################
USER_NAME="${USER:-won0316}"

TOXCAST_MODEL_ROOT="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model"
TRAIN_ROOT="${TOXCAST_MODEL_ROOT}/experiments/training/DART_MTL"
RESULTS_ROOT="${TRAIN_ROOT}/results"

# ✅ seed별 로그 루트(요구사항)
LOG_ROOT="${TRAIN_ROOT}/logs"   # .../logs/seed_0/<ASSAY>/<ASSAY>_<MODEL>_<FP>.log

# seed별 데이터 위치
DATA_ROOT="${TRAIN_ROOT}/data"  # .../data/seed_0/train_df.csv, .../data/seed_0/fingerprints/train/*.csv

# 복구 결과 저장
BACKUP_ROOT="${RESULTS_ROOT}/backup"
BACKUP_LOG_DIR="${BACKUP_ROOT}/_logs_local"

# seed 범위(원하면 수정)
SEEDS=(0 1 2)

# ✅ 이번에 수행할 assay 2개만
ASSAYS=(
  "ATG_Ahr_CIS"
  "ATG_BRE_CIS"
)

###############################################################################
# (옵션) model/fp 필터 토글
#  - 0: 해당 로그 폴더에 있는 모든 model/fp 조합 수행
#  - 1: 아래 ALLOWED_* 만 수행
###############################################################################
USE_MODEL_FP_FILTER=1

# 필터 ON일 때만 적용될 허용 목록(원하는대로 수정)
ALLOWED_MODELS=(gbt logistic)
ALLOWED_FPS=(Layered Pattern RDKit)

###############################################################################
# (옵션) 병렬 실행 개수
#  - 1이면 완전 순차 실행
#  - 로그인 노드/자원 상황에 따라 조절
###############################################################################
MAX_PARALLEL=10

###############################################################################
# 경로 준비
###############################################################################
mkdir -p "${BACKUP_LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFIT_PY="${SCRIPT_DIR}/refit_from_log_params_precomputed_fp.py"
TASKS_TSV="${SCRIPT_DIR}/tasks_refit_local.tsv"

###############################################################################
# conda 활성화 (bashrc 미사용)
###############################################################################
source /home1/won0316/anaconda3/etc/profile.d/conda.sh
conda activate toxcast_env

###############################################################################
# 유틸: 배열 포함 여부(정확 match)
###############################################################################
in_list() {
  local x="$1"; shift
  local arr=("$@")
  local a
  for a in "${arr[@]}"; do
    [[ "$a" == "$x" ]] && return 0
  done
  return 1
}

###############################################################################
# 1) tasks 생성: seed별 로그 폴더에서 {assay}_{model}_{fp}.log를 찾는다
###############################################################################
: > "${TASKS_TSV}"
echo -e "seed\tassay\tmodel\tfp\tlog_path\tout_path\ttrain_csv\tfp_dir" >> "${TASKS_TSV}"

echo "[INFO] USE_MODEL_FP_FILTER=${USE_MODEL_FP_FILTER}"
if [[ "${USE_MODEL_FP_FILTER}" -eq 1 ]]; then
  echo "[INFO] ALLOWED_MODELS=(${ALLOWED_MODELS[*]})"
  echo "[INFO] ALLOWED_FPS=(${ALLOWED_FPS[*]})"
fi
echo "[INFO] MAX_PARALLEL=${MAX_PARALLEL}"
echo "[INFO] ASSAYS=(${ASSAYS[*]})"
echo "[INFO] SEEDS=(${SEEDS[*]})"

for seed in "${SEEDS[@]}"; do
  train_csv="${DATA_ROOT}/seed_${seed}/train_df.csv"
  fp_dir="${DATA_ROOT}/seed_${seed}/fingerprints/train"

  [[ -s "${train_csv}" ]] || { echo "[WARN] missing train_csv: ${train_csv}"; continue; }
  [[ -d "${fp_dir}" ]]    || { echo "[WARN] missing fp_dir: ${fp_dir}"; continue; }

  for assay in "${ASSAYS[@]}"; do
    log_assay_dir="${LOG_ROOT}/seed_${seed}/${assay}"
    [[ -d "${log_assay_dir}" ]] || { echo "[WARN] missing log dir: ${log_assay_dir}"; continue; }

    shopt -s nullglob
    for lf in "${log_assay_dir}/${assay}_"*.log; do
      base="$(basename "${lf}")"
      base="${base%.log}"

      # 파일명 규칙: {assay}_{model}_{fp}.log
      fp="${base##*_}"
      rest="${base%_*}"
      model="${rest##*_}"
      assay_from_name="${rest%_*}"

      [[ "${assay_from_name}" == "${assay}" ]] || continue

      # 로그가 최소한 파라미터 라인을 포함하는지 체크
      grep -q "Best Model Parameters" "${lf}" || continue

      # (옵션) model/fp 필터
      if [[ "${USE_MODEL_FP_FILTER}" -eq 1 ]]; then
        in_list "${model}" "${ALLOWED_MODELS[@]}" || continue
        in_list "${fp}"    "${ALLOWED_FPS[@]}"    || continue
      fi

      # FP csv 존재 체크(없으면 스킵)
      if [[ ! -s "${fp_dir}/${fp}.csv" ]]; then
        if ! ls "${fp_dir}"/*.csv 1>/dev/null 2>&1; then
          continue
        fi
        if ! ls "${fp_dir}"/*.csv | grep -qi "${fp}"; then
          continue
        fi
      fi

      orig="${RESULTS_ROOT}/seed_${seed}/${assay}_${model}_${fp}/model.joblib"
      out="${BACKUP_ROOT}/seed_${seed}/${assay}_${model}_${fp}/model.joblib"

      # 이미 있으면 스킵(원본 results 또는 backup)
      [[ -s "${orig}" ]] && continue
      [[ -s "${out}"  ]] && continue

      echo -e "${seed}\t${assay}\t${model}\t${fp}\t${lf}\t${out}\t${train_csv}\t${fp_dir}" >> "${TASKS_TSV}"
    done
    shopt -u nullglob
  done
done

TASKS_COUNT=$(( $(wc -l < "${TASKS_TSV}") - 1 ))
echo "[INFO] tasks file : ${TASKS_TSV}"
echo "[INFO] tasks count: ${TASKS_COUNT}"

if [[ "${TASKS_COUNT}" -le 0 ]]; then
  echo "[DONE] nothing to run."
  exit 0
fi

###############################################################################
# 2) tasks 실행 (sbatch 없이 직접 python 실행)
#    - 각 작업 로그를 BACKUP_LOG_DIR 아래에 남김
#    - MAX_PARALLEL로 동시 실행 개수 제한
###############################################################################
running=0
fail=0

tail -n +2 "${TASKS_TSV}" | while IFS=$'\t' read -r seed assay model fp log_path out_path train_csv fp_dir; do
  run_tag="s${seed}_${assay}_${model}_${fp}"
  run_log="${BACKUP_LOG_DIR}/refit_${run_tag}.log"

  echo "[RUN] ${run_tag}"
  echo "      log=${log_path}"
  echo "      out=${out_path}"
  echo "      -> ${run_log}"

  # 백그라운드 실행
  python -u "${REFIT_PY}" \
    --seed "${seed}" \
    --assay "${assay}" \
    --model "${model}" \
    --fp "${fp}" \
    --log "${log_path}" \
    --out "${out_path}" \
    --train_csv "${train_csv}" \
    --fp_dir "${fp_dir}" \
    > "${run_log}" 2>&1 &

  running=$((running + 1))

  # 동시 실행 제한
  if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    # 하나 끝날 때까지 대기
    if ! wait -n; then
      fail=$((fail + 1))
    fi
    running=$((running - 1))
  fi
done

# 남은 것 모두 대기
while [[ "${running}" -gt 0 ]]; do
  if ! wait -n; then
    fail=$((fail + 1))
  fi
  running=$((running - 1))
done

if [[ "${fail}" -gt 0 ]]; then
  echo "[DONE] finished with failures: ${fail}"
  echo "[HINT] check logs in: ${BACKUP_LOG_DIR}"
  exit 1
fi

echo "[DONE] all tasks completed successfully."
echo "[HINT] backup models: ${BACKUP_ROOT}/seed_*/*/model.joblib"
echo "[HINT] run logs     : ${BACKUP_LOG_DIR}"
