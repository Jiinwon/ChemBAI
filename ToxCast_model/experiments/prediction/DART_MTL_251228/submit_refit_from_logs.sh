#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 설정
###############################################################################
USER_NAME="${USER:-won0316}"

TOXCAST_MODEL_ROOT="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model"
TRAIN_ROOT="${TOXCAST_MODEL_ROOT}/experiments/training/DART_MTL"
RESULTS_ROOT="${TRAIN_ROOT}/results"
LOG_DIR="${TRAIN_ROOT}/logs/seed_multi"

# seed별 데이터 위치
DATA_ROOT="${TRAIN_ROOT}/data"   # .../data/seed_0/train_df.csv, .../data/seed_0/fingerprints/train/*.csv

# 복구 결과 저장(요구)
BACKUP_ROOT="${RESULTS_ROOT}/backup"
BACKUP_LOG_DIR="${BACKUP_ROOT}/_logs"

# seed 범위(요구)
SEEDS=(0 1 2)

# 제출 제한(요구)
MAX_TOTAL_JOBS=20
MAX_RUNNING_JOBS=10

# 파티션 순서(요구)
PARTITIONS=(gpu1 gpu6 gpu4 gpu2 gpu3 gpu5)

###############################################################################
# (중요) model/fp 필터 토글
#  - 기본값(0): 전체 조합(로그에 있는 모든 model/fp) 제출
#  - 1로 바꾸면: 아래 ALLOWED_* 목록만 제출
###############################################################################
USE_MODEL_FP_FILTER=1   # <- 필요할 때만 1로 바꿔서 켜기

# 필터를 켰을 때만 적용될 허용 목록
ALLOWED_MODELS=(gbt logistic)
ALLOWED_FPS=(Layered Pattern RDKit)

# 누락된 어세이(요구)
MISSING_ASSAYS=(
  #"BSK_hDFCGF_TIMP1"
  #"BSK_hDFCGF_VCAM1"
  #"BSK_KF3CT_TIMP2"
  #"NVS_ENZ_hVEGFR3"
  #"CEETOX_H295R_OHPROG"
  #"CEETOX_H295R_ESTRADIOL"
  #"CEETOX_H295R_ESTRONE"
  #"CEETOX_H295R_PROG"
  #"TOX21_ERa_BLA_Antagonist_ratio"
  #"TOX21_p53_BLA_p2_ratio"
  #"TOX21_PGC_ERR_Antagonist"
  #"TOX21_ERb_BLA_Antagonist_ratio"
  #"TOX21_PR_BLA_Antagonist_ratio"
  #"TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780"

  "ATG_Ahr_CIS"
  "ATG_BRE_CIS"
  "ATG_DR5_RAR_CIS"
  "ATG_ERE_CIS"
  "ATG_HIF1a_CIS"
  "BSK_3C_uPAR"
  "BSK_4H_MCP1"
  "BSK_BE3C_PAI1"
  "BSK_CASM3C_uPAR"
  "BSK_hDFCGF_PAI1"
)

###############################################################################
# 경로 준비
###############################################################################
mkdir -p "${BACKUP_LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFIT_PY="${SCRIPT_DIR}/refit_from_log_params_precomputed_fp.py"
TASKS_TSV="${SCRIPT_DIR}/tasks_refit_missing_precomputed_fp.tsv"

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
# 1) tasks 생성
#    - 로그 존재
#    - 해당 seed 블록 존재
#    - train_df.csv 존재
#    - fingerprints/train/{FP}.csv 존재
#    - orig 모델 없고, backup에도 없을 때만
###############################################################################
: > "${TASKS_TSV}"
echo -e "seed\tassay\tmodel\tfp\tlog_path\tout_path\ttrain_csv\tfp_dir" >> "${TASKS_TSV}"

echo "[INFO] USE_MODEL_FP_FILTER=${USE_MODEL_FP_FILTER}"
if [[ "${USE_MODEL_FP_FILTER}" -eq 1 ]]; then
  echo "[INFO] ALLOWED_MODELS=(${ALLOWED_MODELS[*]})"
  echo "[INFO] ALLOWED_FPS=(${ALLOWED_FPS[*]})"
else
  echo "[INFO] model/fp filter OFF (all combos found in logs will be considered)"
fi

for assay in "${MISSING_ASSAYS[@]}"; do
  shopt -s nullglob
  for lf in "${LOG_DIR}/multi_${assay}_"*.log; do
    base="$(basename "${lf}")"
    base="${base#multi_}"
    base="${base%.log}"

    # 파일명 규칙: multi_{assay}_{model}_{fp}.log
    fp="${base##*_}"
    rest="${base%_*}"
    model="${rest##*_}"
    assay_from_name="${rest%_*}"

    [[ "${assay_from_name}" == "${assay}" ]] || continue

    # -------------------------
    # (추가) model/fp 필터 ON일 때만 적용
    # -------------------------
    if [[ "${USE_MODEL_FP_FILTER}" -eq 1 ]]; then
      in_list "${model}" "${ALLOWED_MODELS[@]}" || continue
      in_list "${fp}"    "${ALLOWED_FPS[@]}"    || continue
    fi

    for seed in "${SEEDS[@]}"; do
      # 로그에 해당 seed 블록 없으면 스킵
      grep -q "START seed_${seed}:" "${lf}" || continue

      train_csv="${DATA_ROOT}/seed_${seed}/train_df.csv"
      fp_dir="${DATA_ROOT}/seed_${seed}/fingerprints/train"

      # 데이터/FP 없으면 스킵
      [[ -s "${train_csv}" ]] || continue
      [[ -d "${fp_dir}" ]] || continue

      # FP가 csv로 저장되어 있으므로 {FP}.csv 존재 체크
      # (없으면 제출 자체를 안 해서 FileNotFoundError 방지)
      if [[ ! -s "${fp_dir}/${fp}.csv" ]]; then
        # 혹시 대소문자/접미어 차이 대응(있으면 통과)
        if ! ls "${fp_dir}"/*.csv 1>/dev/null 2>&1; then
          continue
        fi
        if ! ls "${fp_dir}"/*.csv | grep -qi "${fp}"; then
          continue
        fi
      fi

      orig="${RESULTS_ROOT}/seed_${seed}/${assay}_${model}_${fp}/model.joblib"
      out="${BACKUP_ROOT}/seed_${seed}/${assay}_${model}_${fp}/model.joblib"

      [[ -s "${orig}" ]] && continue
      [[ -s "${out}"  ]] && continue

      echo -e "${seed}\t${assay}\t${model}\t${fp}\t${lf}\t${out}\t${train_csv}\t${fp_dir}" >> "${TASKS_TSV}"
    done
  done
  shopt -u nullglob
done

echo "[INFO] tasks file: ${TASKS_TSV}"
echo "[INFO] tasks count: $(( $(wc -l < "${TASKS_TSV}") - 1 ))"

###############################################################################
# 2) 제출 제한 + 파티션 선택
###############################################################################
get_total_jobs() { squeue -u "${USER_NAME}" -h | wc -l | tr -d ' '; }
get_running_jobs() { squeue -u "${USER_NAME}" -h -t R | wc -l | tr -d ' '; }

wait_for_quota() {
  while true; do
    total="$(get_total_jobs)"
    running="$(get_running_jobs)"
    if [[ "${total}" -lt "${MAX_TOTAL_JOBS}" && "${running}" -lt "${MAX_RUNNING_JOBS}" ]]; then
      return 0
    fi
    sleep 2
  done
}

pick_partition() {
  # 요구: gpu1 -> gpu6 -> gpu4 -> gpu2 -> gpu3 -> gpu5 순회
  # 각 체크 사이 2초 대기, (none)이면 다음 파티션, 다 none이면 gpu1부터 반복
  while true; do
    for P in "${PARTITIONS[@]}"; do
      sleep 2
      nodelist="$(sinfo -h -p "${P}" -t idle -o "%N" 2>/dev/null | head -n 1 || true)"
      if [[ -n "${nodelist}" && "${nodelist}" != "(none)" ]]; then
        echo "${P}"
        return 0
      fi
    done
  done
}

###############################################################################
# 3) 작업 제출
###############################################################################
echo "[INFO] submitting jobs..."

tail -n +2 "${TASKS_TSV}" | while IFS=$'\t' read -r seed assay model fp log_path out_path train_csv fp_dir; do
  wait_for_quota
  part="$(pick_partition)"

  job_name="REFIT_s${seed}_${model}_${fp}_${assay}"
  job_name="${job_name:0:180}"

  out_log="${BACKUP_LOG_DIR}/${job_name}_%j.out"
  err_log="${BACKUP_LOG_DIR}/${job_name}_%j.err"

  echo "[SUBMIT] part=${part} seed=${seed} assay=${assay} model=${model} fp=${fp}"

  sbatch \
    --partition="${part}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=24G \
    --time=02:00:00 \
    --job-name="${job_name}" \
    -o "${out_log}" \
    -e "${err_log}" \
    --export=ALL \
    <<EOF
#!/usr/bin/env bash
set -euo pipefail

# /etc/bashrc unbound 변수 이슈 회피: ~/.bashrc source 하지 않음
source /home1/won0316/anaconda3/etc/profile.d/conda.sh
conda activate toxcast_env

python -u "${REFIT_PY}" \
  --seed "${seed}" \
  --assay "${assay}" \
  --model "${model}" \
  --fp "${fp}" \
  --log "${log_path}" \
  --out "${out_path}" \
  --train_csv "${train_csv}" \
  --fp_dir "${fp_dir}"
EOF

done

echo "[DONE] submit loop finished."
echo "[HINT] backup models: ${BACKUP_ROOT}/seed_*/*/model.joblib"
echo "[HINT] job logs: ${BACKUP_LOG_DIR}"
