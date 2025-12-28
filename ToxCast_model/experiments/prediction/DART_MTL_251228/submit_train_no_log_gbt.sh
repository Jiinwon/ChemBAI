#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER:-won0316}"

TOXCAST_MODEL_ROOT="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model"
TRAIN_ROOT="${TOXCAST_MODEL_ROOT}/experiments/training/DART_MTL"
RESULTS_ROOT="${TRAIN_ROOT}/results"
DATA_ROOT="${TRAIN_ROOT}/data"

# 저장 위치(backup)
BACKUP_ROOT="${RESULTS_ROOT}/backup"
BACKUP_LOG_DIR="${BACKUP_ROOT}/_logs_no_log_train"

# 이번 케이스는 seed_1, seed_2만
SEEDS=(2)

# 제출 제한(기존 그대로)
MAX_TOTAL_JOBS=20
MAX_RUNNING_JOBS=10

# 파티션 순서(기존 그대로)
PARTITIONS=(gpu1 gpu6 gpu4 gpu2 gpu3 gpu5)

# 대상 조합(네가 준 리스트)
COMBOS=(
  "TOX21_ERa_BLA_Antagonist_ratio_gbt_Pattern"
  "TOX21_p53_BLA_p2_ratio_gbt_Pattern"
  "TOX21_PGC_ERR_Antagonist_gbt_Pattern"
  "TOX21_ERb_BLA_Antagonist_ratio_gbt_Pattern"
  "TOX21_PR_BLA_Antagonist_ratio_gbt_Pattern"
  "TOX21_ERa_LUC_VM7_Agonist_10nM_ICI182780_gbt_Pattern"
)

mkdir -p "${BACKUP_LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train_no_log_precomputed_fp_gbt.py"
TASKS_TSV="${SCRIPT_DIR}/tasks_train_no_log_gbt.tsv"

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

# tasks 생성
: > "${TASKS_TSV}"
echo -e "seed\tassay\tmodel\tfp\ttrain_csv\tfp_dir\torig_model\tout_model" >> "${TASKS_TSV}"

for seed in "${SEEDS[@]}"; do
  train_csv="${DATA_ROOT}/seed_${seed}/train_df.csv"
  fp_dir="${DATA_ROOT}/seed_${seed}/fingerprints/train"

  [[ -s "${train_csv}" ]] || continue
  [[ -d "${fp_dir}" ]] || continue

  for combo in "${COMBOS[@]}"; do
    # combo: {assay}_{model}_{fp}  (assay에는 _가 많을 수 있음)
    fp="${combo##*_}"
    rest="${combo%_*}"
    model="${rest##*_}"
    assay="${rest%_*}"

    # gbt만 의도된 케이스
    [[ "${model}" == "gbt" ]] || continue
    [[ "${fp}" == "Pattern" ]] || continue

    # FP 파일 존재 체크(없으면 학습 불가)
    if [[ ! -s "${fp_dir}/${fp}.csv" ]]; then
      # 대소문자/변형 대비
      if ! ls "${fp_dir}"/*.csv 1>/dev/null 2>&1; then
        continue
      fi
      if ! ls "${fp_dir}"/*.csv | grep -qi "${fp}"; then
        continue
      fi
    fi

    orig_model="${RESULTS_ROOT}/seed_${seed}/${assay}_${model}_${fp}/model.joblib"
    out_model="${BACKUP_ROOT}/seed_${seed}/${assay}_${model}_${fp}/model.joblib"

    # 이미 있으면 스킵
    [[ -s "${orig_model}" ]] && continue
    [[ -s "${out_model}"  ]] && continue

    echo -e "${seed}\t${assay}\t${model}\t${fp}\t${train_csv}\t${fp_dir}\t${orig_model}\t${out_model}" >> "${TASKS_TSV}"
  done
done

echo "[INFO] tasks file: ${TASKS_TSV}"
echo "[INFO] tasks count: $(( $(wc -l < "${TASKS_TSV}") - 1 ))"

# 제출
tail -n +2 "${TASKS_TSV}" | while IFS=$'\t' read -r seed assay model fp train_csv fp_dir orig_model out_model; do
  wait_for_quota
  part="$(pick_partition)"

  job_name="TRAIN_NOLOG_s${seed}_${model}_${fp}_${assay}"
  job_name="${job_name:0:180}"

  out_log="${BACKUP_LOG_DIR}/${job_name}_%j.out"
  err_log="${BACKUP_LOG_DIR}/${job_name}_%j.err"

  echo "[SUBMIT] part=${part} seed=${seed} assay=${assay} model=${model} fp=${fp}"

  sbatch \
    --partition="${part}" \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=24G \
    --time=18:00:00 \
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

export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-1}"

python -u "${TRAIN_PY}" \
  --seed "${seed}" \
  --assay "${assay}" \
  --fp "${fp}" \
  --train_csv "${train_csv}" \
  --fp_dir "${fp_dir}" \
  --out "${out_model}" \
  --grid fast \
  --cv 5 \
  --n_jobs -1
EOF

done

echo "[DONE] submit loop finished."
echo "[HINT] backup models: ${BACKUP_ROOT}/seed_*/*/model.joblib"
echo "[HINT] job logs: ${BACKUP_LOG_DIR}"
