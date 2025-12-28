#!/usr/bin/env bash
set -u

###############################################################################
# 설정
###############################################################################
BASE_DIR="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/ToxCast_model/experiments/prediction/DART_MTL_251228"

PRED_PY="predict_dart_mtl_wide_cachedfp_probpred.py"
MERGE_PY="merge_dart_mtl_scores.py"
OUT_XLSX="${BASE_DIR}/merged_scores_all.xlsx"

SEEDS=(0 1 2)
SPLITS=(train val)

# 10분 대기(요구)
WAIT_AFTER_PRED_SEC=600

# 한 번에 여러 개 켜둘 때 로그 파일명(요구대로 고정)
# (매 회 overwrite됨. 누적하고 싶으면 ">"를 ">>"로 바꾸면 됨)

# 중복 실행 방지 락(같은 스크립트를 2번 켜도 하나만 돌아가게)
LOCKDIR="${BASE_DIR}/.hourly_run_lock"

###############################################################################
# 유틸
###############################################################################
ts() { date "+%F %T"; }

run_one_cycle() {
  echo "[$(ts)] [CYCLE] start"

  cd "${BASE_DIR}" || exit 1

  # 1) 예측 nohup 백그라운드 실행
  for seed in "${SEEDS[@]}"; do
    for split in "${SPLITS[@]}"; do
      echo "[$(ts)] [PRED] seed=${seed} split=${split}"
      nohup python -u "${PRED_PY}" --seed "${seed}" --split "${split}" \
        > "nohup_pred_seed${seed}_${split}.log" 2>&1 &
      sleep 1
    done
  done

  # 2) 10분 대기
  echo "[$(ts)] [WAIT] ${WAIT_AFTER_PRED_SEC}s"
  sleep "${WAIT_AFTER_PRED_SEC}"

  # 3) merge 실행
  echo "[$(ts)] [MERGE] start"
  if ! python -u "${MERGE_PY}" \
      --base_dir "${BASE_DIR}" \
      --out_xlsx "${OUT_XLSX}"; then
    echo "[$(ts)] [MERGE] failed (skip git step)"
    return 0
  fi
  echo "[$(ts)] [MERGE] done -> ${OUT_XLSX}"

  # 4) git add/commit/push
  #    변경 없으면 commit/push 스킵
  echo "[$(ts)] [GIT] add/commit/push"
  git add -A

  if [[ -z "$(git status --porcelain)" ]]; then
    echo "[$(ts)] [GIT] no changes -> skip commit/push"
    return 0
  fi

  msg="auto: update merged_scores_all $(date '+%F %T')"
  if git commit -m "${msg}"; then
    if git push; then
      echo "[$(ts)] [GIT] push OK"
    else
      echo "[$(ts)] [GIT] push FAILED"
    fi
  else
    echo "[$(ts)] [GIT] commit FAILED"
  fi
}

###############################################################################
# 메인
###############################################################################
# 락 획득 (이미 돌고 있으면 종료)
if ! mkdir "${LOCKDIR}" 2>/dev/null; then
  echo "[$(ts)] [LOCK] already running: ${LOCKDIR}"
  exit 0
fi
trap 'rmdir "${LOCKDIR}" 2>/dev/null || true' EXIT

# conda 활성화(필요하면)
# 이미 toxcast_env가 활성화된 상태로 실행할 거면 아래 2줄은 주석 처리해도 됨.
source /home1/won0316/anaconda3/etc/profile.d/conda.sh
conda activate toxcast_env

echo "[$(ts)] [START] hourly loop begin (BASE_DIR=${BASE_DIR})"

while true; do
  run_one_cycle

  # "한시간에 한번씩"을 정각 기준으로 맞추기:
  now="$(date +%s)"
  sleep_sec=$((3600 - (now % 3600)))
  # 딱 정각에 이미 끝났으면 3600초로
  if [[ "${sleep_sec}" -eq 0 ]]; then
    sleep_sec=3600
  fi
  echo "[$(ts)] [SLEEP] ${sleep_sec}s (to next hour)"
  sleep "${sleep_sec}"
done
