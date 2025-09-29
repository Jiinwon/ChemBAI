#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERR] line:$LINENO cmd:${BASH_COMMAND}" >&2' ERR

# 사용자 설정
PY_SCRIPT="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/slurm/submit_v3_training.py"
LOG_DIR="/home1/won0316/_RESEARCH/0817_Genotoxicity/1_Git_upload/ChemBAI/slurm_logs"
LOG_FILE="$LOG_DIR/submit_v3_training.log"

# 로그 디렉토리 보장
mkdir -p "$LOG_DIR"

# nohup 실행 (stdout + stderr 같이 로그파일에 기록)
nohup python "$PY_SCRIPT" > "$LOG_FILE" 2>&1 &

echo "Submitted $PY_SCRIPT via nohup, logging to $LOG_FILE"
