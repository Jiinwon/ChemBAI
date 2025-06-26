#!/usr/bin/env bash
set -euo pipefail

# 1) Create data & fingerprint and results directories
#    기존 ChemBAI 리포지토리 클론 단계는 로컬에 이미 있으므로 생략
mkdir -p "${CHEMBAI_FP_PATH}" "${CHEMBAI_RESULTS_PATH}"  # 분자지문 및 결과 저장 디렉터리 확보

# 2) Generate molecular fingerprints if directory is empty
if [ -d "${CHEMBAI_FP_PATH}" ] && [ -z "$(ls -A "${CHEMBAI_FP_PATH}")" ]; then
  echo "Generating fingerprints from ${CHEMBAI_INPUT_PATH} to ${CHEMBAI_FP_PATH}..."
  python3 smiles2fing.py \
    --input "${CHEMBAI_INPUT_PATH}" \
    --output "${CHEMBAI_FP_PATH}"  # 입력 SMILES로부터 지문 생성
fi

# 3) Optional: run model training
#    필요한 경우 아래 주석을 해제하여 자동으로 학습 스크립트를 실행하세요
# bash train_model.sh
