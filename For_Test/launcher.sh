#!/usr/bin/env bash
#SBATCH --job-name=ChemBAI         # 작업 이름
#SBATCH --partition=gpu1               # 기본 파티션
#SBATCH --gres=gpu          # GPU 리소스
#SBATCH --cpus-per-task=8              # CPU 코어 수
#SBATCH --mem=16G                      # 메모리
#SBATCH --time=02:00:00                # 최대 실행 시간
#SBATCH --output=logs/%x_%j.out        # 로그 디렉토리, %x=job-name, %j=jobid
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# 1) Conda 환경 활성화
source activate toxcast_env

# 2) 파이프라인 실행
python run_pipeline.sh