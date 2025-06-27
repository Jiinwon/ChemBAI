#!/bin/bash

pwd
# config.py에서 변수 읽기
models=($(python - <<'EOF'
import config
print(' '.join(config.MODELS))
EOF
))
fingerprints=($(python - <<'EOF'
import config
print(' '.join(config.FINGERPRINTS))
EOF
))
file_path=$(python - <<'EOF'
import config, os
from toxcast_pkg.common import find_single_excel_file
p = config.TRAIN_FILE_PATH
if os.path.isdir(p):
    p = find_single_excel_file(p)
print(p)
EOF
)
fp_path=$(python - <<'EOF'
import config
print(config.TRAIN_FP_PATH)
EOF
)
data_name=$(python - <<'EOF'
import config
print(config.DATA_NAME)
EOF
)

results_dir=$(python - <<'EOF'
import config
print(config.RESULTS_DIR)
EOF
)
logs_dir=$(python - <<'EOF'
import config
from pathlib import Path
print((config.BASE_DIR / 'logs').as_posix())
EOF
)

# deprecated but kept for compatibility
current_date=$(date +%Y-%m-%d)

# 동시에 실행할 작업의 최대 수
max_jobs=45
current_jobs=0

# 각 모델에 대해 실험 실행
for assay_num in {1..3}; do # 반복 범위 변경 가능

    # assay_name 로드
    assay_name=$(python -c "import pandas as pd; df = pd.read_excel('${file_path}', header=None); print(df.iloc[0, int('${assay_num}') + 1])")

    # 로그 디렉토리 생성
    mkdir -p "${logs_dir}/${assay_name}"
    for model in "${models[@]}"; do
        for fingerprint in "${fingerprints[@]}"; do
            # 결과 저장 디렉토리 생성
            mkdir -p "${results_dir}/model_save_path/${assay_name}/${assay_name}_${fingerprint}_${model}"
            model_save_path="${results_dir}/model_save_path/${assay_name}/${assay_name}_${fingerprint}_${model}"

            echo "[$(date)]   Submitting job for assay_num: $assay_name, model: $model with fingerprint: $fingerprint"
            echo "[$(date)]   Running ${assay_name}/${assay_name}_${fingerprint}_${model}"

            # Python 스크립트를 백그라운드에서 실행
            python ./run/${model}.py \
                --fingerprint_type ${fingerprint} \
                --file_path ${file_path} \
                --model_save_path ${model_save_path} \
                --assay_num $((assay_num)) \
                --fp_path ${fp_path} \
                > "${logs_dir}/${assay_name}/${assay_name}_${fingerprint}_${model}.log" \
                2> "${logs_dir}/${assay_name}/${assay_name}_${fingerprint}_${model}.err" &

            # 작업 관리
            ((current_jobs++))
            if (( current_jobs >= max_jobs )); then
                # 최대 작업 수에 도달하면 대기
                wait
                current_jobs=0
            fi
        done
    done
done

# 모든 작업 종료 대기
wait
