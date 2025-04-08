#!/bin/bash

pwd
# 모델 리스트 정의
models=('dt') # 'xgb' 'gbt' 'rf' 'logistic') # 모델 정의
fingerprints=('MACCS') # 'Morgan' 'RDKit' 'Layered' 'Pattern') # 지문 정의
file_path="./data/example_data_DBPs/for_train/example_DBPs_ER.xlsx" # 데이터 경로
fp_path="./data/example_data_DBPs/for_train/fingerprint_outputs" # fingerprints 경로
data_name="example_DBPs_ER"

current_date=$(date +%Y-%m-%d)

# 동시에 실행할 작업의 최대 수
max_jobs=45
current_jobs=0

# 각 모델에 대해 실험 실행
for assay_num in {1..3}; do # 반복 범위 변경 가능

    # assay_name 로드
    assay_name=$(python -c "import pandas as pd; df = pd.read_excel('${file_path}', header=None); print(df.iloc[0, int('${assay_num}') + 1])")

    # 로그 디렉토리 생성
    mkdir -p "./logs/${data_name}/$current_date/${assay_name}"
    for model in "${models[@]}"; do
        for fingerprint in "${fingerprints[@]}"; do
            # 결과 저장 디렉토리 생성
            mkdir -p ./results/${data_name}/$current_date/model_save_path/${assay_name}/${assay_name}_${fingerprint}_${model}
            model_save_path="./results/${data_name}/$current_date/model_save_path/${assay_name}/${assay_name}_${fingerprint}_${model}"

            echo "[$(date)]   Submitting job for assay_num: $assay_name, model: $model with fingerprint: $fingerprint"
            echo "[$(date)]   Running ${assay_name}/${assay_name}_${fingerprint}_${model}"

            # Python 스크립트를 백그라운드에서 실행
            python ./run/${model}.py \
                --fingerprint_type ${fingerprint} \
                --file_path ${file_path} \
                --model_save_path ${model_save_path} \
                --assay_num $((assay_num)) \
                --fp_path ${fp_path} \
                > ./logs/${data_name}/$current_date/${assay_name}/${assay_name}_${fingerprint}_${model}.log \
                2> ./logs/${data_name}/$current_date/${assay_name}/${assay_name}_${fingerprint}_${model}.err &

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
