#!/bin/bash

set -euo pipefail

pwd

# 공통 설정 불러오기
read -r version run_dir <<<"$(python - <<'EOF'
import config
version = getattr(config, "VERSION", 1)
run_dir = "run_v3" if version == 3 else "run"
print(version, run_dir)
EOF
)"

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

mkdir -p "${results_dir}" "${logs_dir}"

if [[ "$version" == "3" ]]; then
    train_csv=$(python - <<'EOF'
import config
print(config.TRAIN_FILE_PATH)
EOF
)
    val_csv=$(python - <<'EOF'
import config
print(config.VAL_FILE_PATH)
EOF
)
    test_csv=$(python - <<'EOF'
import config
print(config.TEST_FILE_PATH)
EOF
)
    train_fp_dir=$(python - <<'EOF'
import config
print(config.TRAIN_FP_PATH)
EOF
)
    val_fp_dir=$(python - <<'EOF'
import config
print(config.VAL_FP_PATH)
EOF
)
    test_fp_dir=$(python - <<'EOF'
import config
print(config.TEST_FP_PATH)
EOF
)
    mapfile -t assays < <(python - <<'EOF'
import config
from toxcast_pkg.v3_data import get_assay_names_from_csv
print('\n'.join(get_assay_names_from_csv(config.TRAIN_FILE_PATH)))
EOF
)
else
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
    mapfile -t assays < <(python - <<'EOF'
import pandas as pd
import sys
path = sys.argv[1]
df = pd.read_excel(path, sheet_name='data', header=None)
print('\n'.join(str(v) for v in df.iloc[0,2:].tolist()))
EOF
"${file_path}")
fi

max_jobs=45
current_jobs=0

assay_index=0
for assay_name in "${assays[@]}"; do
    assay_dir="${logs_dir}/${assay_name}"
    mkdir -p "$assay_dir"
    for model in "${models[@]}"; do
        for fingerprint in "${fingerprints[@]}"; do
            save_dir="${results_dir}/model_save_path/${assay_name}/${assay_name}_${fingerprint}_${model}"
            mkdir -p "$save_dir"

            log_file="${assay_dir}/${assay_name}_${fingerprint}_${model}.log"
            err_file="${assay_dir}/${assay_name}_${fingerprint}_${model}.err"

            echo "[$(date)] Running ${assay_name}/${assay_name}_${fingerprint}_${model}"

            if [[ "$version" == "3" ]]; then
                python "./${run_dir}/${model}.py" \
                    --fingerprint_type "$fingerprint" \
                    --train_csv "$train_csv" \
                    --val_csv "$val_csv" \
                    --test_csv "$test_csv" \
                    --train_fp_dir "$train_fp_dir" \
                    --val_fp_dir "$val_fp_dir" \
                    --test_fp_dir "$test_fp_dir" \
                    --assay_name "$assay_name" \
                    --assay_index "$assay_index" \
                    --model_save_path "$save_dir" \
                    --random_state 42 \
                    >"$log_file" 2>"$err_file" &
            else
                python "./${run_dir}/${model}.py" \
                    --fingerprint_type "$fingerprint" \
                    --file_path "$file_path" \
                    --model_save_path "$save_dir" \
                    --assay_num "$assay_index" \
                    --fp_path "$fp_path" \
                    >"$log_file" 2>"$err_file" &
            fi

            ((current_jobs++))
            if (( current_jobs >= max_jobs )); then
                wait
                current_jobs=0
            fi
        done
    done
    ((assay_index++))
done

wait
