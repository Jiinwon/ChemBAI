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
    data_dir=$(python - <<'EOF'
import config
print(config.DATA_DIR)
EOF
)

    mapfile -t seed_dirs < <(find "$data_dir" -maxdepth 1 -mindepth 1 -type d -name 'seed_*' | sort)
    if [[ ${#seed_dirs[@]} -eq 0 ]]; then
        echo "No seed directories found under $data_dir" >&2
        exit 1
    fi

    for seed_dir in "${seed_dirs[@]}"; do
        seed_name="$(basename "$seed_dir")"
        seed_results_dir="${results_dir}/${seed_name}"
        seed_logs_dir="${logs_dir}/${seed_name}"
        mkdir -p "$seed_results_dir" "$seed_logs_dir"

        train_csv="$seed_dir/train_df.csv"
        val_csv="$seed_dir/val_df.csv"
        test_csv="$seed_dir/test_df.csv"
        [[ -f "$train_csv" ]] || train_csv="$seed_dir/train/train_df.csv"
        [[ -f "$val_csv" ]] || val_csv="$seed_dir/val/val_df.csv"
        [[ -f "$test_csv" ]] || test_csv="$seed_dir/test/test_df.csv"

        train_fp_dir="$seed_dir/fingerprints/train"
        val_fp_dir="$seed_dir/fingerprints/val"
        test_fp_dir="$seed_dir/fingerprints/test"
        [[ -d "$seed_dir/train/fingerprints" ]] && train_fp_dir="$seed_dir/train/fingerprints"
        [[ -d "$seed_dir/val/fingerprints" ]] && val_fp_dir="$seed_dir/val/fingerprints"
        [[ -d "$seed_dir/test/fingerprints" ]] && test_fp_dir="$seed_dir/test/fingerprints"

        if [[ ! -f "$train_csv" ]]; then
            echo "Missing train_df.csv for $seed_dir" >&2
            continue
        fi

        mapfile -t assays < <(python - <<'EOF'
import sys
from toxcast_pkg.v3_data import get_assay_names_from_csv
print("\n".join(get_assay_names_from_csv(sys.argv[1])))
EOF
"$train_csv")

        for assay_name in "${assays[@]}"; do
            assay_dir="${seed_logs_dir}/${assay_name}"
            mkdir -p "$assay_dir"
            for model in "${models[@]}"; do
                for fingerprint in "${fingerprints[@]}"; do
                    save_dir="${seed_results_dir}/${assay_name}_${model}_${fingerprint}"
                    mkdir -p "$save_dir"

                    log_file="${assay_dir}/${assay_name}_${model}_${fingerprint}.log"
                    err_file="${assay_dir}/${assay_name}_${model}_${fingerprint}.err"

                    echo "[$(date)] Running ${seed_name}/${assay_name}_${model}_${fingerprint}"

                    python "./${run_dir}/${model}.py" \
                        --fingerprint_type "$fingerprint" \
                        --train_csv "$train_csv" \
                        --val_csv "$val_csv" \
                        --test_csv "$test_csv" \
                        --train_fp_dir "$train_fp_dir" \
                        --val_fp_dir "$val_fp_dir" \
                        --test_fp_dir "$test_fp_dir" \
                        --assay_name "$assay_name" \
                        --model_save_path "$save_dir" \
                        --random_state 42 \
                        >"$log_file" 2>"$err_file"
                done
            done
        done

        python -m toxcast_pkg.v3_summary \
            --seed-dir "$seed_dir" \
            --results-dir "$seed_results_dir" \
            --output "$seed_results_dir/summary.csv"
    done

    python -m toxcast_pkg.v3_summary \
        --results-dir "$results_dir" \
        --aggregate "$results_dir/summary_all_seeds.csv"
    exit 0
fi

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

            python "./${run_dir}/${model}.py" \
                --fingerprint_type "$fingerprint" \
                --file_path "$file_path" \
                --model_save_path "$save_dir" \
                --assay_num "$assay_index" \
                --fp_path "$fp_path" \
                >"$log_file" 2>"$err_file" &

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
