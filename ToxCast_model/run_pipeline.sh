#!/bin/bash
# Simple wrapper to execute the Python pipeline with a dataset argument

cd "$(dirname "$0")" || exit 1

dataset="$1"
if [ -z "$dataset" ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

python pipeline.py --dataset "$dataset"
