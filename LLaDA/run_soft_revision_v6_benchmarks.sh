#!/usr/bin/env bash
set -uo pipefail

ROOT="/root/lx/WINO-DLLM/LLaDA"
PYTHON="/root/lx/miniconda3/envs/delax/bin/python"
RESULT_DIR="$ROOT/results/soft_revision_v6"
EVALUATOR="$ROOT/eval_soft_revision_extended.py"

mkdir -p "$RESULT_DIR"
overall_status=0

for dataset in math500 humaneval mbpp countdown; do
    log_file="$RESULT_DIR/run_v6_${dataset}.log"
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] starting ${dataset}" >> "$log_file"
    if "$PYTHON" "$EVALUATOR" \
        --dataset "$dataset" \
        --method soft_revision_v6 \
        --result-dir "$RESULT_DIR" >> "$log_file" 2>&1; then
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] completed ${dataset}" >> "$log_file"
    else
        status=$?
        overall_status=$status
        echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] failed ${dataset}: exit ${status}" >> "$log_file"
    fi
done

exit "$overall_status"
