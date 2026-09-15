#!/usr/bin/env bash
set -u
dataset="$1"
gpu="$2"
root="/root/lx/WINO-DLLM/worktrees/v6_revision_candidate_oracle/LLaDA"
results="/root/lx/WINO-DLLM/LLaDA/results/v6_revision_candidate_oracle"
python="/root/lx/miniconda3/envs/delax/bin/python"
mkdir -p "$results"
cd "$root" || exit 1
CUDA_VISIBLE_DEVICES="$gpu" "$python" eval_v6_candidate_oracle.py run --dataset "$dataset" \
  > "$results/run_${dataset}.log" 2>&1
status=$?
printf '%s\n' "$status" > "$results/.${dataset}.exit"
exit "$status"
