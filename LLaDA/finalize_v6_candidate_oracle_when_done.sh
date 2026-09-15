#!/usr/bin/env bash
set -u
results="/root/lx/WINO-DLLM/LLaDA/results/v6_revision_candidate_oracle"
python="/root/lx/miniconda3/envs/delax/bin/python"
for dataset in gsm8k math500 mbpp humaneval; do
  while [ ! -f "$results/.${dataset}.exit" ]; do sleep 15; done
  if [ "$(cat "$results/.${dataset}.exit")" != "0" ]; then
    printf 'Worker %s failed; see run_%s.log\n' "$dataset" "$dataset" \
      > "$results/finalize.log"
    exit 1
  fi
done
cd /root/lx/WINO-DLLM/worktrees/v6_revision_candidate_oracle/LLaDA || exit 1
"$python" eval_v6_candidate_oracle.py finalize > "$results/finalize.log" 2>&1
