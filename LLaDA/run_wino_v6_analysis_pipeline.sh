#!/usr/bin/env bash
set -u
root=/root/lx/WINO-DLLM/worktrees/wino_v6_revision_analysis/LLaDA
out=/root/lx/WINO-DLLM/LLaDA/results/wino_v6_revision_analysis
py=/root/lx/miniconda3/envs/delax/bin/python
mkdir -p "$out"
cd "$root" || exit 1

if [ "$1" = trace ]; then
  dataset="$2"; gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$py" trace_wino_identity_dataset.py "$dataset" > "$out/run_analysis1_${dataset}.log" 2>&1
  printf '%s\n' "$?" > "$out/.analysis1_${dataset}.exit"
  exit
fi

if [ "$1" = old ]; then
  dataset="$2"; gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$py" run_analysis2_old.py "$dataset" > "$out/run_analysis2_${dataset}.log" 2>&1
  printf '%s\n' "$?" > "$out/.analysis2_${dataset}.exit"
  exit
fi

if [ "$1" = schedule ]; then
  for dataset in math500 mbpp humaneval; do
    while [ ! -f "$out/.analysis1_${dataset}.exit" ]; do sleep 15; done
    [ "$(cat "$out/.analysis1_${dataset}.exit")" = 0 ] || exit 1
  done
  "$py" analyze_wino_v6_revision.py analysis1 > "$out/analysis1_finalize.log" 2>&1 || exit 1
  "$py" analyze_wino_v6_revision.py prepare2 >> "$out/analysis1_finalize.log" 2>&1 || exit 1
  tmux new-session -d -s wv6a2_gsm -c "$root" "bash run_wino_v6_analysis_pipeline.sh old gsm8k 4"
  tmux new-session -d -s wv6a2_math -c "$root" "bash run_wino_v6_analysis_pipeline.sh old math500 3"
  tmux new-session -d -s wv6a2_code -c "$root" "bash run_wino_v6_analysis_pipeline.sh old mbpp 0; [ \"\$(cat '$out/.analysis2_mbpp.exit')\" = 0 ] && bash run_wino_v6_analysis_pipeline.sh old humaneval 0"
  for dataset in gsm8k math500 mbpp humaneval; do
    while [ ! -f "$out/.analysis2_${dataset}.exit" ]; do sleep 15; done
    [ "$(cat "$out/.analysis2_${dataset}.exit")" = 0 ] || exit 1
  done
  "$py" finalize_wino_v6_analysis.py > "$out/finalize.log" 2>&1
  exit $?
fi

exit 2
