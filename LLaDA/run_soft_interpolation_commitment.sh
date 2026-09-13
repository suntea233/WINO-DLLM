#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PYTHON="/root/lx/miniconda3/envs/delax/bin/python"
HERE="/root/lx/WINO-DLLM-soft_interpolation_commitment/LLaDA"
OUT="/root/lx/WINO-DLLM/LLaDA/results/soft_interpolation_commitment"

cd "$HERE"
mkdir -p "$OUT"
run_one() {
  local alpha="$1"
  local dataset="$2"
    alpha_tag="${alpha/./p}"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" eval_soft_interpolation_commitment.py \
      --dataset "$dataset" --alpha "$alpha" --output-dir "$OUT" \
      > "$OUT/run_${dataset}_alpha_${alpha_tag}.log" 2>&1
}

if [[ -n "${JOBS:-}" ]]; then
  for job in $JOBS; do
    alpha="${job%%:*}"
    dataset="${job#*:}"
    run_one "$alpha" "$dataset"
  done
else
  for alpha in ${ALPHAS:-0.25 0.5 0.75}; do
    for dataset in gsm8k math500 humaneval mbpp; do
      run_one "$alpha" "$dataset"
    done
  done
fi
if [[ "${FINALIZE:-1}" == "wait" ]]; then
  while true; do
    complete=1
    for alpha in 0p25 0p5 0p75; do
      for dataset in gsm8k math500 humaneval mbpp; do
        [[ -f "$OUT/${dataset}_alpha_${alpha}.json" ]] || complete=0
      done
    done
    [[ "$complete" == "1" ]] && break
    sleep 30
  done
  "$PYTHON" summarize_soft_interpolation_commitment.py --result-dir "$OUT"
elif [[ "${FINALIZE:-1}" == "1" ]]; then
  "$PYTHON" summarize_soft_interpolation_commitment.py --result-dir "$OUT"
fi
