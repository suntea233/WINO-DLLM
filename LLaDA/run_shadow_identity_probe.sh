#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/lx/WINO-DLLM/LLaDA"
PYTHON="/root/lx/miniconda3/envs/delax/bin/python"
RESULT_ROOT="$ROOT/results/soft_revision_v6"
RERUN_DIR="$RESULT_ROOT/shadow_probe_rerun"

mkdir -p "$RERUN_DIR"
"$PYTHON" "$ROOT/eval_soft_revision_v6.py" \
    --output-dir "$RERUN_DIR" \
    --num-samples 1319
"$PYTHON" "$ROOT/analyze_shadow_identity_probe.py"
