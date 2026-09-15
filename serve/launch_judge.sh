#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_LEN="${MAX_LEN:-16384}"
TP="${TP:-1}"
VENV="${VENV:-$(cd "$(dirname "$0")" && pwd)/.venv}"

export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

exec "$VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --tensor-parallel-size "$TP" \
    --served-model-name judge
