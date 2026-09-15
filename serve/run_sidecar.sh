#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
[ -f .env ] && . ./.env
GPU="${GPU:-5}"
PORT="${PORT:-8001}"
M="${CROPD_MODELS:-models}"
BASE="${BASE:-$M/Qwen3-4B}"
EXPERT_PREFIX="${EXPERT_PREFIX:-$M/expert}"
UTIL="${UTIL:-0.75}"
MAX_LEN="${MAX_LEN:-6144}"
BATCHED_TOKENS="${BATCHED_TOKENS:-4096}"
VLLM_BIN="${VLLM_BIN:-.venv/bin/vllm}"
for a in quant symbolic mech evidence; do
  [ -d "$EXPERT_PREFIX-$a/lora_adapter" ] || { echo "missing adapter: $EXPERT_PREFIX-$a/lora_adapter"; exit 1; }
done
while true; do
  echo "=== sidecar (re)start $(date -u +%FT%TZ) on GPU $GPU ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$BASE" \
    --port "$PORT" --served-model-name repair \
    --enable-lora --max-loras 5 --max-lora-rank 64 \
    --lora-modules "quant=$EXPERT_PREFIX-quant/lora_adapter" "symbolic=$EXPERT_PREFIX-symbolic/lora_adapter" \
      "mech=$EXPERT_PREFIX-mech/lora_adapter" "evidence=$EXPERT_PREFIX-evidence/lora_adapter" \
    --gpu-memory-utilization "$UTIL" --max-model-len "$MAX_LEN" --max-num-batched-tokens "$BATCHED_TOKENS"
  echo "=== sidecar exited (rc=$?) $(date -u +%FT%TZ), restarting in 15s ==="
  sleep 15
done
