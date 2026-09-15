#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && . ./.env
EXP="${1:?usage: fold_checkpoint.sh EXP STEP}"
STEP="${2:?usage: fold_checkpoint.sh EXP STEP}"
M="${CROPD_MODELS:-models}"
BASE="${BASE:-$M/Qwen3-4B}"
TARGET="${TARGET:-$M/$EXP-s$STEP}"
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m verl.model_merger merge --backend fsdp \
  --local_dir "outputs/ckpt/$EXP/global_step_$STEP/actor" --target_dir "$TARGET"
CROPD_FOLD_BASE="$BASE" .venv/bin/python scripts/train/fold_lora.py "$TARGET"
echo "folded $TARGET (LoRA adapter kept in $TARGET/lora_adapter)"
