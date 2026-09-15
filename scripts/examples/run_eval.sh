#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && . ./.env
export CROPD_EVAL_NOTHINK=1 CROPD_EVAL_MAX_TOKENS="${CROPD_EVAL_MAX_TOKENS:-4096}" CROPD_BENCH_NOTHINK=1
M="${CROPD_MODELS:-models}"
GPU="${GPU:-0}"
MAX_NEW="${MAX_NEW:-4096}"
MODELS="${MODELS:?set MODELS to a space-separated list of model directory names under \$CROPD_MODELS}"
mkdir -p logs
for name in $MODELS; do
  echo "=== INTERNAL DEV $name $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eval/eval_matrix.py --name "$name" --model "$M/$name" > "logs/eval_dev_$name.log" 2>&1
  echo "=== RULE-SCORED BENCHMARKS $name $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eval/eval_bench.py --name "$name" --model "$M/$name" \
    --bench gpqa_diamond,math500,scibench,aime26,chembench --max-new "$MAX_NEW" > "logs/eval_rule_$name.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eval/eval_bench.py --name "$name" --model "$M/$name" \
    --bench mmlu_pro --subsample 2000 --max-new "$MAX_NEW" >> "logs/eval_rule_$name.log" 2>&1
  echo "=== JUDGE-SCORED BENCHMARKS $name $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eval/eval_bench.py --name "$name" --model "$M/$name" \
    --bench rar_science --subsample 500 --max-new "$MAX_NEW" --judge-workers 64 > "logs/eval_rar_$name.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python scripts/eval/eval_bench.py --name "$name" --model "$M/$name" \
    --bench rar_med --subsample 150 --max-new "$MAX_NEW" --judge-workers 64 >> "logs/eval_rar_$name.log" 2>&1
  echo "=== DONE $name $(date '+%F %T') ==="
done
.venv/bin/python scripts/eval/bench_table.py
