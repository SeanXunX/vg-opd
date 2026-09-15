#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
[ -f .env ] && . ./.env

STEPS="${STEPS:-300}"
EXP="${EXP:-vgopd-v1}"
MODEL_PATH="${MODEL_PATH:-${CROPD_MODELS:-models}/Qwen3-4B}"
GPUS="${GPUS:-2,3,4}"
NGPU="${NGPU:-3}"
UTIL="${UTIL:-0.45}"
RESP="${RESP:-3072}"
MAXLEN="${MAXLEN:-4096}"
PPO_TOK="${PPO_TOK:-12288}"
BATCH="${BATCH:-64}"
SMOKE="${SMOKE:-0}"
VAL_BEFORE=True
if [ "$SMOKE" = "1" ]; then BATCH=15; VAL_BEFORE=False; EXP="${EXP}-smoke"; fi
export CUDA_VISIBLE_DEVICES="$GPUS"
export CROPD_VGOPD_HOOK=1
export CROPD_DISTILL_EXT=cropd.verl_ext.losses
export CROPD_REPAIR_BASE_URL="${CROPD_REPAIR_BASE_URL:-http://localhost:8001/v1}"
export CROPD_JUDGE_BASE_URL="${CROPD_JUDGE_BASE_URL:-http://localhost:8000/v1}"
export CROPD_REWARD_TIME_BUDGET="${CROPD_REWARD_TIME_BUDGET:-30}"
export CROPD_JUDGE_TIMEOUT="${CROPD_JUDGE_TIMEOUT:-60}"
export CROPD_HOOK_DEBUG="${CROPD_HOOK_DEBUG:-$SMOKE}"
export CROPD_HOOK_MODE="${CROPD_HOOK_MODE:-vgopd}"
export CROPD_GATE="${CROPD_GATE:-1}"
export CROPD_ROUTE_MODE="${CROPD_ROUTE_MODE:-criterion}"
export CROPD_ATTR_REPAIR="${CROPD_ATTR_REPAIR:-0}"
export CROPD_TEACHER_RUBRIC="${CROPD_TEACHER_RUBRIC:-0}"
export CROPD_PROBE_SKIP_SCORE="${CROPD_PROBE_SKIP_SCORE:-0.75}"
TOPK="${TOPK:-0}"
LOSS_MODE="${LOSS_MODE:-vgopd_weighted}"
TOPK_CFG=128
if [ "$TOPK" -gt 0 ]; then
  LOSS_MODE=forward_kl_topk
  TOPK_CFG="$TOPK"
  export CROPD_TEACHER_TOPK="$TOPK"
  export CROPD_W_AT_AGG=1
fi
VAL_N="${VAL_N:-1}"
VAL_SAMPLE="${VAL_SAMPLE:-0}"
VAL_ARGS=""
if [ "$VAL_SAMPLE" = "1" ]; then VAL_ARGS="actor_rollout_ref.rollout.val_kwargs.n=$VAL_N actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0"; fi
LOGGER='["console"]'
[ -n "${WANDB_API_KEY:-}" ] && [ "$SMOKE" != "1" ] && LOGGER='["console","wandb"]'
mkdir -p outputs

.venv/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/processed/verl_merge_v1/train.parquet \
    data.val_files=data/processed/verl_merge_v1/dev.parquet \
    data.train_batch_size="$BATCH" \
    data.max_prompt_length=1024 \
    data.max_response_length="$RESP" \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_TOK" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$UTIL" \
    actor_rollout_ref.rollout.max_model_len="$MAXLEN" \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward_model.reward_manager=naive \
    reward.custom_reward_function.path=cropd/rewards/criteria_reward.py \
    reward.custom_reward_function.name=compute_score \
    distillation.enabled=True \
    distillation.nnodes=0 \
    distillation.n_gpus_per_node=0 \
    distillation.distillation_loss.loss_mode="$LOSS_MODE" \
    distillation.distillation_loss.topk="$TOPK_CFG" \
    distillation.distillation_loss.use_task_rewards="${USE_TASK_REWARDS:-False}" \
    distillation.distillation_loss.use_policy_gradient="${USE_PG:-False}" \
    trainer.n_gpus_per_node="$NGPU" \
    trainer.nnodes=1 \
    trainer.logger="$LOGGER" \
    trainer.project_name="${PROJECT:-vgopd}" \
    trainer.experiment_name="$EXP" \
    trainer.total_training_steps="$STEPS" \
    trainer.val_before_train="$VAL_BEFORE" \
    trainer.test_freq="${TEST_FREQ:-25}" \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_local_dir="outputs/ckpt/${EXP}" \
    $VAL_ARGS \
    "$@" 2>&1 | tee "outputs/${EXP}.log"
