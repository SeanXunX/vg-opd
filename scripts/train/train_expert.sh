#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
[ -f .env ] && . ./.env

AXIS="${AXIS:-quant}"
STEPS="${STEPS:-500}"
EXP="${EXP:-expert-${AXIS}-v1}"
MODEL_PATH="${MODEL_PATH:-${CROPD_MODELS:-models}/Qwen3-4B}"
NGPU="${NGPU:-8}"
UTIL="${UTIL:-0.45}"
RESP="${RESP:-3072}"
MAXLEN="${MAXLEN:-4096}"
PPO_TOK="${PPO_TOK:-12288}"
LOGGER='["console"]'
[ -n "${WANDB_API_KEY:-}" ] && LOGGER='["console","wandb"]'
mkdir -p outputs

.venv/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="data/processed/verl_v1/train_${AXIS}.parquet" \
    data.val_files=data/processed/verl_v1/dev.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length="$RESP" \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    trainer.n_gpus_per_node="$NGPU" \
    trainer.nnodes=1 \
    trainer.logger="$LOGGER" \
    trainer.project_name="${PROJECT:-vgopd}" \
    trainer.experiment_name="$EXP" \
    trainer.total_training_steps="$STEPS" \
    trainer.val_before_train=True \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_local_dir="outputs/ckpt/${EXP}" \
    "$@" 2>&1 | tee "outputs/${EXP}.log"
