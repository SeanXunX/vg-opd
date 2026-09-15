#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && . ./.env
M="${CROPD_MODELS:-models}"
BASE="${BASE:-$M/Qwen3-4B}"
PREFIX="${PREFIX:-expert}"
GPUS="${GPUS:-0,1,2,3}"
NGPU="${NGPU:-4}"
UTIL="${UTIL:-0.55}"
PPO_TOK="${PPO_TOK:-12288}"
STEPS="${STEPS:-200}"
AXES="${AXES:-quant symbolic mech evidence}"
COMMON="+data.apply_chat_template_kwargs.enable_thinking=false trainer.save_freq=25 trainer.max_actor_ckpt_to_keep=8 trainer.test_freq=25"
CLOSED="actor_rollout_ref.actor.optim.lr=5e-6 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0"
OPEN="actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.entropy_coeff=0"
mkdir -p logs
for axis in $AXES; do
  exp="$PREFIX-$axis"
  case "$axis" in
    quant|symbolic) script=scripts/train/train_expert.sh; batch=60; n=8; extra="$CLOSED" ;;
    mech|evidence)  script=scripts/train/train_expert_open.sh; batch=60; n=6; extra="$OPEN" ;;
    *) echo "unknown axis: $axis"; exit 1 ;;
  esac
  echo "=== LAUNCH $exp $(date '+%F %T') ==="
  env CUDA_VISIBLE_DEVICES="$GPUS" AXIS="$axis" EXP="$exp" MODEL_PATH="$BASE" GPUS="$GPUS" NGPU="$NGPU" \
      UTIL="$UTIL" PPO_TOK="$PPO_TOK" RESP=4096 MAXLEN=5120 STEPS="$STEPS" \
      bash "$script" data.train_batch_size="$batch" actor_rollout_ref.actor.ppo_mini_batch_size="$batch" \
      actor_rollout_ref.rollout.n="$n" $COMMON $extra "$@" < /dev/null > "logs/$exp.launch.log" 2>&1
  echo "=== DONE $exp rc=$? $(date '+%F %T') ==="
done
