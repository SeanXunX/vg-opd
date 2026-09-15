#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && . ./.env
M="${CROPD_MODELS:-models}"
BASE="${BASE:-$M/Qwen3-4B}"
PREFIX="${PREFIX:-student}"
GPUS="${GPUS:-2,3,4}"
NGPU="${NGPU:-3}"
BATCH="${BATCH:-21}"
UTIL="${UTIL:-0.55}"
PPO_TOK="${PPO_TOK:-6144}"
STEPS="${STEPS:-300}"
LAMBDA="${LAMBDA:-0.25}"
TRAIN="${TRAIN:-data/processed/verl_merge_v1/train.parquet}"
DEV="${DEV:-data/processed/verl_merge_v1/dev.parquet}"
ARMS="${ARMS:-vgopd grpo}"
DATA="data.train_files=$TRAIN data.val_files=$DEV"
COMMON_ENV="BATCH=$BATCH GPUS=$GPUS NGPU=$NGPU UTIL=$UTIL RESP=4096 MAXLEN=5120 PPO_TOK=$PPO_TOK MODEL_PATH=$BASE STEPS=$STEPS TEST_FREQ=25 VAL_N=4 VAL_SAMPLE=1 CROPD_TEACHER_RUBRIC=0"
FUSED="actor_rollout_ref.actor.entropy_coeff=0.01 actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl +data.apply_chat_template_kwargs.enable_thinking=false trainer.save_freq=25 trainer.max_actor_ckpt_to_keep=8 $DATA"
PURE="distillation.distillation_loss.distillation_loss_coef=1.0 actor_rollout_ref.actor.entropy_coeff=0.005 actor_rollout_ref.actor.optim.lr=5e-6 +data.apply_chat_template_kwargs.enable_thinking=false trainer.save_freq=25 trainer.max_actor_ckpt_to_keep=8 $DATA"
mkdir -p logs
run() {
  local exp=$1; shift
  echo "=== LAUNCH $exp $(date '+%F %T') ==="
  env $COMMON_ENV EXP="$exp" "$@" < /dev/null > "logs/$exp.launch.log" 2>&1
  echo "=== DONE $exp rc=$? $(date '+%F %T') ==="
}
for arm in $ARMS; do
  case "$arm" in
    vgopd)      run "$PREFIX-vgopd" USE_TASK_REWARDS=True USE_PG=True CROPD_HOOK_MODE=vgopd \
                  bash scripts/train/train_student.sh distillation.distillation_loss.distillation_loss_coef="$LAMBDA" $FUSED ;;
    grpo)       run "$PREFIX-grpo" CROPD_PROBE_SKIP_SCORE=-1 \
                  bash scripts/train/train_student.sh distillation.enabled=False $FUSED actor_rollout_ref.actor.entropy_coeff=0 ;;
    grpo_opsd)  run "$PREFIX-grpo-opsd" USE_TASK_REWARDS=True USE_PG=True CROPD_HOOK_MODE=opsd \
                  bash scripts/train/train_student.sh distillation.distillation_loss.distillation_loss_coef="$LAMBDA" $FUSED ;;
    grpo_mopd)  run "$PREFIX-grpo-mopd" USE_TASK_REWARDS=True USE_PG=True CROPD_HOOK_MODE=mopd \
                  bash scripts/train/train_student.sh distillation.distillation_loss.distillation_loss_coef="$LAMBDA" $FUSED ;;
    cripo)      run "$PREFIX-cripo" MAXLEN=8192 CROPD_CRIPO=1 CROPD_PROBE_SKIP_SCORE=-1 \
                  bash scripts/train/train_student.sh distillation.enabled=False $FUSED actor_rollout_ref.actor.entropy_coeff=0 ;;
    vgopd_pure) run "$PREFIX-vgopd-pure" USE_TASK_REWARDS=False USE_PG=True CROPD_HOOK_MODE=vgopd \
                  bash scripts/train/train_student.sh $PURE ;;
    opsd_pure)  run "$PREFIX-opsd-pure" USE_TASK_REWARDS=False USE_PG=True CROPD_HOOK_MODE=opsd \
                  bash scripts/train/train_student.sh $PURE ;;
    mopd_pure)  run "$PREFIX-mopd-pure" USE_TASK_REWARDS=False USE_PG=True CROPD_HOOK_MODE=mopd \
                  bash scripts/train/train_student.sh $PURE ;;
    *) echo "unknown arm: $arm"; exit 1 ;;
  esac
done
