#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/outputs/train/act-sft/libero-spatial/checkpoints/global_step_4800/actor/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/rl/sac/act/self-collected-libero-spatial-single-batch}"
REPLAY_ROOT="${REPLAY_ROOT:-$OUTPUT_ROOT/replay}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
NUM_GPUS="${NUM_GPUS:-4}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-200}"
TD3_BC_ALPHA="${TD3_BC_ALPHA:-0.5}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "Missing ACT SFT checkpoint: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi

mkdir -p \
  "$OUTPUT_ROOT/checkpoints" \
  "$REPLAY_ROOT" \
  "$OUTPUT_ROOT/tensorboard" \
  "$OUTPUT_ROOT/videos"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export TENSORBOARD_DIR="$OUTPUT_ROOT/tensorboard"

"$PYTHON" -m verl_vla.entrypoints.train.sac \
  "model/override@cluster.actor_rollout_ref.model.override_config=act" \
  "model/adapter@cluster.actor_rollout_ref.model.adapter=act" \
  "ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=$MUJOCO_GL" \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=$VERL_LOGGING_LEVEL" \
  "cluster.resource.model.nnodes=1" \
  "cluster.resource.model.gpus_per_node=$NUM_GPUS" \
  "cluster.resource.model.workers_per_node=1" \
  "cluster.resource.env.device=cpu" \
  "cluster.resource.env.workers_per_node=2" \
  "cluster.env.env_loop.pipeline_stage_num=1" \
  "cluster.env.env_loop.max_interactions=20" \
  "cluster.env.env_worker.auto_reset=true" \
  "cluster.env.env_worker.modes=[train,eval]" \
  "cluster.env.env_worker.num_envs=8" \
  "cluster.env.env_worker.simulator.libero.max_episode_steps=200" \
  "cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial" \
  "cluster.env.env_worker.simulator.libero.task_ids=[0]" \
  "cluster.env.env_worker.recorder.enable=true" \
  "cluster.env.env_worker.recorder.recorders=[video]" \
  "cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos" \
  "cluster.actor_rollout_ref.model.path=$MODEL_PATH" \
  "cluster.actor_rollout_ref.model.load_tokenizer=false" \
  "cluster.actor_rollout_ref.model.adapter.policy_type=libero" \
  "cluster.actor_rollout_ref.model.adapter.critic.enabled=true" \
  "cluster.actor_rollout_ref.model.adapter.critic.type=mean_pool" \
  "cluster.actor_rollout_ref.model.adapter.critic.head_num=2" \
  "cluster.actor_rollout_ref.model.adapter.critic.hidden_dims=[512,256,128]" \
  "cluster.actor_rollout_ref.model.adapter.sac_rollout_noise_scale=0.1" \
  "cluster.actor_rollout_ref.model.adapter.sac_train_noise_scale=0.0" \
  "cluster.actor_rollout_ref.model.adapter.freeze_vision_tower=false" \
  "cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" \
  "cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[ACTEncoderLayer,ACTDecoderLayer]" \
  "cluster.actor_rollout_ref.actor.fsdp_config.use_torch_compile=false" \
  "cluster.actor_rollout_ref.actor.optim.lr=$ACTOR_LR" \
  "cluster.actor_rollout_ref.actor.optim.clip_grad=3.5" \
  "cluster.actor_rollout_ref.actor.optim.warmup_style=constant" \
  "cluster.actor_rollout_ref.actor.mini_batch_size=64" \
  "cluster.actor_rollout_ref.actor.micro_batch_size=4" \
  "cluster.actor_rollout_ref.actor.actor_update_interval=1" \
  "cluster.actor_rollout_ref.actor.sac.auto_entropy=false" \
  "cluster.actor_rollout_ref.actor.sac.initial_alpha=0.0" \
  "cluster.actor_rollout_ref.actor.td3.enabled=true" \
  "cluster.actor_rollout_ref.actor.td3.bc_alpha=$TD3_BC_ALPHA" \
  "cluster.actor_rollout_ref.actor.cql.enabled=false" \
  "cluster.actor_rollout_ref.actor.cql.alpha=0.5" \
  "cluster.actor_rollout_ref.actor.cql.temperature=1.0" \
  "cluster.actor_rollout_ref.actor.cql.noise_scale=1.0" \
  "cluster.actor_rollout_ref.actor.critic.lr=1e-4" \
  "cluster.actor_rollout_ref.actor.critic.gamma=0.999" \
  "cluster.actor_rollout_ref.actor.critic.tau=0.005" \
  "cluster.actor_rollout_ref.actor.critic.grad_clip=10.0" \
  "cluster.actor_rollout_ref.actor.critic.skip_update_when_actor_update=true" \
  "cluster.actor_rollout_ref.actor.critic.warmup_steps=$CRITIC_WARMUP_STEPS" \
  "cluster.actor_rollout_ref.actor.replay.critic_positive_sample_ratio=0.5" \
  "cluster.actor_rollout_ref.actor.replay.actor_positive_sample_ratio=0.9" \
  "cluster.actor_rollout_ref.actor.replay.online_sample_batch_size=64" \
  "cluster.actor_rollout_ref.actor.replay.save_interval=$((TOTAL_TRAINING_STEPS + 1))" \
  "cluster.actor_rollout_ref.actor.replay.online_single_size=512" \
  "cluster.actor_rollout_ref.actor.replay.save_dir=$REPLAY_ROOT" \
  "cluster.actor_rollout_ref.rollout.output_critic_value=false" \
  "cluster.checkpoint.resume_mode=disable" \
  "cluster.checkpoint.default_local_dir=$OUTPUT_ROOT/checkpoints" \
  "cluster.checkpoint.max_actor_ckpt_to_keep=2" \
  "trainer.logger=[console,tensorboard]" \
  "trainer.project_name=act-libero-sac" \
  "trainer.experiment_name=self_collected_task0_single_batch" \
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
  "trainer.rollout_interval=100" \
  "trainer.rollout_times=0" \
  "trainer.warm_rollout_steps=0" \
  "trainer.val_before_train=false" \
  "trainer.eval_episodes=50" \
  "trainer.save_freq=50" \
  "trainer.test_freq=50" \
  "$@"
