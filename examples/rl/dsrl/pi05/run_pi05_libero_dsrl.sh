#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/file_system/liujincheng/models/global_step_14_merged_hf}"
REPLAY_POOL_DIR="${REPLAY_POOL_DIR:-$REPO_ROOT/outputs/libero_dsrl/pi05_libero_spatial_task1_critic_10q_mean_replay/replay_pool}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/libero_dsrl/pi05_libero_task1_offline}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-350}"
NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p \
  "$OUTPUT_ROOT/checkpoints" \
  "$OUTPUT_ROOT/tensorboard" \
  "$OUTPUT_ROOT/videos"
mkdir -p /root/LIBERO/libero/datasets 2>/dev/null || true
for ((rank = 0; rank < NUM_GPUS; rank++)); do
  replay_file="$REPLAY_POOL_DIR/sac_replay_pool_rank_$rank.pt"
  if [[ ! -f "$replay_file" ]]; then
    echo "Missing fixed replay pool shard: $replay_file" >&2
    exit 1
  fi
done

export MUJOCO_GL=osmesa
export VERL_LOGGING_LEVEL=INFO
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export TENSORBOARD_DIR="$OUTPUT_ROOT/tensorboard"

python -m verl_vla.entrypoints.train.sac \
  "ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa" \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  "ray_kwargs.ray_init.runtime_env.env_vars.HF_TOKEN=" \
  "cluster.resource.model.nnodes=1" \
  "cluster.resource.model.gpus_per_node=$NUM_GPUS" \
  "cluster.resource.model.workers_per_node=1" \
  "cluster.resource.env.device=cpu" \
  "cluster.resource.env.workers_per_node=4" \
  "cluster.env.env_loop.pipeline_stage_num=2" \
  "cluster.env.env_loop.max_interactions=20" \
  "cluster.env.env_worker.auto_reset=true" \
  "cluster.env.env_worker.modes=[train,eval]" \
  "cluster.env.env_worker.num_envs=8" \
  "cluster.env.env_worker.simulator.simulator_type=libero" \
  "cluster.env.env_worker.simulator.libero.max_episode_steps=200" \
  "cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial" \
  "cluster.env.env_worker.simulator.libero.task_ids=[1]" \
  "cluster.env.env_worker.recorder.enable=true" \
  "cluster.env.env_worker.recorder.recorders=[video]" \
  "cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos" \
  "cluster.actor_rollout_ref.model.path=$MODEL_PATH" \
  "cluster.actor_rollout_ref.model.tokenizer_path=$MODEL_PATH" \
  "cluster.actor_rollout_ref.model.enable_gradient_checkpointing=false" \
  "cluster.actor_rollout_ref.model.adapter.embodiment=libero" \
  "cluster.actor_rollout_ref.model.adapter.dsrl.enabled=true" \
  "cluster.actor_rollout_ref.model.adapter.dsrl.state_dim=8" \
  "cluster.actor_rollout_ref.model.adapter.dsrl.mlp.feature_latent_dim=64" \
  "cluster.actor_rollout_ref.model.adapter.dsrl.mlp.state_latent_dim=64" \
  "cluster.actor_rollout_ref.model.adapter.dsrl.mlp.hidden_dims=[128,128,128]" \
  "cluster.actor_rollout_ref.model.adapter.critic.type=mean_pool" \
  "cluster.actor_rollout_ref.model.adapter.critic.head_num=10" \
  "cluster.actor_rollout_ref.model.adapter.critic.input_dim=2112" \
  "cluster.actor_rollout_ref.model.adapter.critic.hidden_dims=[512,256,128]" \
  "cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" \
  "cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SiglipEncoderLayer,GemmaDecoderLayerWithExpert]" \
  "cluster.actor_rollout_ref.actor.optim.lr=1e-4" \
  "cluster.actor_rollout_ref.actor.optim.clip_grad=3.5" \
  "cluster.actor_rollout_ref.actor.optim.warmup_style=constant" \
  "cluster.actor_rollout_ref.actor.mini_batch_size=512" \
  "cluster.actor_rollout_ref.actor.micro_batch_size=8" \
  "cluster.actor_rollout_ref.actor.actor_update_interval=1" \
  "cluster.actor_rollout_ref.actor.cql.enabled=true" \
  "cluster.actor_rollout_ref.actor.cql.alpha=0.5" \
  "cluster.actor_rollout_ref.actor.cql.noise_scale=1.0" \
  "cluster.actor_rollout_ref.actor.critic.lr=1e-4" \
  "cluster.actor_rollout_ref.actor.critic.gamma=0.999" \
  "cluster.actor_rollout_ref.actor.critic.tau=0.005" \
  "cluster.actor_rollout_ref.actor.critic.grad_clip=10.0" \
  "cluster.actor_rollout_ref.actor.critic.skip_update_when_actor_update=true" \
  "cluster.actor_rollout_ref.actor.critic.warmup_steps=100" \
  "cluster.actor_rollout_ref.actor.replay.critic_positive_sample_ratio=0.5" \
  "cluster.actor_rollout_ref.actor.replay.online_sample_batch_size=512" \
  "cluster.actor_rollout_ref.actor.replay.save_interval=$((TOTAL_TRAINING_STEPS + 1))" \
  "cluster.actor_rollout_ref.actor.replay.online_single_size=20000" \
  "cluster.actor_rollout_ref.actor.replay.save_dir=$REPLAY_POOL_DIR" \
  "cluster.actor_rollout_ref.rollout.output_critic_value=false" \
  "cluster.checkpoint.resume_mode=disable" \
  "cluster.checkpoint.default_local_dir=$OUTPUT_ROOT/checkpoints" \
  "cluster.checkpoint.max_actor_ckpt_to_keep=2" \
  "trainer.logger=[console,tensorboard]" \
  "trainer.project_name=pi05-libero-dsrl" \
  "trainer.experiment_name=pi05_libero_spatial_task1_dsrl" \
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
  "trainer.rollout_interval=$TOTAL_TRAINING_STEPS" \
  "trainer.rollout_times=0" \
  "trainer.warm_rollout_steps=0" \
  "trainer.save_freq=100" \
  "trainer.test_freq=100" \
  "trainer.eval_episodes=50" \
  "$@"
