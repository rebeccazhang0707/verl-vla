#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "$REPO_ROOT"

exec vvla-train-sac \
  --config-dir "./examples/rl/dsrl/pi05/libero_spatial_online_from_sft_step100" \
  --config-name dsrl \
  output_dir=./outputs/rl/dsrl/pi05/libero-spatial-task9-cnn-noise002 \
  cluster.actor_rollout_ref.model.path=Miical/pi05-libero-spatial-sft-step-100 \
  cluster.resource.model.gpus_per_node=8 \
  cluster.resource.model.workers_per_node=1 \
  cluster.resource.env.device=cpu \
  cluster.resource.env.workers_per_node=16 \
  cluster.env.env_worker.num_envs=2 \
  cluster.env.env_worker.simulator.libero.task_ids='[9]' \
  cluster.actor_rollout_ref.actor.mini_batch_size=128 \
  cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  cluster.actor_rollout_ref.model.adapter.dsrl.rollout_noise_scale=0.02 \
  trainer.experiment_name=pi05-libero-spatial-task9-cnn-noise002-dsrl \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa \
  ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=./outputs/rl/dsrl/pi05/libero-spatial-task9-cnn-noise002/tensorboard \
  "$@"
