#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "$REPO_ROOT"

exec vvla-train-sac \
  --config-dir "./examples/rl/td3_bc/pi05/libero_spatial_task2_online_from_sft_step100" \
  --config-name td3_bc \
  output_dir=./outputs/rl/td3-bc/pi05/libero-spatial-task2-online-from-sft-step100 \
  cluster.actor_rollout_ref.model.path=Miical/pi05-libero-spatial-sft-step-100 \
  cluster.resource.model.gpus_per_node=8 \
  cluster.resource.model.workers_per_node=8 \
  cluster.resource.env.device=cpu \
  cluster.resource.env.workers_per_node=4 \
  cluster.env.env_worker.num_envs=8 \
  cluster.actor_rollout_ref.actor.mini_batch_size=128 \
  cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa \
  ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=./outputs/rl/td3-bc/pi05/libero-spatial-task2-online-from-sft-step100/tensorboard \
  "$@"
