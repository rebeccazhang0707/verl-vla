#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "$REPO_ROOT"

exec vvla-train-sac \
  --config-dir "./examples/rl/td3_bc/gaussian_actor/libero_spatial_task0_online_from_hf_step100" \
  --config-name td3_bc \
  output_dir=./outputs/rl/td3-bc/gaussian-actor/libero-spatial-task0-online-from-hf-step100 \
  cluster.actor_rollout_ref.model.path=Miical/gaussian-actor-libero-spatial-task0-step100-baseline \
  cluster.resource.model.gpus_per_node=1 \
  cluster.resource.model.workers_per_node=1 \
  cluster.resource.env.device=cuda \
  cluster.resource.env.gpus_per_node=2 \
  cluster.env.env_worker.num_envs=8 \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=egl \
  ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=./outputs/rl/td3-bc/gaussian-actor/libero-spatial-task0-online-from-hf-step100/tensorboard \
  "$@"
