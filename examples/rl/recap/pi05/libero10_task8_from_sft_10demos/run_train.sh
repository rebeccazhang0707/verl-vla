#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "$REPO_ROOT"

exec vvla-train-recap \
  --config-dir "./examples/rl/recap/pi05/libero10_task8_from_sft_10demos" \
  --config-name libero10_task8 \
  output_dir=./outputs/rl/recap/pi05/libero10-task8-from-sft-10demos \
  initial_policy_path=Miical/pi05-libero10-task8-sft-10demos \
  recap.policy_eval.cluster.resource.model.gpus_per_node=8 \
  recap.policy_eval.cluster.resource.env.device=cpu \
  recap.policy_eval.cluster.resource.env.workers_per_node=8 \
  recap.policy_eval.cluster.env.env_worker.num_envs=2 \
  recap.collect_data.cluster.resource.model.gpus_per_node=8 \
  recap.collect_data.cluster.resource.env.device=cpu \
  recap.collect_data.cluster.resource.env.workers_per_node=8 \
  recap.collect_data.cluster.env.env_worker.num_envs=2 \
  recap.train_value_model.cluster.resource.model.gpus_per_node=2 \
  recap.train_value_model.data.batch_size=256 \
  recap.train_value_model.data.num_workers=8 \
  recap.train_value_model.data.prefetch_factor=8 \
  recap.train_value_model.cluster.actor_rollout_ref.actor.mini_batch_size=256 \
  recap.train_value_model.cluster.actor_rollout_ref.actor.micro_batch_size=32 \
  recap.value_infer.num_gpus=2 \
  recap.value_infer.data.batch_size=64 \
  recap.value_infer.data.num_workers=8 \
  recap.value_infer.data.prefetch_factor=8 \
  recap.train_policy.cluster.resource.model.gpus_per_node=8 \
  recap.train_policy.data.batch_size=256 \
  recap.train_policy.data.num_workers=8 \
  recap.train_policy.data.prefetch_factor=8 \
  recap.train_policy.cluster.actor_rollout_ref.actor.mini_batch_size=256 \
  recap.train_policy.cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa \
  ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=./outputs/rl/recap/pi05/libero10-task8-from-sft-10demos/tensorboard \
  "$@"
