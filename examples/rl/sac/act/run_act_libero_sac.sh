#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "$REPO_ROOT"

vvla-train-sac \
  --config-dir "./examples/rl/sac/act" \
  --config-name act_sac \
  cluster.actor_rollout_ref.model.path="./outputs/train/act-sft/libero-spatial/checkpoints/global_step_$(cat "./outputs/train/act-sft/libero-spatial/checkpoints/latest_checkpointed_iteration.txt")/actor/huggingface" \
  cluster.resource.model.gpus_per_node=1 \
  cluster.resource.env.gpus_per_node=1 \
  cluster.resource.env.workers_per_node=2 \
  cluster.env.env_worker.num_envs=8 \
  cluster.actor_rollout_ref.actor.mini_batch_size=64 \
  cluster.actor_rollout_ref.actor.micro_batch_size=8 \
  cluster.actor_rollout_ref.actor.optim.lr=5e-6 \
  'trainer.logger=[console,tensorboard]' \
  trainer.total_training_steps=400 \
  trainer.eval_episodes=50 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  "$@"
