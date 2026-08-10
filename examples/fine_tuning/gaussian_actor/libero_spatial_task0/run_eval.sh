#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "$REPO_ROOT"

CHECKPOINT_ROOT="./outputs/train/gaussian-actor-sft/libero-spatial-task0-step-sweep/checkpoints"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-500}"
OUTPUT_DIR="./outputs/eval/gaussian-actor-sft/libero-spatial-task0-step-sweep/step-${CHECKPOINT_STEP}-task0-50x256-2m2e64-i100"

vvla-eval \
  model/override@cluster.actor_rollout_ref.model.override_config=gaussian_actor \
  model/adapter@cluster.actor_rollout_ref.model.adapter=gaussian_actor \
  cluster.actor_rollout_ref.model.path="$CHECKPOINT_ROOT/global_step_${CHECKPOINT_STEP}/actor/huggingface" \
  cluster.actor_rollout_ref.model.load_tokenizer=false \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  'cluster.env.env_worker.simulator.libero.task_ids=[0]' \
  cluster.env.env_worker.simulator.libero.num_trials_per_task=50 \
  cluster.env.env_worker.simulator.libero.max_episode_steps=256 \
  cluster.env.env_loop.max_interactions=100 \
  cluster.resource.model.gpus_per_node=2 \
  cluster.resource.env.device=cuda \
  cluster.resource.env.gpus_per_node=2 \
  cluster.resource.env.workers_per_node=1 \
  cluster.env.env_worker.num_envs=16 \
  output_dir="$OUTPUT_DIR" \
  "$@"
