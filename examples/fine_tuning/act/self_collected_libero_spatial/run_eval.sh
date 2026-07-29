#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "$REPO_ROOT"

CHECKPOINT_ROOT="./outputs/train/act-sft/libero-spatial/checkpoints"
LATEST_STEP="$(<"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")"

vvla-eval \
  model/override@cluster.actor_rollout_ref.model.override_config=act \
  model/adapter@cluster.actor_rollout_ref.model.adapter=act \
  cluster.actor_rollout_ref.model.path="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/actor/huggingface" \
  cluster.actor_rollout_ref.model.load_tokenizer=false \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.resource.model.gpus_per_node=1 \
  cluster.resource.env.gpus_per_node=1 \
  cluster.env.env_worker.num_envs=8 \
  output_dir="./outputs/eval/act-sft/libero-spatial/task-1-parallel" \
  "$@"
