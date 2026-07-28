#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

if [[ -z "${MODEL_PATH:-}" ]]; then
  CHECKPOINT_ROOT="./outputs/train/act-sft/libero-spatial/checkpoints"
  LATEST_STEP="$(<"$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")"
  MODEL_PATH="$CHECKPOINT_ROOT/global_step_${LATEST_STEP}/actor/huggingface"
fi

vvla-dagger \
  model/override@cluster.actor_rollout_ref.model.override_config=act \
  model/adapter@cluster.actor_rollout_ref.model.adapter=act \
  cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
  cluster.actor_rollout_ref.model.load_tokenizer=false \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[gamepad]' \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/dagger-gamepad/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial_gamepad_dagger \
  cluster.env.env_worker.recorder.video.root="./outputs/dagger-gamepad/videos" \
  max_episodes=10 \
  resume=false \
  "$@"
