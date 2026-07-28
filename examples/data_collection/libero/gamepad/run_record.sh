#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

vvla-record \
  num_episodes=10 \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[gamepad]' \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/record-gamepad/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial_gamepad \
  cluster.env.env_worker.recorder.video.root="./outputs/record-gamepad/videos" \
  resume=false \
  "$@"
