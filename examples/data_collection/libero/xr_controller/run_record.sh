#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CERT_DIR="${CERT_DIR:-$PROJECT_ROOT/certs}"

cd "$PROJECT_ROOT"

vvla-record \
  num_episodes=10 \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[xr_controller]' \
  cluster.env.env_worker.teleop.xr_controller.pos_sensitivity=150.0 \
  cluster.env.env_worker.teleop.xr_controller.rot_sensitivity=4.0 \
  cluster.env.env_worker.teleop.server.ssl_certfile="$CERT_DIR/teleop-server.crt" \
  cluster.env.env_worker.teleop.server.ssl_keyfile="$CERT_DIR/teleop-server.key" \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/record-xr/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial_xr \
  cluster.env.env_worker.recorder.video.root="./outputs/record-xr/videos" \
  resume=false \
  "$@"
