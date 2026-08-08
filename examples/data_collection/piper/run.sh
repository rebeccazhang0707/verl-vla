#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}"
QUESTARM_ROOT="$DATA_ROOT/verl-vla/QuestArmTeleop"

if [[ "${CONDA_DEFAULT_ENV:-}" != "verl-vla-piper" ]]; then
  echo "Activate the Piper environment first: conda activate verl-vla-piper" >&2
  exit 1
fi

if [[ ! -f "$QUESTARM_ROOT/install/setup.bash" ]]; then
  echo "Missing QuestArm ROS workspace: $QUESTARM_ROOT/install/setup.bash" >&2
  echo "Install it with: $SCRIPT_DIR/setup.sh" >&2
  exit 1
fi

set +u
source "$QUESTARM_ROOT/install/setup.bash"
set -u

cd "$PROJECT_ROOT"

mode="teleop"
if [[ "${1:-}" == "teleop" || "${1:-}" == "record" ]]; then
  mode="$1"
  shift
fi

overrides=(
  "cluster.resource.env.device=cpu"
  "cluster.resource.env.workers_per_node=1"
  "cluster.resource.env.gpus_per_node=0"
  "cluster.env.env_worker.simulator.simulator_type=piper"
  "cluster.env.env_worker.teleop.keyboard.pos_sensitivity=0.01"
  "cluster.env.env_worker.teleop.keyboard.rot_sensitivity=0.05"
  "cluster.env.env_worker.teleop.xr_controller.pos_sensitivity=1.25"
  "cluster.env.env_worker.teleop.xr_controller.rot_sensitivity=1.25"
  "cluster.env.env_worker.teleop.server.jpeg_quality=90"
)

if [[ "$mode" == "record" ]]; then
  overrides+=(
    "cluster.env.env_worker.recorder.lerobot.root=outputs/record/piper/lerobot"
    "cluster.env.env_worker.recorder.lerobot.repo_id=local/verl_vla_piper"
    "cluster.env.env_worker.recorder.video.root=outputs/record/piper/videos"
  )
fi

exec python -m "verl_vla.entrypoints.$mode" "${overrides[@]}" "$@"
