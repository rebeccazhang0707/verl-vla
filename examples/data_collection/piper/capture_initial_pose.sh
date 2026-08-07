#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "verl-vla-piper" ]]; then
  echo "Activate the Piper environment first: conda activate verl-vla-piper" >&2
  exit 1
fi

DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}"
QUESTARM_ROOT="$DATA_ROOT/verl-vla/QuestArmTeleop"
if [[ ! -f "$QUESTARM_ROOT/install/setup.bash" ]]; then
  echo "Missing QuestArm ROS workspace: $QUESTARM_ROOT/install/setup.bash" >&2
  exit 1
fi

set +u
source "$QUESTARM_ROOT/install/setup.bash"
set -u

exec python "$SCRIPT_DIR/capture_initial_pose.py" "$@"
