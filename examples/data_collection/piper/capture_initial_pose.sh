#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROS_ENV_NAME="${PIPER_ROS_CONDA_ENV:-vt}"

if command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Cannot find a Conda installation." >&2
  exit 1
fi

DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}"
QUESTARM_ROOT="${PIPER_QUESTARM_ROOT:-$DATA_ROOT/verl-vla/QuestArmTeleop}"
setup_candidates=(
  "$QUESTARM_ROOT/install/setup.bash"
  "$PROJECT_ROOT/../QuestArmTeleop/install-ninja2/setup.bash"
  "$PROJECT_ROOT/../QuestArmTeleop/install/setup.bash"
)
QUESTARM_SETUP=""
for candidate in "${setup_candidates[@]}"; do
  if [[ -f "$candidate" ]]; then
    QUESTARM_SETUP="$candidate"
    break
  fi
done
if [[ -z "$QUESTARM_SETUP" ]]; then
  echo "Cannot find a built QuestArmTeleop workspace." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
set +u
conda activate "$ROS_ENV_NAME"
# shellcheck source=/dev/null
source "$QUESTARM_SETUP"
set -u

exec python "$SCRIPT_DIR/capture_initial_pose.py" "$@"
