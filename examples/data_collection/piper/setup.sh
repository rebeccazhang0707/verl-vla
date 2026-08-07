#!/usr/bin/env bash
set -euo pipefail

# Install the isolated QuestArm ROS runtime used by PiperEnv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROS_ENV_NAME="${PIPER_ROS_CONDA_ENV:-vt}"
DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}"
QUESTARM_ROOT="${PIPER_QUESTARM_ROOT:-$DATA_ROOT/verl-vla/QuestArmTeleop}"

ROS_CHANNELS=(
  "https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge"
  "https://prefix.dev/robostack-humble"
)
ROS_PACKAGES=(
  "python=3.11"
  "ros-humble-desktop"
  "colcon-common-extensions"
  "compilers"
  "cmake"
  "ninja"
  "pkg-config"
  "pinocchio=3.2.0"
  "casadi=3.6.7"
  "numpy=1.26.4"
  "scipy"
  "pyyaml"
  "python-can"
  "pip"
)

QUESTARM_URL="https://github.com/agilexrobotics/QuestArmTeleop.git"
QUESTARM_COMMIT="4420567a9031357b0dced64c6c0ab697d07d7e25"
AGX_ARM_ROS_URL="https://github.com/agilexrobotics/agx_arm_ros.git"
AGX_ARM_ROS_COMMIT="22a9cf6c5ad2fd2e0743531936bc5dab007fa5bc"
AGX_ARM_URDF_URL="https://github.com/agilexrobotics/agx_arm_urdf.git"
AGX_ARM_URDF_COMMIT="9ffe0cdb26b8bb03b84a648f3cd119822049f2e7"
PYAGXARM_URL="https://github.com/agilexrobotics/pyAgxArm.git"
PYAGXARM_COMMIT="799b8412fbe8b9156bc9892d3dbeb2df7e98be71"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

checkout_repo() {
  local url="$1"
  local commit="$2"
  local destination="$3"

  if [[ ! -d "$destination/.git" ]]; then
    if [[ -e "$destination" ]]; then
      echo "Refusing to replace non-Git path: $destination" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$destination")"
    git clone --filter=blob:none "$url" "$destination"
  fi

  if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to change a repository with tracked modifications: $destination" >&2
    exit 1
  fi

  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

require_command git

if [[ -n "${PIPER_CONDA_BASE:-}" && -x "$PIPER_CONDA_BASE/bin/conda" ]]; then
  CONDA_BASE="$PIPER_CONDA_BASE"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  CONDA_BASE="$HOME/miniconda3"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "Miniconda/Conda is required. Install it before running this script." >&2
  exit 1
fi

channel_args=()
for channel in "${ROS_CHANNELS[@]}"; do
  channel_args+=("-c" "$channel")
done

MICROMAMBA="$CONDA_BASE/bin/micromamba"
if [[ -x "$MICROMAMBA" ]]; then
  if "$MICROMAMBA" env list -r "$CONDA_BASE" | awk '{print $1}' | grep -Fxq "$ROS_ENV_NAME"; then
    "$MICROMAMBA" install -y -r "$CONDA_BASE" -n "$ROS_ENV_NAME" \
      --override-channels "${channel_args[@]}" "${ROS_PACKAGES[@]}"
  else
    "$MICROMAMBA" create -y -r "$CONDA_BASE" -n "$ROS_ENV_NAME" \
      --override-channels "${channel_args[@]}" "${ROS_PACKAGES[@]}"
  fi
  run_in_ros_env() {
    "$MICROMAMBA" run -r "$CONDA_BASE" -n "$ROS_ENV_NAME" "$@"
  }
else
  CONDA="$CONDA_BASE/bin/conda"
  if "$CONDA" env list | awk '{print $1}' | grep -Fxq "$ROS_ENV_NAME"; then
    "$CONDA" install -y -n "$ROS_ENV_NAME" \
      --override-channels "${channel_args[@]}" "${ROS_PACKAGES[@]}"
  else
    "$CONDA" create -y -n "$ROS_ENV_NAME" \
      --override-channels "${channel_args[@]}" "${ROS_PACKAGES[@]}"
  fi
  run_in_ros_env() {
    "$CONDA" run -n "$ROS_ENV_NAME" "$@"
  }
fi

checkout_repo "$QUESTARM_URL" "$QUESTARM_COMMIT" "$QUESTARM_ROOT"
checkout_repo "$AGX_ARM_ROS_URL" "$AGX_ARM_ROS_COMMIT" "$QUESTARM_ROOT/src/agx_arm_ros"
checkout_repo \
  "$AGX_ARM_URDF_URL" \
  "$AGX_ARM_URDF_COMMIT" \
  "$QUESTARM_ROOT/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf"
checkout_repo "$PYAGXARM_URL" "$PYAGXARM_COMMIT" "$QUESTARM_ROOT/.deps/pyAgxArm"

run_in_ros_env python -m pip install --no-deps "$QUESTARM_ROOT/.deps/pyAgxArm"

run_in_ros_env bash -c '
  set -euo pipefail
  workspace="$1"
  cd "$workspace"
  CC=cc CXX=c++ colcon --log-base log build \
    --base-paths src \
    --build-base build \
    --install-base install \
    --packages-select agx_arm_msgs agx_arm_description agx_arm_ctrl oculus_reader \
    --cmake-args -G Ninja
' _ "$QUESTARM_ROOT"

run_in_ros_env bash -c '
  set -euo pipefail
  source "$1/install/setup.bash"
  python -c "import casadi, pinocchio, pyAgxArm, rclpy"
  for package in agx_arm_ctrl agx_arm_description agx_arm_msgs oculus_reader; do
    ros2 pkg prefix "$package" >/dev/null
  done
' _ "$QUESTARM_ROOT"

cat <<EOF

Piper ROS environment is ready.

Run teleoperation with:
  $SCRIPT_DIR/run.sh

The default installation paths are discovered automatically. If you installed
to custom paths, override ros_conda_sh, ros_conda_env, or questarm_setup_path
through the Piper Hydra configuration when launching.
EOF
