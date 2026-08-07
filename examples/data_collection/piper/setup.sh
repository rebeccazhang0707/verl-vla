#!/usr/bin/env bash
set -euo pipefail

# Install verl-vla and its Piper ROS runtime into one Conda environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PIPER_ENV_NAME="verl-vla-piper"
DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}"
QUESTARM_ROOT="$DATA_ROOT/verl-vla/QuestArmTeleop"

ROS_CHANNELS=(
  "https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge"
  "https://prefix.dev/robostack-humble"
)
ROS_PACKAGES=(
  "python=3.11"
  "ros-humble-desktop"
  "ros-humble-camera-info-manager=3.1.10"
  "colcon-common-extensions"
  "compilers"
  "cmake"
  "ffmpeg"
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
V4L2_CAMERA_URL="https://gitlab.com/boldhearts/ros2_v4l2_camera.git"
V4L2_CAMERA_COMMIT="22d6ce190f5caebd20cc35a2635c05b49008f447"

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

  if [[ -n "$(git -C "$destination" status --porcelain --untracked-files=no --ignore-submodules=all)" ]]; then
    echo "Refusing to change a repository with tracked modifications: $destination" >&2
    exit 1
  fi

  git -C "$destination" fetch --depth 1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

require_command conda
require_command git

channel_args=()
for channel in "${ROS_CHANNELS[@]}"; do
  channel_args+=("-c" "$channel")
done

CONDA="$(conda info --base)/bin/conda"
if "$CONDA" env list | awk '{print $1}' | grep -Fxq "$PIPER_ENV_NAME"; then
  "$CONDA" install -y -n "$PIPER_ENV_NAME" \
    --override-channels "${channel_args[@]}" "${ROS_PACKAGES[@]}"
else
  "$CONDA" create -y -n "$PIPER_ENV_NAME" \
    --override-channels "${channel_args[@]}" "${ROS_PACKAGES[@]}"
fi
run_in_piper_env() {
  "$CONDA" run -n "$PIPER_ENV_NAME" "$@"
}

checkout_repo "$QUESTARM_URL" "$QUESTARM_COMMIT" "$QUESTARM_ROOT"
checkout_repo "$AGX_ARM_ROS_URL" "$AGX_ARM_ROS_COMMIT" "$QUESTARM_ROOT/src/agx_arm_ros"
checkout_repo \
  "$AGX_ARM_URDF_URL" \
  "$AGX_ARM_URDF_COMMIT" \
  "$QUESTARM_ROOT/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf"
checkout_repo "$PYAGXARM_URL" "$PYAGXARM_COMMIT" "$QUESTARM_ROOT/.deps/pyAgxArm"
checkout_repo "$V4L2_CAMERA_URL" "$V4L2_CAMERA_COMMIT" "$QUESTARM_ROOT/src/v4l2_camera"

run_in_piper_env python -m pip install \
  "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cpu/torchvision-0.22.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"
run_in_piper_env env CC=cc CXX=c++ python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --requirement "$PROJECT_ROOT/requirements-lerobot.txt"
run_in_piper_env python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-deps "lerobot==0.4.4"
run_in_piper_env python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -e "$PROJECT_ROOT"
run_in_piper_env python -m pip install --no-deps "$QUESTARM_ROOT/.deps/pyAgxArm"

run_in_piper_env bash -c '
  set -euo pipefail
  workspace="$1"
  cd "$workspace"
  CC=cc CXX=c++ CXXFLAGS="-Wno-array-bounds -Wno-stringop-overflow" colcon --log-base log build \
    --base-paths src \
    --build-base build \
    --install-base install \
    --packages-select agx_arm_msgs agx_arm_description agx_arm_ctrl oculus_reader v4l2_camera \
    --cmake-args -G Ninja
' _ "$QUESTARM_ROOT"

run_in_piper_env bash -c '
  set -euo pipefail
  set +u
  source "$1/install/setup.bash"
  set -u
  python -c "import av, casadi, lerobot, pinocchio, pyAgxArm, rclpy, torchcodec, verl_vla"
  for package in agx_arm_ctrl agx_arm_description agx_arm_msgs oculus_reader v4l2_camera; do
    ros2 pkg prefix "$package" >/dev/null
  done
' _ "$QUESTARM_ROOT"

cat <<EOF

The unified verl-vla Piper environment is ready.

Activate it once in each terminal:
  conda activate $PIPER_ENV_NAME

Then run keyboard teleoperation with:
  $SCRIPT_DIR/run.sh

Record a LeRobot dataset with:
  $SCRIPT_DIR/run.sh record

EOF
