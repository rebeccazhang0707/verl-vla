# Installation

This guide installs the verified LIBERO runtime in a local `.venv`. Run all
commands from the verl-vla repository root.

## Requirements

The commands below target Ubuntu 22.04 with Python 3.10 or later. A
CUDA-capable NVIDIA GPU is recommended for responsive LIBERO rendering.
OSMesa can be used for CPU rendering when a GPU is unavailable, although
interactive control may be less responsive.

Install the required system packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  ffmpeg \
  git \
  libgl1 \
  libglib2.0-0 \
  libosmesa6 \
  python3-dev \
  python3-venv
```

## Create the virtual environment

Create and activate a repository-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Activate it again whenever you open a new terminal:

```bash
source .venv/bin/activate
```

If PyPI access is slow from mainland China, configure this virtual environment
to use the Tsinghua mirror:

```bash
python -m pip config --site set \
  global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Install verl-vla and LIBERO

Install the verified LeRobot runtime first. LeRobot is installed separately
with `--no-deps` so its dependency resolution does not replace the versions
selected for verl-vla:

```bash
python -m pip install --requirement requirements-lerobot.txt
python -m pip install --no-deps lerobot==0.4.4
```

Then install verl-vla in editable mode with its LIBERO dependencies:

```bash
python -m pip install --editable ".[libero]"
```

## Install LIBERO assets

The PyPI distribution of LIBERO does not contain all assets required by the
simulator. Install the revision verified by verl-vla:

```bash
python scripts/install_libero_assets.py
```

The installer downloads the pinned LIBERO source archive, verifies its SHA-256
checksum, and installs the assets into the active environment's LIBERO
package. Run it again after recreating `.venv` or reinstalling LIBERO.

## Verify the installation

Run the LIBERO installation check:

```bash
python scripts/install_checks/check_libero.py
```

This check uses OSMesa to validate the simulator, task assets, cameras, and a
headless environment step without requiring a GPU display.

Verify the recording dependencies:

```bash
python -c \
  "import av, torchcodec; from verl_vla.recorder import get_lerobot_dataset_cls; get_lerobot_dataset_cls()"
```

If the environment will use GPU rendering, also confirm that PyTorch can see
CUDA:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Select a rendering backend

verl-vla uses EGL by default for GPU-accelerated headless rendering:

```text
MUJOCO_GL=egl
```

If EGL rendering appears on a different physical GPU from the one selected by
`CUDA_VISIBLE_DEVICES`, apply the
[robosuite EGL device-selection patch](../../troubleshooting/simulators/libero/index.md).

When running LIBERO without a GPU, select a CPU environment resource and
OSMesa in the launch command:

```text
cluster.resource.env.device=cpu
ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa
```

The device-specific examples include the appropriate launch overrides where
needed.

## Next steps

Choose a LIBERO input device:

- [Keyboard](keyboard.md)
- [Gamepad](gamepad.md)
- [XR Controller](xr-controller.md)
- [LeRobot Leader Arm](lerobot-leader-arm.md)
