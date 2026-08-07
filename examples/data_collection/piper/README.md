# Piper X teleoperation

This workflow runs verl-vla and ROS in separate Python environments:

```text
verl-vla main environment
  -> local UDP
ROS Humble environment
  -> QuestArm IK -> agx_arm_ctrl -> SocketCAN -> Piper X
```

ROS Python packages are never imported into the verl-vla process. `PiperEnv`
starts and stops the ROS subprocess only when the Piper simulator is selected.

## 1. Install verl-vla

Create the normal verl-vla environment and install the Piper extra:

```bash
uv venv --python 3.12 .env
uv pip install --python .env/bin/python -e '.[piper]'
```

The extra contains only dependencies used by the main process. ROS, rclpy,
Pinocchio, CasADi and pyAgxArm deliberately stay out of this environment.

## 2. Install the isolated ROS runtime

Miniconda (or another Conda installation) and Git must already be available.
Then run:

```bash
examples/data_collection/piper/setup.sh
```

The script creates or updates the `vt` Conda environment with its pinned ROS
dependencies, checks out verified upstream revisions, installs pyAgxArm into
that environment, builds the four ROS packages used by PiperEnv, and verifies
their imports. By default, the upstream workspace is stored at:

```text
~/.local/share/verl-vla/QuestArmTeleop
```

Override the location or environment name when necessary:

```bash
PIPER_QUESTARM_ROOT=/data/QuestArmTeleop \
PIPER_ROS_CONDA_ENV=vt \
examples/data_collection/piper/setup.sh
```

The repositories are pinned to commits that were verified with Piper X. The
installer refuses to switch an existing checkout that has tracked local
changes.

## 3. Configure Piper

The checked-in Piper Hydra config is
`src/verl_vla/workflows/config/env/simulator/piper.yaml`. It owns the CAN
mapping, cameras, QuestArm runtime, control scales, IK weights and gripper
parameters. There is no separate environment file to copy or maintain.

`can_channels` is always ordered as `[physical_left, physical_right]`.
Interface names are machine-specific. This repository's tested configuration
uses `[can_right, can_left]`; update `piper.yaml` when installing on another
machine.

The CAN interfaces must already exist and be UP at 1 Mbps before launch.
Installing `can-utils` is recommended for diagnostics. Camera lists may both
be empty. If cameras are configured, their device/name lists must have the
same order and length.

## 4. Start teleoperation

```bash
examples/data_collection/piper/run.sh
```

`run.sh` only selects the Piper simulator and forwards additional Hydra
overrides. A temporary hardware override does not require another config file:

```bash
examples/data_collection/piper/run.sh \
  'cluster.env.env_worker.simulator.piper.can_channels=[can0,can1]'
```

Keyboard mode uses the teleop workflow's default HTTP server. XR/WebXR needs
HTTPS and can be selected with normal teleop configuration overrides:

```bash
examples/data_collection/piper/run.sh \
  'cluster.env.env_worker.teleop.devices=[xr_controller]' \
  'cluster.env.env_worker.teleop.server.ssl_certfile=/path/to/teleop.crt' \
  'cluster.env.env_worker.teleop.server.ssl_keyfile=/path/to/teleop.key'
```

The shared Piper YAML contains the tested position and rotation scale `1.25`,
IK position weight `5.0`, and smoothing weight `0.5`.

Set `initial_joint_angles` in the Piper YAML to a `2 x 6` list of joint angles
in radians when reset should return to a fixed pose. Its first row is the left
arm and its second row is the right arm. When the value is `null`, PiperEnv
captures both arms' positions when it starts and uses that pose for subsequent
resets.

`reset_duration_s` controls how long the smooth reset trajectory takes without
changing the speed of normal keyboard or XR control. Its default is `3.0`
seconds; `reset_timeout_s` must be larger than this duration.

With the Piper ROS service running, print the current pose in the exact YAML
format with:

```bash
examples/data_collection/piper/capture_initial_pose.sh
```
