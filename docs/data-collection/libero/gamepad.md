# Gamepad

This example uses a browser-connected gamepad to control a LIBERO robot. It
covers three workflows:

- teleoperate the robot without saving data;
- record human demonstrations as a LeRobot dataset; and
- intervene while a trained policy runs to collect DAgger trajectories.

Complete the [LIBERO installation](installation.md) before running the
commands below.

## Configuration

The LIBERO gamepad strategy exposes motion sensitivity, control buttons, and
axis mappings:

| Configuration | Default | Description |
| --- | --- | --- |
| `cluster.env.env_worker.teleop.gamepad.pos_sensitivity` | `0.5` | Translation scale applied to the analog sticks |
| `cluster.env.env_worker.teleop.gamepad.rot_sensitivity` | `0.5` | Rotation scale applied to the right stick and D-pad |
| `cluster.env.env_worker.teleop.gamepad.intervention_button` | `RT` | Button held to enable gamepad control |
| `cluster.env.env_worker.teleop.gamepad.gripper_button` | `X` | Button used to toggle the gripper |
| `cluster.env.env_worker.teleop.gamepad.left_stick_x_axis` | `axis_0` | Left-stick horizontal axis |
| `cluster.env.env_worker.teleop.gamepad.left_stick_y_axis` | `axis_1` | Left-stick vertical axis |
| `cluster.env.env_worker.teleop.gamepad.right_stick_x_axis` | `axis_2` | Right-stick horizontal axis |
| `cluster.env.env_worker.teleop.gamepad.right_stick_y_axis` | `axis_3` | Right-stick vertical axis |

Override these values in a Gamepad Teleop, Record, or DAgger command. For
example:

```text
cluster.env.env_worker.teleop.gamepad.pos_sensitivity=0.3
cluster.env.env_worker.teleop.gamepad.rot_sensitivity=0.3
```

Reduce the sensitivities for finer control. If a controller reports a
non-standard axis layout, use the four axis options to remap its sticks.

## Controls

The browser reads the gamepad through the Gamepad API. Connect the controller,
open the teleoperation dashboard, and press or move one control so the browser
can discover it. The dashboard shows the device ID, active axes, pressed
buttons, and current command.

| Input | Control |
| --- | --- |
| Left Stick Y | Move along the +x / -x axis |
| Left Stick X | Move along the +y / -y axis |
| Right Stick Y | Move along the +z / -z axis |
| Right Stick X | Rotate around the +yaw / -yaw axis |
| D-Pad Left / Right | Rotate around the +roll / -roll axis |
| D-Pad Up / Down | Rotate around the +pitch / -pitch axis |
| `RT` | Hold to enable gamepad control |
| `X` | Toggle the gripper |
| `RB` | Mark the current episode as successful |
| `LT` | Discard and restart the current episode |
| `LB` | Start or end the current episode |

The robot receives a neutral action while `RT` is released. Hold `RT` while
moving the sticks, using the D-pad, or changing the gripper.

## Teleoperate the robot

The following command starts task 0 from the `libero_spatial` suite with EGL
rendering:

```bash
vvla-teleop \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[gamepad]'
```

Open `http://localhost:18000` in a browser. If LIBERO is running on another
machine, replace `localhost` with that machine's hostname or IP address.

Confirm that the dashboard reports the gamepad as connected. Hold `RT` and use
the controls above to operate the robot. Press `LB` to reset the environment;
the environment also resets automatically when the episode reaches its
maximum length. Press `Ctrl+C` in the terminal to stop teleoperation.

The default interaction rate is 30 Hz. To use another target rate, append
`cluster.env.env_worker.target_step_hz`, for example:

```text
cluster.env.env_worker.target_step_hz=15
```

If LIBERO cannot sustain the configured rate, the worker prints a warning with
the measured frequency. Reduce the target rate when this occurs repeatedly,
especially when using OSMesa CPU rendering.

### Use CPU rendering

When a GPU is unavailable, add the CPU resource and OSMesa overrides:

```bash
vvla-teleop \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[gamepad]' \
  cluster.resource.env.device=cpu \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa
```

CPU rendering may make interactive control less responsive.

## Record demonstrations

This example records 10 gamepad demonstrations for task 0:

```bash
vvla-record \
  num_episodes=10 \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[gamepad]' \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/record-gamepad/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial_gamepad \
  cluster.env.env_worker.recorder.video.root="./outputs/record-gamepad/videos" \
  resume=false
```

Open `http://localhost:18000`. At the beginning of each episode, the dashboard
console asks for confirmation. Press `Enter` on the keyboard or `LB` on the
gamepad to begin recording.

During recording:

- hold `RT` while controlling the robot;
- completing the task saves the trajectory and resets the environment;
- `RB` marks the episode as successful and saves it;
- `LB` ends and saves the current episode early; and
- `LT` discards the current episode and starts it again.

The command exits after 10 completed episodes. The dataset and videos are
written to:

```text
outputs/record-gamepad/lerobot/local/libero_spatial_gamepad
outputs/record-gamepad/videos
```

Set `resume=true` to append new episodes to an existing dataset. Existing
episodes count toward `num_episodes`.

### Record with CPU rendering

Append the following overrides to the recording command:

```text
cluster.resource.env.device=cpu
ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa
```

## Collect DAgger trajectories

DAgger runs an existing policy in LIBERO while allowing the gamepad to
override its actions. The collection workflow uses environment and rollout
workers only; policy fine-tuning is performed separately after collection.

The following example loads the latest ACT checkpoint produced by the
[Quick Start](../../getting-started/index.md) and collects 10 trajectories:

```bash
vvla-dagger \
  model/override@cluster.actor_rollout_ref.model.override_config=act \
  model/adapter@cluster.actor_rollout_ref.model.adapter=act \
  cluster.actor_rollout_ref.model.path="./outputs/train/act-sft/libero-spatial/checkpoints/global_step_$(cat "./outputs/train/act-sft/libero-spatial/checkpoints/latest_checkpointed_iteration.txt")/actor/huggingface" \
  cluster.actor_rollout_ref.model.load_tokenizer=false \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[gamepad]' \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/dagger-gamepad/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial_gamepad_dagger \
  cluster.env.env_worker.recorder.video.root="./outputs/dagger-gamepad/videos" \
  max_episodes=10 \
  resume=false
```

Open the observation URL printed by the command and confirm that the dashboard
reports the gamepad as connected. Press `Enter` on the keyboard or `LB` on the
gamepad when prompted to begin each trajectory.

The policy controls the robot while `RT` is released. Hold `RT` to intervene
with the gamepad, and release it to return control to the policy. Actions
executed while `RT` is held are marked as interventions in the recorded
dataset.

During collection, `RB` marks the current trajectory as successful, `LB` ends
and saves it early, and `LT` discards and restarts it.

The collected dataset and videos are written below:

```text
outputs/dagger-gamepad/lerobot/local/libero_spatial_gamepad_dagger
outputs/dagger-gamepad/videos
```

Set `resume=true` to continue an existing DAgger dataset. The resulting
intervention-enhanced dataset can then be used for another policy fine-tuning
stage, completing one DAgger post-training iteration.
