# Keyboard

This example uses the browser keyboard to control a LIBERO robot. It covers
three workflows:

- teleoperate the robot without saving data;
- record human demonstrations as a LeRobot dataset; and
- intervene while a trained policy runs to collect DAgger trajectories.

Complete the [LIBERO installation](installation.md) before running the
commands below.

## Configuration

The LIBERO keyboard strategy exposes separate sensitivities for translation
and rotation:

| Configuration | Default | Description |
| --- | --- | --- |
| `cluster.env.env_worker.teleop.keyboard.pos_sensitivity` | `0.05` | Translation applied on each control step |
| `cluster.env.env_worker.teleop.keyboard.rot_sensitivity` | `0.12` | Rotation applied on each control step |

Override these values in any Keyboard Teleop, Record, or DAgger command. For
example:

```text
cluster.env.env_worker.teleop.keyboard.pos_sensitivity=0.03
cluster.env.env_worker.teleop.keyboard.rot_sensitivity=0.08
```

Reduce the values for finer control or increase them for faster motion.

## Controls

Click the teleoperation dashboard before using the keyboard. The dashboard
shows the active bindings and the latest command sent to the environment.

| Key | Control |
| --- | --- |
| `W` / `S` | Move along the -x / +x axis |
| `A` / `D` | Move along the +y / -y axis |
| `Q` / `E` or `Page Up` / `Page Down` | Move along the +z / -z axis |
| `Z` / `X` | Rotate around the +roll / -roll axis |
| `T` / `G` or `↑` / `↓` | Rotate around the +pitch / -pitch axis |
| `C` / `V` or `←` / `→` | Rotate around the +yaw / -yaw axis |
| `K` | Toggle the gripper |
| `L` | Reset the keyboard control state |
| `Space` | Enter or leave intervention during policy rollout |
| `R` | Mark the current episode as successful |
| `Backspace` | Discard and restart the current episode |
| `Enter` | End the current episode |

> **Tip:** LIBERO Spatial task 0 can be completed using only `W/A/S/D` and
> `Page Up/Page Down`.

## Teleoperate the robot

The following command starts task 0 from the `libero_spatial` suite with EGL
rendering:

```bash
vvla-teleop \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[keyboard]'
```

Open `http://localhost:18000` in a browser. If LIBERO is running on another
machine, replace `localhost` with that machine's hostname or IP address.

![LIBERO keyboard teleoperation dashboard](../../_static/images/teleop-dashboard.png)

Follow the controls shown in the dashboard to operate the robot. Press
`Enter` to reset the environment. The environment also resets automatically
when the episode reaches its maximum length. Press `Ctrl+C` in the terminal to
stop teleoperation.

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
  cluster.env.env_worker.teleop.devices='[keyboard]' \
  cluster.resource.env.device=cpu \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa
```

CPU rendering may make interactive control less responsive.

## Record demonstrations

This example records 10 human demonstrations for task 0:

```bash
vvla-record \
  num_episodes=10 \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[keyboard]' \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/record/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial \
  cluster.env.env_worker.recorder.video.root="./outputs/record/videos" \
  resume=false
```

Open `http://localhost:18000`. At the beginning of each episode, the dashboard
console asks you to press `Enter`. Recording starts after this confirmation.

During recording:

- completing the task saves the trajectory and resets the environment;
- `R` marks the episode as successful and saves it;
- `Enter` ends and saves the current episode early; and
- `Backspace` discards the current episode and starts it again.

The command exits after 10 completed episodes. The dataset and videos are
written to:

```text
outputs/record/lerobot/local/libero_spatial
outputs/record/videos
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

DAgger runs an existing policy in LIBERO while allowing the keyboard to
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
  max_episodes=10 \
  resume=false
```

Open the observation URL printed by the command and focus the dashboard.
Press `Enter` when prompted to begin each trajectory.

The policy controls the robot by default. Press `Space` to enter manual
intervention, use the movement and gripper controls above, and press `Space`
again to return control to the policy. Actions executed during manual control
are marked as interventions in the recorded dataset.

The collected dataset and videos are written below:

```text
outputs/dagger/lerobot/local/verl_vla_libero_dagger
outputs/dagger/videos
```

Set `resume=true` to continue an existing DAgger dataset. The resulting
intervention-enhanced dataset can then be used for another policy fine-tuning
stage, completing one DAgger post-training iteration.
