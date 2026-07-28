# XR Controller

This example uses a WebXR controller to control a LIBERO robot. It covers
three workflows:

- teleoperate the robot without saving data;
- record human demonstrations as a LeRobot dataset; and
- intervene while a trained policy runs to collect DAgger trajectories.

Complete the [LIBERO installation](installation.md) before running the
commands below.

## Generate an HTTPS certificate

WebXR requires a secure HTTPS context. Generate a self-signed certificate for
the machine running LIBERO before starting an XR workflow.

Set `SERVER_IP` to the address that the XR headset uses to reach the machine:

```bash
export SERVER_IP="192.168.1.100"
mkdir -p certs

openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 3650 \
  -keyout certs/teleop-server.key \
  -out certs/teleop-server.crt \
  -subj "/CN=${SERVER_IP}" \
  -addext "subjectAltName=IP:${SERVER_IP},DNS:localhost" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment,keyCertSign" \
  -addext "extendedKeyUsage=serverAuth"
```

Install `certs/teleop-server.crt` as a trusted certificate on the XR headset.
The exact import procedure depends on the headset and browser. Before starting
WebXR, open `https://${SERVER_IP}:18000` from the headset and confirm that the
page loads without a certificate warning.

Keep `teleop-server.key` private. The `certs/` directory is ignored by Git.

## Configuration

The LIBERO XR strategy exposes controller selection, motion sensitivity, and
button configuration:

| Configuration | Framework default | LIBERO value | Description |
| --- | --- | --- | --- |
| `cluster.env.env_worker.teleop.xr_controller.hand` | `right` | `right` | Controller hand used for robot control |
| `cluster.env.env_worker.teleop.xr_controller.pos_sensitivity` | `1.0` | `150.0` | Translation scale applied to controller motion |
| `cluster.env.env_worker.teleop.xr_controller.rot_sensitivity` | `0.5` | `4.0` | Rotation scale applied to controller motion |
| `cluster.env.env_worker.teleop.xr_controller.intervention_button` | `squeeze` | `squeeze` | Button held to enable XR control |
| `cluster.env.env_worker.teleop.xr_controller.gripper_button` | `trigger` | `trigger` | Analog button used to control the gripper |
| `cluster.env.env_worker.teleop.xr_controller.button_threshold` | `0.5` | `0.5` | Activation threshold for squeeze and trigger |

The commands below apply the larger motion sensitivities used for LIBERO.
You can override them further in any XR Teleop, Record, or DAgger command:

```text
cluster.env.env_worker.teleop.xr_controller.pos_sensitivity=150.0
cluster.env.env_worker.teleop.xr_controller.rot_sensitivity=4.0
```

Reduce the sensitivities for finer control or increase them for faster motion.

## Controls

After opening the teleoperation dashboard in the headset browser, select
`Enter XR` and approve the browser's WebXR permission request. Loading the
dashboard alone does not start controller tracking; XR actions become
available only after the immersive session begins.

Controller motion is converted into incremental position and rotation commands
in real time. While `squeeze` is held, each command is computed from the
change since the previous controller frame, and that frame is then updated for
the next step. Pressing `squeeze` initializes this frame-to-frame tracking; it
does not hold the first pose as a fixed reference.

| Input | Control |
| --- | --- |
| Controller grip pose | Relative position and rotation |
| `squeeze` | Hold to enable XR control |
| `trigger` | Open or close the gripper |
| `A` / `X` | Start or end the current episode |
| `B` / `Y` | Discard and restart the current episode |
| Keyboard `R` | Mark the current episode as successful |
| Keyboard `Enter` | Start or end the current episode |
| Keyboard `Backspace` | Discard and restart the current episode |

The robot receives a neutral action while `squeeze` is released. Hold
`squeeze` while moving the controller or using the trigger.

## Teleoperate the robot

Set the certificate paths and start task 0 from the `libero_spatial` suite:

```bash
vvla-teleop \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[xr_controller]' \
  cluster.env.env_worker.teleop.xr_controller.pos_sensitivity=150.0 \
  cluster.env.env_worker.teleop.xr_controller.rot_sensitivity=4.0 \
  cluster.env.env_worker.teleop.server.ssl_certfile="$PWD/certs/teleop-server.crt" \
  cluster.env.env_worker.teleop.server.ssl_keyfile="$PWD/certs/teleop-server.key"
```

Open `https://${SERVER_IP}:18000` in the headset browser and select
`Enter XR`. Confirm that the dashboard reports incoming XR frames, then hold
`squeeze` and move the configured controller to operate the robot.

Press `A/X` or keyboard `Enter` to reset the environment. The environment also
resets automatically when the episode reaches its maximum length. Press
`Ctrl+C` in the terminal to stop teleoperation.

The default interaction rate is 30 Hz. To use another target rate, append
`cluster.env.env_worker.target_step_hz`, for example:

```text
cluster.env.env_worker.target_step_hz=15
```

If LIBERO cannot sustain the configured rate, the worker prints a warning with
the measured frequency. Reduce the target rate when this occurs repeatedly,
especially when using OSMesa CPU rendering.

### Use CPU rendering

When a GPU is unavailable, append these overrides:

```text
cluster.resource.env.device=cpu
ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa
```

CPU rendering may make interactive control less responsive.

## Record demonstrations

This example records 10 XR demonstrations for task 0:

```bash
vvla-record \
  num_episodes=10 \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids='[0]' \
  cluster.env.env_worker.teleop.devices='[xr_controller]' \
  cluster.env.env_worker.teleop.xr_controller.pos_sensitivity=150.0 \
  cluster.env.env_worker.teleop.xr_controller.rot_sensitivity=4.0 \
  cluster.env.env_worker.teleop.server.ssl_certfile="$PWD/certs/teleop-server.crt" \
  cluster.env.env_worker.teleop.server.ssl_keyfile="$PWD/certs/teleop-server.key" \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/record-xr/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial_xr \
  cluster.env.env_worker.recorder.video.root="./outputs/record-xr/videos" \
  resume=false
```

Open `https://${SERVER_IP}:18000`, select `Enter XR`, and wait for controller
tracking. At the beginning of each episode, press `A/X` on the controller or
keyboard `Enter` when prompted to begin recording.

During recording:

- hold `squeeze` while controlling the robot;
- completing the task saves the trajectory and resets the environment;
- keyboard `R` marks the episode as successful and saves it;
- `A/X` ends and saves the current episode early; and
- `B/Y` discards the current episode and starts it again.

The command exits after 10 completed episodes. The dataset and videos are
written to:

```text
outputs/record-xr/lerobot/local/libero_spatial_xr
outputs/record-xr/videos
```

Set `resume=true` to append new episodes to an existing dataset. Existing
episodes count toward `num_episodes`.

## Collect DAgger trajectories

DAgger runs an existing policy in LIBERO while allowing the XR controller to
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
  cluster.env.env_worker.teleop.devices='[xr_controller]' \
  cluster.env.env_worker.teleop.xr_controller.pos_sensitivity=150.0 \
  cluster.env.env_worker.teleop.xr_controller.rot_sensitivity=4.0 \
  cluster.env.env_worker.teleop.server.ssl_certfile="$PWD/certs/teleop-server.crt" \
  cluster.env.env_worker.teleop.server.ssl_keyfile="$PWD/certs/teleop-server.key" \
  cluster.env.env_worker.recorder.lerobot.root="./outputs/dagger-xr/lerobot" \
  cluster.env.env_worker.recorder.lerobot.repo_id=local/libero_spatial_xr_dagger \
  cluster.env.env_worker.recorder.video.root="./outputs/dagger-xr/videos" \
  max_episodes=10 \
  resume=false
```

Open `https://${SERVER_IP}:18000`, select `Enter XR`, and confirm that
controller tracking is active. Press `A/X` or keyboard `Enter` when prompted
to begin each trajectory.

The policy controls the robot while `squeeze` is released. Hold `squeeze` to
intervene with the XR controller, and release it to return control to the
policy. Actions executed while `squeeze` is held are marked as interventions
in the recorded dataset.

During collection, keyboard `R` marks the trajectory as successful, `A/X`
ends and saves it early, and `B/Y` discards and restarts it.

The collected dataset and videos are written below:

```text
outputs/dagger-xr/lerobot/local/libero_spatial_xr_dagger
outputs/dagger-xr/videos
```

Set `resume=true` to continue an existing DAgger dataset. The resulting
intervention-enhanced dataset can then be used for another policy fine-tuning
stage, completing one DAgger post-training iteration.
