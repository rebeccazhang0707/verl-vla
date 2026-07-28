# Environment Integration

verl-vla treats simulators and physical robots as interchangeable environment
backends. Each backend translates its native runtime into the shared
environment contract, while `BaseEnv` supplies the execution behavior needed
by rollout, evaluation, recording, and human intervention.

This separation keeps environment-specific code focused on the robot or
simulator. Action-chunk execution, partial resets, teleoperation, recording,
and worker communication remain shared across integrations.

## Environment architecture

An environment integration has three layers:

1. The **native backend** owns the simulator or robot runtime.
2. A **`BaseEnv` implementation** translates native observations, actions,
   rewards, and episode signals into the verl-vla contract.
3. The **environment worker** hosts the environment and exchanges normalized
   batches with `TrainCluster`.

`BaseEnv` is the main integration boundary. Its public `reset`, `step`, and
`close` methods are already implemented. In addition to the Gym lifecycle,
the base class:

- executes policy actions with shape `[num_envs, chunk_steps, action_dim]`;
- supports stepping and resetting selected environments within a vectorized
  backend;
- publishes observations and runtime state to the teleoperation dashboard;
- replaces policy actions with human actions during intervention;
- records the action that was actually executed; and
- saves an episode when the environment terminates or is truncated.

Environment implementations should not reproduce this orchestration. They
only implement the native operations called by `BaseEnv`.

## Implement an environment

### Define the environment identity

Every environment declares a stable `env_type`. The same value connects the
environment to its simulator configuration, recorder strategy, and
teleoperation strategies.

```python
class MyRobotEnv(BaseEnv):
    env_type = "my_robot"
```

The selected simulator configuration must be available under the matching
name, for example `cfg.simulator.my_robot`.

### Implement the `BaseEnv` hooks

A minimal integration implements four hooks:

```python
import numpy as np

from verl_vla.envs.base import BaseEnv


class MyRobotEnv(BaseEnv):
    env_type = "my_robot"

    def __init__(self, cfg, rank, world_size, stage_id=0, **kwargs):
        del kwargs
        self.simulator_cfg = cfg.simulator.my_robot
        self.backend = None
        super().__init__(cfg, rank, world_size, stage_id=stage_id)

    def env_init(self):
        self.backend = MyRobotRuntime(self.simulator_cfg)

    def env_reset(self, *, env_ids, reset_eval=False, extra=None):
        del reset_eval, extra
        native_obs = self.backend.reset(env_ids)
        return self._make_observation_batch(native_obs, env_ids)

    def env_step(self, action, *, env_ids):
        native_obs, reward, terminated, truncated, info = self.backend.step(
            action, env_ids=env_ids
        )
        return {
            **self._make_observation_batch(native_obs, env_ids),
            "next.reward": np.asarray(reward, dtype=np.float32),
            "next.terminated": np.asarray(terminated, dtype=bool),
            "next.truncated": np.asarray(truncated, dtype=bool),
            "next.success": np.asarray(
                [item["success"] for item in info], dtype=bool
            ),
            "extra": info,
        }

    def env_close(self):
        self.backend.close()
```

`env_init` is called before teleoperation and recording are created, so all
backend resources required by their strategies should be ready when it
returns. `env_close` owns the corresponding cleanup.

`env_reset` and `env_step` receive global `env_ids`. The returned arrays and
lists contain only those environments, in the same order. This local-result
ordering is important when a vectorized backend performs partial resets or
steps.

### Normalize observations and episode signals

Both reset and step results use the same observation structure:

```python
{
    "observation": [
        {
            "observation.images.image": image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": state,
        },
        # One dictionary per returned environment.
    ],
    "task": ["pick up the object"],
    "task_id": np.asarray([0], dtype=np.int64),
}
```

Image keys use the `observation.images.` prefix. This allows models,
recorders, and the teleoperation dashboard to discover camera observations
without depending on the native backend. State is published under
`observation.state`, while `task` contains the language instruction associated
with each environment.

`env_step` adds four one-dimensional signals:

- `next.reward`;
- `next.terminated`;
- `next.truncated`; and
- `next.success`.

Use `terminated` when the environment reaches a natural terminal condition
and `truncated` when an external limit, such as the maximum episode length,
ends the episode. Evaluation uses `success` independently of the reward
representation.

An evaluation backend may also return `eval_episode_id` from both reset and
step. This provides a stable identifier for benchmark-case aggregation.
Environment-specific metadata can be returned through `extra`; additional
dashboard-only camera views can be returned through `teleop_images`.

### Add configuration and worker construction

Define a typed simulator configuration next to the integration and add one
Hydra configuration under `workflows/config/env/simulator/`. The configuration
should describe only the native backend and should use `simulator_type` as its
selection key.

Then connect the backend at the two explicit construction points:

1. Add the typed configuration to `SimulatorConfig` and allow its
   `simulator_type`.
2. Add an `EnvWorker.init_worker` branch that constructs `EnvManager` with the
   new environment class.

Finally, include the simulator configuration in
`workflows/config/env/simulator/simulator.yaml`. Existing workflows can then
select the backend through the standard environment configuration rather than
creating a new rollout or training pipeline.

## Recorder

Recording is a side effect of the shared environment step. `BaseEnv` passes
the pre-action observation, executed action, task, episode signals,
intervention flag, and optional critic value to `MultiRecorder`.
`MultiRecorder` fans each transition out to the enabled implementations:

- `LeRobotDatasetRecorder` writes training-ready LeRobot episodes.
- `VideoRecorder` writes annotated rollout videos.
- `AsyncRecorder` can move recorder operations to a background thread while
  preserving the same interface.

The environment decides when an episode is complete through `terminated` and
`truncated`; `BaseEnv` then calls `save_episode` automatically. Incomplete
frames are discarded on an explicit reset and remain buffered across action
chunks.

### Add recording support for a new environment

Recorders do not parse a native environment directly. Instead, a
`BaseLeRobotStrategy` defines the dataset schema and converts one normalized
environment transition into one frame. The same strategy also identifies the
images used by the video recorder.

```python
import numpy as np

from verl_vla.recorder.strategies import BaseLeRobotStrategy


class MyRobotLeRobotStrategy(BaseLeRobotStrategy):
    def __init__(
        self,
        *,
        image_shape=(256, 256, 3),
        state_dim,
        action_dim,
        fps=30,
    ):
        self.image_shape = image_shape
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._fps = fps

    @property
    def fps(self):
        return self._fps

    @property
    def robot_type(self):
        return "my_robot"

    def features(self):
        return {
            "observation.images.image": {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (self.action_dim,),
                "names": ["action"],
            },
            # Include next.* and info.* fields required by the workflow.
        }

    def make_frame(self, *, observation, action, task, **step):
        return {
            "observation.images.image": observation[
                "observation.images.image"
            ],
            "observation.state": np.asarray(
                observation["observation.state"], dtype=np.float32
            ),
            "action": np.asarray(action, dtype=np.float32),
            "task": str(task),
            # Convert the required values from step to the declared schema.
        }
```

Register the factory under the environment's `env_type`:

```python
register_lerobot_strategy("my_robot", MyRobotLeRobotStrategy)
```

If the schema depends on runtime properties such as camera resolution,
override `get_recorder_strategy_kwargs` on the environment. `BaseEnv` forwards
the returned values to the strategy factory:

```python
def get_recorder_strategy_kwargs(self):
    return {
        "image_shape": (
            self.simulator_cfg.camera_height,
            self.simulator_cfg.camera_width,
            3,
        )
    }
```

Once the strategy is registered, both LeRobot dataset recording and video
recording can be enabled through the shared recorder configuration.

## Teleoperation and intervention

Teleoperation is divided into three responsibilities:

- A **device** translates browser or hardware events into a stable device
  snapshot. Built-in devices include keyboard, gamepad, XR controller, and a
  LeRobot leader arm.
- An **intervention strategy**—the teleoperation policy—translates one device's
  state into actions for one environment type.
- `TeleopController` connects devices and strategies to the observation server,
  publishes environment state, and applies human action overrides.

This split is intentional. A keyboard device reports pressed keys but does not
know whether `W` means Cartesian motion, a joint command, or locomotion. That
meaning belongs to the environment-specific teleoperation policy. The same
device can therefore control different embodiments without placing robot
semantics in the browser input layer.

### Write a teleoperation policy

Create an `InterventionStrategyBase` implementation for each supported
`(env_type, device_type)` pair:

```python
import numpy as np

from verl_vla.teleop.strategies import InterventionStrategyBase


class MyRobotKeyboardStrategy(InterventionStrategyBase):
    env_type = "my_robot"
    device_type = "keyboard"

    def __init__(self, cfg, *, simulator_cfg):
        super().__init__(cfg)
        self.action_dim = simulator_cfg.action_dim
        self.active = False

    def reset(self):
        self.active = False

    def is_intervening(self, device):
        self._process_events(device)
        return self.active

    def apply_action(self, action, device):
        if not self.is_intervening(device):
            return action
        return self._action_from_snapshot(device.snapshot())

    def get_action(self, device):
        return self._action_from_snapshot(device.snapshot())

    def snapshot(self, device):
        return {
            "strategy": f"{self.env_type}:{self.device_type}",
            "active": self.active,
            "key_bindings": {
                "Space": "toggle intervention",
                "W/S": "move forward/backward",
            },
        }

    def _process_events(self, device):
        for event in device.drain_events():
            if (
                event["event_type"] == "keydown"
                and event["code"] == "SPACE"
                and not event["repeat"]
            ):
                self.active = not self.active

    def _action_from_snapshot(self, snapshot):
        action = np.zeros(self.action_dim, dtype=np.float32)
        pressed = set(snapshot["pressed_keys"])
        if "W" in pressed:
            action[0] += self.cfg.pos_sensitivity
        if "S" in pressed:
            action[0] -= self.cfg.pos_sensitivity
        return action
```

The policy contract has two action paths:

- `apply_action` receives a policy action and optionally replaces or modifies
  it during human intervention.
- `get_action` produces a standalone human action for pure teleoperation and
  demonstration recording.

`is_intervening` defines when human control is active. `snapshot` exposes
strategy state and control instructions in the browser dashboard, so include
human-readable bindings there. `reset` must clear episode-local state.

Register the policy in the teleoperation strategy registry:

```python
_REGISTRY.register(MyRobotKeyboardStrategy)
```

The registry key is derived from the class-level `env_type` and `device_type`.
Add the strategy import and registration alongside the built-in strategies in
`teleop/strategies/registry.py`.

For an existing input device, no browser or device change is required. If a
new hardware device is needed, implement `DeviceBase` by defining `reset`,
`handle_event`, and `snapshot`, then add its construction and configuration to
`TeleopController`.

### How intervention enters the environment loop

During rollout, `BaseEnv` checks each configured strategy before every native
environment step. Active strategies transform the proposed policy action, and
the resulting action is sent to the backend and recorder. The transition is
marked with `info.is_intervention`, allowing downstream algorithms to
distinguish autonomous actions from human corrections without changing the
environment API.

During demonstration recording, `get_action` drives the environment directly.
The same normalized observations and recorder strategy are used in both
modes, so demonstrations, interventions, policy rollouts, and evaluation data
share one environment integration.

With these pieces in place, a new backend can participate in the complete
verl-vla workflow: collect demonstrations, run policy rollouts, accept human
corrections, write datasets and videos, evaluate policies, and feed the
resulting experience into subsequent post-training.
