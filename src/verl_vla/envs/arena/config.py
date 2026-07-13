# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

from verl.base_config import BaseConfig


@dataclass
class ArenaSimulatorConfig(BaseConfig):
    """Simulator config for Isaac Lab Arena environments."""

    simulator_type: str = "arena"
    seed: int = 42
    action_dim: int = 50
    state_dim: int | None = None

    env_name: str = "galileo_g1_locomanip_pick_and_place"
    # Task object USD id. None => each embodiment supplies its own default in
    # ``add_cli_args`` (G1 WBC -> "brown_box", GR1 fridge -> "ranch_dressing_hope_robolab").
    object: str | None = None
    embodiment: str = "g1_wbc_joint"
    object_set: str | None = None
    kitchen_style: int = 2
    task_description: str = "Pick and place the brown box."
    enable_cameras: bool = True
    camera_names: tuple[str, ...] = ("robot_head_cam_rgb",)
    image_shape: tuple[int, int, int] = (480, 640, 3)
    rl_success_reward: bool = True
    subtask_reward: bool = False

    env_spacing: float = 30.0
    disable_fabric: bool = False
    solve_relations: bool = True
    enable_pinocchio: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    presets: str | None = None

    # Embodiment adapter selector. Defaults to g1_wbc_joint so configs that predate
    # the embodiment abstraction keep working unchanged. See embodiment.py.
    arena_state_mode: str = "g1_wbc_joint"

    # External (non-built-in) Arena environment, registered by "module_path:ClassName"
    # exactly like Arena's --external_environment_class_path (e.g. the LIBERO env).
    external_env_class_path: str | None = None

    # Whether the wrapper steps the raw policy action (True) or routes through the
    # stable-hold / teleop adapter (False). ``None`` => embodiment default
    # (identity G1 WBC: False; mapped GR1 / Franka LIBERO: True). Host here to override.
    use_policy_action: bool | None = None

    # Stable-hold overrides (G1 WBC teleop smoke). ``None`` => use the embodiment class
    # default (G1: 43 / 46 / 0.75; other embodiments: disabled). Host here to override.
    stable_hold_joint_slice: int | None = None
    base_height_index: int | None = None
    base_height_command: float | None = None

    # Joint-space mapping for mapped embodiments (e.g. GR1). ``None`` spec => identity
    # joint-space (G1 WBC). ``arena_joint_space_dir`` is the directory with the three
    # joint-space YAMLs; file names default to the GR1 layout when omitted.
    arena_joint_space_spec: str | None = None
    arena_joint_space_dir: str | None = None
    arena_joint_space_policy_yaml: str | None = None
    arena_joint_space_action_yaml: str | None = None
    arena_joint_space_state_yaml: str | None = None


@dataclass
class ArenaLiberoSimulatorConfig(ArenaSimulatorConfig):
    """Arena simulator config for the Franka LIBERO external environment."""

    # LIBERO task selection (task suite + id within the suite).
    libero_task_suite: str = "libero_10"
    libero_task_id: int = 0
    libero_randomize_object_pose: bool = False
    libero_robot_init_noise_std: float = 0.0
    # None lets the env auto-resolve from the embedded/mounted LIBERO data dirs.
    arena_libero_in_lab_root: str | None = None
    arena_libero_config_dir: str | None = None
    arena_libero_assets_dir: str | None = None
    arena_libero_assembled_dataset_dir: str | None = None
