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

from dataclasses import dataclass, field

from hydra.utils import instantiate
from verl.base_config import BaseConfig


@dataclass
class ArenaEnvironmentConfig(BaseConfig):
    """Configuration for one environment hosted by Isaac Lab Arena."""

    action_dim: int = 50
    state_dim: int | None = None
    env_name: str = "galileo_g1_locomanip_pick_and_place"
    object: str | None = None
    embodiment: str = "g1_wbc_joint"
    object_set: str | None = None
    kitchen_style: int = 2
    task_description: str = "Pick and place the brown box."
    camera_names: tuple[str, ...] = ("robot_head_cam_rgb",)
    image_shape: tuple[int, int, int] = (480, 640, 3)
    subtask_reward: bool = False
    env_spacing: float = 30.0
    arena_state_mode: str = "g1_wbc_joint"
    external_env_class_path: str | None = None
    use_policy_action: bool | None = None
    stable_hold_joint_slice: int | None = None
    base_height_index: int | None = None
    base_height_command: float | None = None
    arena_joint_space_spec: str | None = None
    arena_joint_space_dir: str | None = None
    arena_joint_space_policy_yaml: str | None = None
    arena_joint_space_action_yaml: str | None = None
    arena_joint_space_state_yaml: str | None = None


@dataclass
class ArenaLiberoEnvironmentConfig(ArenaEnvironmentConfig):
    """Configuration for the Franka LIBERO environment hosted by Arena."""

    libero_task_suite: str = "libero_10"
    libero_task_id: int = 0
    libero_randomize_object_pose: bool = False
    libero_robot_init_noise_std: float = 0.0
    arena_libero_in_lab_root: str | None = None
    arena_libero_config_dir: str | None = None
    arena_libero_assets_dir: str | None = None
    arena_libero_assembled_dataset_dir: str | None = None


@dataclass
class ArenaSimulatorConfig(BaseConfig):
    """Arena backend config with one selected environment profile."""

    simulator_type: str = "arena"
    environment: str = "g1"

    seed: int = 42
    enable_cameras: bool = True
    sim_dt: float | None = None
    decimation: int | None = None
    render_interval: int | None = None
    rl_success_reward: bool = True
    disable_fabric: bool = False
    solve_relations: bool = True
    enable_pinocchio: bool = True
    placement_seed: int | None = None
    resolve_on_reset: bool | None = None
    presets: str | None = None

    g1: ArenaEnvironmentConfig = field(default_factory=ArenaEnvironmentConfig)
    gr1: ArenaEnvironmentConfig = field(default_factory=ArenaEnvironmentConfig)
    libero: ArenaLiberoEnvironmentConfig = field(default_factory=ArenaLiberoEnvironmentConfig)

    def __post_init__(self) -> None:
        for name, config_type in (
            ("g1", ArenaEnvironmentConfig),
            ("gr1", ArenaEnvironmentConfig),
            ("libero", ArenaLiberoEnvironmentConfig),
        ):
            config = getattr(self, name)
            if not isinstance(config, config_type):
                object.__setattr__(self, name, instantiate(config))
        if self.environment not in {"g1", "gr1", "libero"}:
            raise ValueError(f"Unsupported Arena environment: {self.environment}")

    @property
    def environment_config(self) -> ArenaEnvironmentConfig:
        """Return the typed config for the selected Arena environment."""

        return getattr(self, self.environment)
