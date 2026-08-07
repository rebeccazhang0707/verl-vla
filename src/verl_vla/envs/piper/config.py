# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from verl.base_config import BaseConfig


@dataclass
class PiperConfig(BaseConfig):
    """Dual Piper X environment backed exclusively by QuestArm ROS."""

    simulator_type: str = "piper"
    can_channels: list[str] = field(default_factory=lambda: ["can0", "can1"])
    action_dim: int = field(default=14, init=False)
    state_dim: int = field(default=28, init=False)
    task_description: str = "Teleoperate the Piper arms."
    ros_conda_sh: str = ""
    ros_conda_env: str = "vt"
    questarm_setup_path: str = ""
    position_scale: float = 1.25
    rotation_scale: float = 1.25
    keyboard_position_step: float = 0.02
    keyboard_rotation_step: float = 0.10
    ik_position_weight: float = 5.0
    ik_smooth_weight: float = 0.5
    initial_joint_angles: list[list[float]] | None = None
    reset_duration_s: float = 3.0
    reset_timeout_s: float = 15.0
    reset_joint_tolerance: float = 0.03
    gripper_open_width: float = 0.1
    gripper_close_width: float = 0.0
    gripper_width_step: float = 0.005
    gripper_force: float = 1.0
    camera_devices: list[str] = field(default_factory=lambda: ["/dev/video0", "/dev/video2", "/dev/video4"])
    camera_names: list[str] = field(default_factory=lambda: ["front", "side", "wrist"])
    image_height: int = 480
    image_width: int = 640
    camera_fps: int = 20
    camera_fourcc: str = "MJPG"

    def __post_init__(self):
        if len(self.can_channels) != 2:
            raise ValueError(f"QuestArm Piper requires exactly two CAN channels, got {self.can_channels}")
        if (
            min(
                self.position_scale,
                self.rotation_scale,
                self.keyboard_position_step,
                self.keyboard_rotation_step,
            )
            <= 0
        ):
            raise ValueError("Piper teleop scales must be positive")
        if self.ik_position_weight <= 0 or self.ik_smooth_weight < 0:
            raise ValueError("IK position weight must be positive and smooth weight must be non-negative")
        if self.initial_joint_angles is not None:
            initial_joint_angles = np.asarray(self.initial_joint_angles, dtype=float)
            if initial_joint_angles.shape != (2, 6):
                raise ValueError(f"initial_joint_angles must have shape [2, 6], got {initial_joint_angles.shape}")
            if not np.all(np.isfinite(initial_joint_angles)):
                raise ValueError("initial_joint_angles must contain only finite values")
        if self.reset_duration_s <= 0:
            raise ValueError(f"reset_duration_s must be positive, got {self.reset_duration_s}")
        if self.reset_timeout_s <= self.reset_duration_s:
            raise ValueError(
                f"reset_timeout_s must be greater than reset_duration_s, got "
                f"{self.reset_timeout_s} and {self.reset_duration_s}"
            )
        if self.reset_joint_tolerance <= 0:
            raise ValueError(f"reset_joint_tolerance must be positive, got {self.reset_joint_tolerance}")
        if self.gripper_close_width > self.gripper_open_width:
            raise ValueError("gripper_close_width must not exceed gripper_open_width")
        if self.gripper_width_step <= 0:
            raise ValueError(f"gripper_width_step must be positive, got {self.gripper_width_step}")
        if self.gripper_force < 0:
            raise ValueError(f"gripper_force must be non-negative, got {self.gripper_force}")
        if len(self.camera_devices) != len(self.camera_names):
            raise ValueError(
                f"camera_devices and camera_names must have the same length, got "
                f"{len(self.camera_devices)} and {len(self.camera_names)}"
            )
