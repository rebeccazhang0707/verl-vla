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

from verl.base_config import BaseConfig

SUPPORTED_PIPER_ROBOT_MODELS = ("piper", "piper_h", "piper_l", "piper_x")


@dataclass
class PiperConfig(BaseConfig):
    """Minimal real Piper arm environment configuration."""

    simulator_type: str = "piper"
    can_channels: list[str] = field(default_factory=lambda: ["can0", "can1"])
    arm_names: list[str] = field(default_factory=lambda: ["left", "right"])
    robot_models: list[str] = field(default_factory=lambda: ["piper_x", "piper_x"])
    can_interface: str = "socketcan"
    bitrate: int = 1_000_000
    firmware_version: str = "v189"
    speed_percent: int = 10
    reset_speed_percent: int = 5
    reset_timeout_s: float = 15.0
    reset_poll_interval_s: float = 0.1
    reset_joint_tolerance: float = 0.03
    initial_joint_angles: list[list[float]] | None = None
    action_dim: int = 14
    state_dim: int = 28
    task_description: str = "Teleoperate the Piper arms."
    gripper_open_width: float = 0.105
    gripper_close_width: float = 0.0
    gripper_width_step: float = 0.005
    gripper_force: float = 1.0
    camera_devices: list[str] = field(default_factory=lambda: ["/dev/video0", "/dev/video2", "/dev/video4"])
    camera_names: list[str] = field(default_factory=lambda: ["front", "side", "wrist"])
    image_height: int = 480
    image_width: int = 640
    camera_fps: int = 20
    camera_fourcc: str = "MJPG"
    ik_jacobian_eps: float = 1e-4
    max_joint_delta_per_step: float = 0.04

    def __post_init__(self):
        if len(self.can_channels) != len(self.arm_names):
            raise ValueError(
                f"can_channels and arm_names must have the same length, got "
                f"{len(self.can_channels)} and {len(self.arm_names)}"
            )
        arm_count = len(self.can_channels)
        robot_models = list(self.robot_models)
        if len(robot_models) != arm_count:
            raise ValueError(f"robot_models must contain {arm_count} per-arm model names, got {len(robot_models)}")
        unsupported_models = sorted(set(robot_models) - set(SUPPORTED_PIPER_ROBOT_MODELS))
        if unsupported_models:
            raise ValueError(
                f"Unsupported Piper robot_models {unsupported_models}; expected one of "
                f"{list(SUPPORTED_PIPER_ROBOT_MODELS)}"
            )
        object.__setattr__(self, "action_dim", arm_count * 7)
        object.__setattr__(self, "state_dim", arm_count * 14)
        if not 0 <= int(self.speed_percent) <= 100:
            raise ValueError(f"speed_percent must be in [0, 100], got {self.speed_percent}")
        if not 0 <= int(self.reset_speed_percent) <= 100:
            raise ValueError(f"reset_speed_percent must be in [0, 100], got {self.reset_speed_percent}")
        if self.reset_timeout_s <= 0:
            raise ValueError(f"reset_timeout_s must be positive, got {self.reset_timeout_s}")
        if self.reset_poll_interval_s <= 0:
            raise ValueError(f"reset_poll_interval_s must be positive, got {self.reset_poll_interval_s}")
        if self.reset_joint_tolerance <= 0:
            raise ValueError(f"reset_joint_tolerance must be positive, got {self.reset_joint_tolerance}")
        if self.initial_joint_angles is not None:
            if len(self.initial_joint_angles) != arm_count:
                raise ValueError(
                    f"initial_joint_angles must contain {arm_count} arm targets, got {len(self.initial_joint_angles)}"
                )
            for arm_idx, joints in enumerate(self.initial_joint_angles):
                if len(joints) != 6:
                    raise ValueError(f"initial_joint_angles[{arm_idx}] must contain 6 joints, got {len(joints)}")
        if len(self.camera_devices) != len(self.camera_names):
            raise ValueError(
                f"camera_devices and camera_names must have the same length, got "
                f"{len(self.camera_devices)} and {len(self.camera_names)}"
            )
