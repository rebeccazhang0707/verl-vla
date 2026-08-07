# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from verl.base_config import BaseConfig

_PIPER_MODELS = {"piper", "piper_h", "piper_l", "piper_x"}
_ARM_NAMES = {"left", "right"}


@dataclass
class PiperArmConfig:
    """One physical Piper and its logical teleoperation hand."""

    name: str
    can_channel: str
    model: str = "piper_x"
    initial_joint_angles: list[float] | None = None

    def __post_init__(self) -> None:
        if self.name not in _ARM_NAMES:
            raise ValueError(f"Piper arm name must be left or right, got {self.name!r}")
        if self.model not in _PIPER_MODELS:
            raise ValueError(f"Unsupported Piper model {self.model!r}; choose from {sorted(_PIPER_MODELS)}")
        if not self.can_channel:
            raise ValueError("Piper can_channel must not be empty")
        if self.initial_joint_angles is not None:
            angles = np.asarray(self.initial_joint_angles, dtype=float)
            if angles.shape != (6,) or not np.all(np.isfinite(angles)):
                raise ValueError(f"{self.name} initial_joint_angles must contain six finite values")


@dataclass
class PiperCameraConfig:
    """One V4L2 camera exposed as a named observation."""

    name: str
    device: str
    width: int = 640
    height: int = 480
    pixel_format: str = "YUYV"

    def __post_init__(self) -> None:
        if not self.name or not self.device:
            raise ValueError("Piper camera name and device must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Piper camera dimensions must be positive, got {self.width}x{self.height}")


@dataclass
class PiperConfig(BaseConfig):
    """One- or two-arm Piper environment backed exclusively by QuestArm ROS."""

    simulator_type: str = "piper"
    arms: list[PiperArmConfig] = field(
        default_factory=lambda: [
            PiperArmConfig(name="left", can_channel="can0"),
            PiperArmConfig(name="right", can_channel="can1"),
        ]
    )
    cameras: list[PiperCameraConfig] = field(default_factory=list)
    action_dim: int = field(init=False)
    state_dim: int = field(init=False)
    task_description: str = "Teleoperate the Piper arms."
    ik_position_weight: float = 5.0
    ik_smooth_weight: float = 0.5
    reset_duration_s: float = 3.0
    reset_timeout_s: float = 15.0
    reset_joint_tolerance: float = 0.03
    gripper_open_width: float = 0.1
    gripper_close_width: float = 0.0
    gripper_width_step: float = 0.005
    gripper_force: float = 1.0

    def __post_init__(self) -> None:
        arms = [arm if isinstance(arm, PiperArmConfig) else PiperArmConfig(**dict(arm)) for arm in self.arms]
        cameras = [
            camera if isinstance(camera, PiperCameraConfig) else PiperCameraConfig(**dict(camera))
            for camera in self.cameras
        ]
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "cameras", cameras)
        if not 1 <= len(arms) <= 2:
            raise ValueError(f"Piper WebXR/keyboard teleoperation supports one or two arms, got {len(arms)}")
        arm_names = [arm.name for arm in arms]
        if len(set(arm_names)) != len(arm_names):
            raise ValueError(f"Piper arm names must be unique, got {arm_names}")
        camera_names = [camera.name for camera in cameras]
        if len(set(camera_names)) != len(camera_names):
            raise ValueError(f"Piper camera names must be unique, got {camera_names}")
        object.__setattr__(self, "action_dim", 7 * len(arms))
        object.__setattr__(self, "state_dim", 14 * len(arms))
        if self.ik_position_weight <= 0 or self.ik_smooth_weight < 0:
            raise ValueError("IK position weight must be positive and smooth weight must be non-negative")
        if self.reset_duration_s <= 0 or self.reset_timeout_s <= self.reset_duration_s:
            raise ValueError("reset_timeout_s must be greater than the positive reset_duration_s")
        if self.reset_joint_tolerance <= 0:
            raise ValueError("reset_joint_tolerance must be positive")
        if self.gripper_close_width > self.gripper_open_width:
            raise ValueError("gripper_close_width must not exceed gripper_open_width")
        if self.gripper_width_step <= 0 or self.gripper_force < 0:
            raise ValueError("gripper_width_step must be positive and gripper_force non-negative")
