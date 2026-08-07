# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from verl_vla.teleop.config import KeyboardTeleopConfig
from verl_vla.teleop.devices import DeviceBase
from verl_vla.teleop.strategies.base import InterventionStrategyBase

_KEY_TO_AXIS = {
    "W": (0, 1.0),
    "S": (0, -1.0),
    "A": (1, 1.0),
    "D": (1, -1.0),
    "Q": (2, 1.0),
    "E": (2, -1.0),
    "Z": (3, 1.0),
    "X": (3, -1.0),
    "T": (4, 1.0),
    "G": (4, -1.0),
    "C": (5, 1.0),
    "V": (5, -1.0),
}


class PiperKeyboardStrategy(InterventionStrategyBase):
    """Map keyboard input to the ROS backend's fixed dual-Piper action."""

    env_type = "piper"
    device_type = "keyboard"

    def __init__(self, cfg: KeyboardTeleopConfig | None = None, *, simulator_cfg: Any):
        cfg = cfg or KeyboardTeleopConfig()
        super().__init__(cfg)
        if int(simulator_cfg.action_dim) != 14:
            raise ValueError(f"Dual-Piper action_dim must be 14, got {simulator_cfg.action_dim}")
        self._position_step = float(simulator_cfg.keyboard_position_step)
        self._rotation_step = float(simulator_cfg.keyboard_rotation_step)
        self._active_arm = 0

    @override
    def reset(self) -> None:
        self._active_arm = 0

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        keys = self._keys(device)
        self._select_arm(keys)
        return bool(keys & (_KEY_TO_AXIS.keys() | {"O", "K"}))

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        action_array = np.asarray(action)
        if action_array.shape != (14,):
            raise ValueError(f"Dual-Piper action must have shape [14], got {action_array.shape}")
        return self.get_action(device).astype(action_array.dtype, copy=False)

    @override
    def get_action(self, device: DeviceBase) -> np.ndarray:
        keys = self._keys(device)
        self._select_arm(keys)
        command = np.zeros(14, dtype=np.float32)
        offset = self._active_arm * 7
        for key in keys & _KEY_TO_AXIS.keys():
            axis, sign = _KEY_TO_AXIS[key]
            scale = self._position_step if axis < 3 else self._rotation_step
            command[offset + axis] += sign * scale
        command[offset + 6] = float("O" in keys) - float("K" in keys)
        return command

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        command = self.get_action(device)
        return {
            "strategy": "piper:keyboard",
            "active_arm": "left" if self._active_arm == 0 else "right",
            "command": command.astype(float).tolist(),
            "unit": "m/rad delta",
            "key_bindings": self.key_bindings(),
        }

    def key_bindings(self) -> dict[str, str]:
        return {
            "1 / 2": "select left / right arm",
            "W/S": "+x / -x",
            "A/D": "+y / -y",
            "Q/E": "+z / -z",
            "Z/X": "+roll / -roll",
            "T/G": "+pitch / -pitch",
            "C/V": "+yaw / -yaw",
            "O/K": "open / close gripper",
        }

    @staticmethod
    def _keys(device: DeviceBase) -> set[str]:
        return set(device.snapshot().get("pressed_keys", []))

    def _select_arm(self, keys: set[str]) -> None:
        if "1" in keys:
            self._active_arm = 0
        elif "2" in keys:
            self._active_arm = 1
