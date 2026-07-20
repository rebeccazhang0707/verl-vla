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


class PiperKeyboardStrategy(InterventionStrategyBase):
    env_type = "piper"
    device_type = "keyboard"

    def __init__(self, cfg: KeyboardTeleopConfig | None = None, *, simulator_cfg: Any):
        keyboard_cfg = cfg or KeyboardTeleopConfig()
        super().__init__(keyboard_cfg)
        self.cfg = keyboard_cfg
        self._active_arm_index = 0
        self._action_dim = 14
        self._arm_count = 2
        self._configure_from_action_dim(int(simulator_cfg.action_dim))

    @override
    def reset(self) -> None:
        self._active_arm_index = 0

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        keys = self._pressed_keys(device)
        self._select_arm_from_keys(keys)
        return bool(self._pressed_motion_keys(keys) or keys & {"O", "K"})

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        action_array = np.asarray(action)
        self._validate_action_shape(action_array)
        command = self.get_action(device)
        return command.astype(action_array.dtype, copy=False)

    @override
    def get_action(self, device: DeviceBase) -> Any:
        return self._command_from_device(device)

    def _command_from_device(self, device: DeviceBase) -> np.ndarray:
        command = np.zeros(self._action_dim, dtype=np.float32)
        keys = self._pressed_keys(device)
        self._select_arm_from_keys(keys)
        arm_offset = self._active_arm_index * 7
        for key_name, (axis_idx, sign, sensitivity) in self._ee_key_map().items():
            if key_name in keys:
                command[arm_offset + axis_idx] += sign * float(sensitivity(self.cfg))
        if "O" in keys:
            command[arm_offset + 6] += 1.0
        if "K" in keys:
            command[arm_offset + 6] -= 1.0
        return command

    def _validate_action_shape(self, action: np.ndarray) -> None:
        if action.ndim != 1:
            raise ValueError(f"Piper keyboard action must be 1-D per env, got shape {action.shape}")
        if int(action.shape[0]) != self._action_dim:
            raise ValueError(f"Piper keyboard expected action_dim {self._action_dim}, got {action.shape[0]}")

    def _configure_from_action_dim(self, action_dim: int) -> None:
        if action_dim <= 0 or action_dim % 7 != 0:
            raise ValueError(f"Piper keyboard action_dim must be a positive multiple of 7, got {action_dim}")
        self._action_dim = action_dim
        self._arm_count = self._action_dim // 7
        self._active_arm_index = min(self._active_arm_index, self._arm_count - 1)

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        command = self._command_from_device(device)
        return {
            "strategy": f"{self.env_type}:{self.device_type}",
            "is_intervening": self.is_intervening(device),
            "active_arm": self._active_arm_index,
            "arm_count": self._arm_count,
            "command": command.astype(float).tolist(),
            "unit": "m/rad ee pose delta + gripper command",
            "key_bindings": self.key_bindings(),
        }

    def key_bindings(self) -> dict[str, str]:
        return {
            "1..9": "select active Piper arm",
            "W/S": "+x / -x",
            "A/D": "+y / -y",
            "Q/E": "+z / -z",
            "Z/X": "+roll / -roll",
            "T/G": "+pitch / -pitch",
            "C/V": "+yaw / -yaw",
            "O/K": "open / close gripper",
            "R": "manual reward",
            "Backspace": "restart recording episode",
            "Enter": "stop recording episode",
        }

    def _pressed_keys(self, device: DeviceBase) -> set[str]:
        return set(device.snapshot().get("pressed_keys", []))

    def _pressed_motion_keys(self, keys: set[str]) -> set[str]:
        return keys & set(self._ee_key_map())

    def _select_arm_from_keys(self, keys: set[str]) -> None:
        for key_name in sorted(keys):
            if not key_name.isdigit():
                continue
            arm_index = int(key_name) - 1
            if 0 <= arm_index < self._arm_count:
                self._active_arm_index = arm_index

    @staticmethod
    def _ee_key_map():
        return {
            "W": (0, 1.0, lambda cfg: cfg.pos_sensitivity),
            "S": (0, -1.0, lambda cfg: cfg.pos_sensitivity),
            "A": (1, 1.0, lambda cfg: cfg.pos_sensitivity),
            "D": (1, -1.0, lambda cfg: cfg.pos_sensitivity),
            "Q": (2, 1.0, lambda cfg: cfg.pos_sensitivity),
            "E": (2, -1.0, lambda cfg: cfg.pos_sensitivity),
            "Z": (3, 1.0, lambda cfg: cfg.rot_sensitivity),
            "X": (3, -1.0, lambda cfg: cfg.rot_sensitivity),
            "T": (4, 1.0, lambda cfg: cfg.rot_sensitivity),
            "G": (4, -1.0, lambda cfg: cfg.rot_sensitivity),
            "C": (5, 1.0, lambda cfg: cfg.rot_sensitivity),
            "V": (5, -1.0, lambda cfg: cfg.rot_sensitivity),
        }
