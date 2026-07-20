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

import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import override

from verl_vla.teleop.config import LerobotTeleopConfig
from verl_vla.teleop.devices import DeviceBase
from verl_vla.teleop.strategies.base import InterventionStrategyBase


class LiberoLerobotStrategy(InterventionStrategyBase):
    env_type = "libero"
    device_type = "lerobot"
    _GRIPPER_TERM = True

    def __init__(self, cfg: LerobotTeleopConfig | None = None, *, simulator_cfg: Any):
        del simulator_cfg
        lerobot_cfg = cfg or LerobotTeleopConfig()
        super().__init__(lerobot_cfg)
        self.cfg = lerobot_cfg
        self._last_command = self._default_command()
        self._has_pose = False
        self._last_leader_pos: np.ndarray | None = None
        self._last_leader_rot: Rotation | None = None
        self._control_state = "idle"

    @override
    def reset(self) -> None:
        self._last_command = self._default_command()
        self._has_pose = False
        self._last_leader_pos = None
        self._last_leader_rot = None
        self._control_state = "idle"

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        self._process_device_events(device)
        return self._control_state != "idle"

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        self._process_device_events(device)
        while self._control_state == "armed":
            time.sleep(0.05)
            self._process_device_events(device)
        if self._control_state != "active":
            return action
        command = self._command_from_device(device)
        action_array = np.asarray(action)
        if action_array.shape == command.shape:
            return command.astype(action_array.dtype, copy=False)
        if action_array.ndim > 0 and action_array.shape[-1] == command.shape[-1]:
            overridden = action_array.copy()
            overridden[...] = command.astype(action_array.dtype, copy=False)
            return overridden
        return command

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        self._process_device_events(device)
        return {
            "strategy": f"{self.env_type}:{self.device_type}",
            "is_intervening": self._control_state != "idle",
            "active": self._control_state != "idle",
            "control_state": self._control_state,
            "has_pose": self._has_pose,
            "control_mode": "relative",
            "command": self._last_command.astype(float).tolist(),
            "key_bindings": self.key_bindings(),
        }

    def key_bindings(self) -> dict[str, str]:
        return {
            "Space": "enter setup hold / exit intervention",
            "Tab": "start intervention / return to setup hold",
            "Move leader arm": "relative position control",
            "Rotate leader wrist": "relative rotation control when enabled",
            "Leader gripper": "open / close gripper",
            "R": "manual reward",
            "Backspace": "restart recording episode",
            "Enter": "stop recording episode",
        }

    def handle_keyboard_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            event_type = str(event.get("event_type", "")).lower()
            if event_type == "keyboard_event":
                event_type = str((event.get("raw") or {}).get("event_type", "")).lower()
            key_name = str(event.get("code") or event.get("key") or "").upper()
            repeat = bool(event.get("repeat", False))
            if event_type != "keydown" or repeat:
                continue
            if key_name == "SPACE":
                if self._control_state == "idle":
                    self._enter_armed()
                else:
                    self._enter_idle()
            elif key_name == "TAB":
                if self._control_state == "armed":
                    self._enter_active()
                elif self._control_state == "active":
                    self._enter_armed()

    def _process_device_events(self, device: DeviceBase) -> None:
        self.handle_keyboard_events(
            event
            for event in device.drain_events()
            if str(event.get("event_type", "")).lower() in {"keyboard_event", "keydown", "keyup"}
        )

    def _command_from_device(self, device: DeviceBase) -> np.ndarray:
        read_ee_pose = getattr(device, "read_ee_pose", None)
        pose = read_ee_pose() if callable(read_ee_pose) else None
        if pose is None:
            self._has_pose = False
            self._last_command = self._default_command()
            return self._last_command

        leader_pos = np.asarray(pose["pos"], dtype=np.float32)
        leader_rot = Rotation.from_rotvec(np.asarray(pose["rotvec"], dtype=np.float32))
        mapped_pos, mapped_rot = self._leader_pose_to_action_frame(leader_pos, leader_rot)
        gripper = float(pose.get("gripper", 0.0))
        self._has_pose = True

        if self._last_leader_pos is None or self._last_leader_rot is None:
            self._last_leader_pos = mapped_pos.copy()
            self._last_leader_rot = mapped_rot
            self._last_command = self._default_command(gripper=gripper)
            return self._last_command

        delta_pos = mapped_pos - self._last_leader_pos
        delta_rot = (self._last_leader_rot.inv() * mapped_rot).as_rotvec()
        self._last_leader_pos = mapped_pos.copy()
        self._last_leader_rot = mapped_rot

        if bool(self.cfg.enable_position):
            delta_pos = delta_pos * float(self.cfg.pos_sensitivity)
        else:
            delta_pos = np.zeros_like(delta_pos)
        if bool(self.cfg.enable_rotation):
            delta_rot = delta_rot * float(self.cfg.rot_sensitivity)
        else:
            delta_rot = np.zeros_like(delta_rot)
        delta_pos = np.clip(delta_pos, -float(self.cfg.max_pos_delta), float(self.cfg.max_pos_delta))
        delta_rot = np.clip(delta_rot, -float(self.cfg.max_rot_delta), float(self.cfg.max_rot_delta))

        command = np.concatenate([delta_pos, delta_rot]).astype(np.float32)
        if self._GRIPPER_TERM:
            command = np.append(command, self._gripper_action(gripper)).astype(np.float32)
        self._last_command = command
        return command

    def _enter_idle(self) -> None:
        self._control_state = "idle"
        self._last_leader_pos = None
        self._last_leader_rot = None
        self._last_command = self._default_command()

    def _enter_armed(self) -> None:
        self._control_state = "armed"
        self._last_leader_pos = None
        self._last_leader_rot = None
        self._last_command = self._default_command()

    def _enter_active(self) -> None:
        self._control_state = "active"
        self._last_leader_pos = None
        self._last_leader_rot = None
        self._last_command = self._default_command()

    def _leader_pose_to_action_frame(self, leader_pos: np.ndarray, leader_rot: Rotation) -> tuple[np.ndarray, Rotation]:
        position_axes = self._as_int_tuple(self.cfg.position_axes)
        position_signs = self._as_float_array(self.cfg.position_signs)
        position_scale = self._as_float_array(self.cfg.position_scale)
        mapped_pos = leader_pos[list(position_axes)] * position_signs * position_scale

        rotation_axes = self._as_int_tuple(self.cfg.rotation_axes)
        rotation_signs = self._as_float_array(self.cfg.rotation_signs)
        mapped_rot = self._remap_rotation(leader_rot, rotation_axes, tuple(float(sign) for sign in rotation_signs))
        return mapped_pos.astype(np.float32, copy=False), mapped_rot

    def _default_command(self, *, gripper: float | None = None) -> np.ndarray:
        command = np.zeros(6, dtype=np.float32)
        if self._GRIPPER_TERM:
            gripper_action = 0.0 if gripper is None else self._gripper_action(gripper)
            command = np.append(command, gripper_action).astype(np.float32)
        return command

    def _gripper_action(self, gripper: float) -> float:
        return 1.0 if gripper < float(self.cfg.gripper_close_threshold) else -1.0

    @staticmethod
    def _remap_vector(vector: np.ndarray, axes: tuple[int, int, int], signs: tuple[float, float, float]) -> np.ndarray:
        return np.asarray([vector[axis] * sign for axis, sign in zip(axes, signs, strict=True)], dtype=np.float32)

    @staticmethod
    def _remap_rotation(rotation: Rotation, axes: tuple[int, int, int], signs: tuple[float, float, float]) -> Rotation:
        transform = np.zeros((3, 3), dtype=np.float32)
        for output_axis, (input_axis, sign) in enumerate(zip(axes, signs, strict=True)):
            transform[output_axis, input_axis] = sign
        return Rotation.from_matrix(transform @ rotation.as_matrix() @ transform.T)

    @staticmethod
    def _as_int_tuple(value: Any) -> tuple[int, int, int]:
        array = np.asarray(value, dtype=np.int64).reshape(3)
        return int(array[0]), int(array[1]), int(array[2])

    @staticmethod
    def _as_float_array(value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape(3)
