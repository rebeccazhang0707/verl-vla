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

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from typing_extensions import override

from verl_vla.teleop.config import XRControllerTeleopConfig
from verl_vla.teleop.devices import DeviceBase, XRControllerDevice
from verl_vla.teleop.strategies.base import InterventionStrategyBase


class PiperPicoXRStrategy(InterventionStrategyBase):
    """Forward PICO WebXR frames to PiperEnv's QuestArm ROS runtime."""

    env_type = "piper"
    device_type = "xr_controller"

    def __init__(
        self,
        cfg: XRControllerTeleopConfig | None = None,
        *,
        simulator_cfg: Any,
        frame_sink: Callable[[dict[str, Any]], None],
    ):
        super().__init__(cfg or XRControllerTeleopConfig())
        self._action_dim = int(simulator_cfg.action_dim)
        if self._action_dim != 14:
            raise ValueError(f"Dual-Piper action_dim must be 14, got {self._action_dim}")
        self._frame_sink = frame_sink
        self.reset()

    @override
    def reset(self) -> None:
        self._timestamp: float | None = None
        self._active = {"left": False, "right": False}

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        self._forward(cast(XRControllerDevice, device).latest_frame())
        return any(self._active.values())

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        action_array = np.asarray(action)
        if action_array.shape != (self._action_dim,):
            raise ValueError(f"Dual-Piper action must have shape [{self._action_dim}], got {action_array.shape}")
        return np.zeros_like(action_array) if self.is_intervening(device) else action

    @override
    def get_action(self, device: DeviceBase) -> np.ndarray:
        self.is_intervening(device)
        return np.zeros(self._action_dim, dtype=np.float32)

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        self.is_intervening(device)
        return {
            "strategy": "piper:xr_controller",
            "active": any(self._active.values()),
            "active_hands": dict(self._active),
            "backend": "QuestArm ROS",
            "key_bindings": self.key_bindings(),
        }

    def key_bindings(self) -> dict[str, str]:
        return {"A/X": "start", "B/Y": "stop", "Trigger": "gripper"}

    def _forward(self, frame: dict[str, Any]) -> None:
        if not frame:
            return
        timestamp = float(frame.get("timestamp", 0.0))
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp
        controllers = frame.get("controllers")
        if not isinstance(controllers, dict):
            return
        for hand in self._active:
            controller = controllers.get(hand)
            if not isinstance(controller, dict):
                self._active[hand] = False
                continue
            buttons = controller.get("buttons")
            if not isinstance(buttons, dict):
                continue
            if self._pressed(buttons, "primary"):
                self._active[hand] = True
            if self._pressed(buttons, "secondary"):
                self._active[hand] = False
        self._frame_sink(frame)

    @staticmethod
    def _pressed(buttons: dict[str, Any], name: str) -> bool:
        button = buttons.get(name)
        return isinstance(button, dict) and float(button.get("value", button.get("pressed", False))) >= 0.5
