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

"""Teleop input devices.

``LerobotDevice`` depends on the optional ``lerobot`` package (and its motor /
serial stack). Import it lazily so Arena/GR00T eval paths that never enable
teleop do not require ``lerobot`` to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from verl_vla.teleop.devices.device_base import DeviceBase, DeviceEvent
from verl_vla.teleop.devices.gamepad import GamepadDevice, GamepadDeviceCfg
from verl_vla.teleop.devices.keyboard import KeyboardDevice, KeyboardDeviceCfg
from verl_vla.teleop.devices.xr_controller import XRControllerDevice, XRControllerDeviceCfg

if TYPE_CHECKING:
    from verl_vla.teleop.devices.lerobot import LerobotDevice, LerobotDeviceCfg

__all__ = [
    "DeviceBase",
    "DeviceEvent",
    "GamepadDevice",
    "GamepadDeviceCfg",
    "KeyboardDevice",
    "KeyboardDeviceCfg",
    "XRControllerDevice",
    "XRControllerDeviceCfg",
]

_LAZY_ATTRS = {
    "LerobotDevice": ("verl_vla.teleop.devices.lerobot", "LerobotDevice"),
    "LerobotDeviceCfg": ("verl_vla.teleop.devices.lerobot", "LerobotDeviceCfg"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
