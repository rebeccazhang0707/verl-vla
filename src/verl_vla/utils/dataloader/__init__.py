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

"""LeRobot dataloader helpers.

``build_lerobot_sft_dataloader`` depends on the optional ``lerobot`` package.
Import it lazily so Arena/GR00T eval (and other non-SFT paths) that only need
``LeRobotDataLoaderConfig`` do not require ``lerobot`` to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import LeRobotDataLoaderConfig

if TYPE_CHECKING:
    from .lerobot import build_lerobot_sft_dataloader, resolve_multiprocessing_context

__all__ = ["LeRobotDataLoaderConfig", "build_lerobot_sft_dataloader", "resolve_multiprocessing_context"]

_LAZY_ATTRS = {
    "build_lerobot_sft_dataloader": (".lerobot", "build_lerobot_sft_dataloader"),
    "resolve_multiprocessing_context": (".lerobot", "resolve_multiprocessing_context"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        from importlib import import_module

        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
