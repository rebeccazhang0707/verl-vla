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

"""Test config for the arena_env tests.

``embodiment.py`` resolves the GR1 ⇄ Arena-sim joint index tables at import time
(``_resolve_maps()``). When the ``isaaclab_arena_gr00t`` package is not installed,
that lookup needs ``ARENA_GR1_JOINT_SPACE_DIR``. If this repo ships the Arena
joint-space YAMLs under the ``IsaacLab-Arena`` submodule, point the env var at them
(via ``setdefault``, so an explicit override always wins) so the mapping tests run
for real instead of skipping.
"""

import importlib
import os
import sys
import types
from pathlib import Path

# tests/envs/arena_env/conftest.py -> repo root is three parents up from this dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GR1_JOINT_DIR = _REPO_ROOT / "IsaacLab-Arena" / "isaaclab_arena_gr00t" / "embodiments" / "gr1"

if _GR1_JOINT_DIR.is_dir():
    os.environ.setdefault("ARENA_GR1_JOINT_SPACE_DIR", str(_GR1_JOINT_DIR))


def _install_models_namespace_stub() -> None:
    """Let arena_env / embodiment import their pure-numpy ``verl_vla.models.gr00t``
    leaf modules on a minimal CPU host.

    ``verl_vla/models/__init__.py`` eagerly calls ``register_vla_models`` (which
    imports transformers + the whole pi0/openvla stack). On a bare CPU env those
    deps are absent, so importing **any** ``verl_vla.models.*`` submodule would
    fail at the package ``__init__`` and force the env tests to self-skip. The
    leaf modules arena_env actually needs (``models.gr00t.utils`` /
    ``models.gr00t.gr00t_policy``) are pure numpy/torch, so we bypass the package
    ``__init__`` by registering a lightweight namespace package (real ``__path__``,
    no ``register_vla_models``). Only done when the real package can't import (a
    full Isaac/Docker env imports it normally and is left untouched).
    """
    if "verl_vla.models" in sys.modules:
        return
    try:
        importlib.import_module("verl_vla.models")
        return  # full env: real package imported fine, leave as-is.
    except Exception:
        pass
    models_dir = _REPO_ROOT / "src" / "verl_vla" / "models"
    if not models_dir.is_dir():
        return
    stub = types.ModuleType("verl_vla.models")
    stub.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
    sys.modules["verl_vla.models"] = stub


_install_models_namespace_stub()
