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

"""CPU import + argparse smoke for the migrated Arena eval / smoke scripts.

These scripts are **docker-only** to *run* (they need gr00t + transformers 4.51.3 +
Isaac Sim + the checkpoint), but the CPU contract is that they (a) import on a
gr00t-free host — every heavy import is deferred into the functions that need it —
and (b) expose a ``build_parser()`` that parses with the training-aligned anchor
defaults (num_action_chunks / chunk + action_horizon=50). Loaded by file path
because they live under ``examples/`` (not on the package path).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")  # numpy + torch are the only top-level deps of the scripts

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "arena_sac"


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, _EXAMPLES / file_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # executes top-level imports only (no heavy deps)
    return mod


def test_eval_module_imports_cpu():
    mod = _load("_eval_arena_gr00t_under_test", "eval_arena_gr00t.py")
    assert hasattr(mod, "build_parser")
    assert hasattr(mod, "main")
    assert mod.CKPT_ACTION_HORIZON == 50


def test_eval_argparse_defaults_match_training_anchors():
    mod = _load("_eval_arena_gr00t_under_test", "eval_arena_gr00t.py")
    args = mod.build_parser().parse_args(["--ckpt", "/models/ckpt"])
    assert args.ckpt == "/models/ckpt"
    assert args.chunk == 16  # num_action_chunks anchor
    assert args.action_horizon == 50  # checkpoint action horizon anchor
    assert args.env_spacing == 10.0  # source recipe env_spacing
    assert args.embodiment_tag == "gr1"
    assert args.arena_embodiment == "gr1_joint"


def test_eval_argparse_actor_choices():
    mod = _load("_eval_arena_gr00t_under_test", "eval_arena_gr00t.py")
    args = mod.build_parser().parse_args(["--ckpt", "/c", "--actor", "sac"])
    assert args.actor == "sac"
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--ckpt", "/c", "--actor", "bogus"])


def test_smoke_module_imports_cpu():
    mod = _load("_smoke_test_gr00t_arena_under_test", "smoke_test_gr00t_arena.py")
    assert hasattr(mod, "build_parser")
    assert hasattr(mod, "main")
    assert mod.CKPT_ACTION_HORIZON == 50


def test_smoke_argparse_defaults_match_training_anchors():
    mod = _load("_smoke_test_gr00t_arena_under_test", "smoke_test_gr00t_arena.py")
    args = mod.build_parser().parse_args(["--ckpt", "/models/ckpt"])
    assert args.ckpt == "/models/ckpt"
    assert args.chunk == 16
    assert args.action_horizon == 50
    assert args.critic_heads == 10
    assert args.denoise_steps == 2
    assert args.compare_gr00t_policy is False
