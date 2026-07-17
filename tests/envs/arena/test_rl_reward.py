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

"""Tests for the Arena RL reward cfg patch (sim owns auto-reset).

The Arena task has no native reward, so ``apply_arena_rl_reward`` turns the
``success`` termination into a ``RewTerm(weight=1/step_dt)`` -- WITHOUT touching the
termination terms (IsaacLab keeps owning per-step auto-reset).

It imports ``isaaclab`` lazily, so we inject tiny fake ``isaaclab.managers`` /
``isaaclab.utils`` modules to exercise the patch on a host without Isaac Sim.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from verl_vla.envs.arena.utils import (
    apply_arena_rl_reward,
    arena_subtask_graded_reward,
    arena_success_reward,
)


@dataclasses.dataclass
class _FakeTerm:
    func: object = None
    params: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _FakeTerminationsCfg:
    success: _FakeTerm | None = None
    object_dropped: _FakeTerm | None = None
    time_out: _FakeTerm | None = None


@dataclasses.dataclass
class _FakeSim:
    dt: float = 1.0 / 200.0


class _FakeEnvCfg:
    def __init__(self):
        self.terminations = _FakeTerminationsCfg(
            success=_FakeTerm(func=lambda env, **_: env, params={"k": 1}),
            object_dropped=_FakeTerm(func=lambda env, **_: env),
            time_out=_FakeTerm(func=lambda env, **_: env),
        )
        self.sim = _FakeSim()
        self.decimation = 4
        self.rewards = None
        self.episode_length_s = None


@pytest.fixture
def _fake_isaaclab(monkeypatch):
    """Inject minimal fake ``isaaclab.managers`` / ``isaaclab.utils`` for the reward patch."""

    class _RewardTermCfg:
        def __init__(self, func=None, weight=None, params=None):
            self.func = func
            self.weight = weight
            self.params = params or {}

    managers_mod = types.ModuleType("isaaclab.managers")
    managers_mod.RewardTermCfg = _RewardTermCfg
    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.configclass = lambda cls: cls  # identity decorator is enough here
    isaaclab_mod = types.ModuleType("isaaclab")

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils_mod)
    return _RewardTermCfg


def test_apply_reward_installs_rewterm_and_keeps_terminations(_fake_isaaclab):
    cfg = _FakeEnvCfg()
    installed = apply_arena_rl_reward(cfg, subtask_reward=False)

    assert installed is True
    # Reward source is installed at weight 1/step_dt.
    assert cfg.rewards is not None
    assert cfg.rewards.task_success is not None
    assert cfg.rewards.task_success.weight == pytest.approx(1.0 / (cfg.sim.dt * cfg.decimation))
    assert cfg.rewards.task_success.func is arena_success_reward

    # Crucially: termination terms are LEFT IN PLACE so IsaacLab keeps auto-resetting.
    assert cfg.terminations.success is not None
    assert cfg.terminations.object_dropped is not None
    assert cfg.terminations.time_out is not None


class _FakeTermManager:
    def __init__(self, success_vals):
        self._success = success_vals

    @property
    def active_terms(self):
        return ["success", "object_dropped", "time_out"]

    def get_term(self, name):
        import torch

        assert name == "success"
        return torch.tensor(self._success)


class _FakeEnv:
    def __init__(self, success_vals, subtask_state=None):
        import torch

        self.termination_manager = _FakeTermManager(success_vals)
        self.device = torch.device("cpu")
        self.extras = {} if subtask_state is None else {"subtask_success_state": subtask_state}


def test_success_reward_reads_success_term():
    import torch

    env = _FakeEnv(success_vals=[False, True])
    out = arena_success_reward(env)
    assert torch.equal(out, torch.tensor([0.0, 1.0]))


def test_graded_reward_reads_subtask_state():
    import torch

    # 2-subtask task: env0 none done (0.0), env1 first done (0.5), env2 both done (1.0).
    env = _FakeEnv(
        success_vals=[False, False, True],
        subtask_state=[[False, False], [True, False], [True, True]],
    )
    out = arena_subtask_graded_reward(env)
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]))
