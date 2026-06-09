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

"""Unit tests for IsaacLabArenaEnv (scheme Y: env packs obs + decodes actions).

These tests deliberately avoid launching Isaac Sim: instances are created with
``object.__new__`` and the per-method ``isaaclab*`` imports are replaced with
fakes (the env defers every Isaac/omni import inside a method). The GR00T
processor is replaced with a tiny stub adapter so no eagle checkpoint is needed;
the GR1 ⇄ Arena-sim joint maps (``self.joint_map`` = real ``GR1_ARENA``) run for
real (the joint-space YAMLs are located via ``conftest``). ``conftest`` also
installs a ``verl_vla.models`` namespace shim so the pure-numpy gr00t leaf
modules import on a minimal CPU host.
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
arena_env = pytest.importorskip("verl_vla.envs.arena_env.arena_env")
_embodiment = pytest.importorskip("verl_vla.envs.arena_env.embodiment")

IsaacLabArenaEnv = arena_env.IsaacLabArenaEnv
GR1_ARENA = _embodiment.GR1_ARENA

STATE_FULL_DIM = GR1_ARENA.state_full_dim   # 54
SIM_ACTION_DIM = GR1_ARENA.sim_action_dim   # 36
POLICY_DIM = GR1_ARENA.policy_dim           # 26

# eagle-tensor stub shapes used by the fake adapter.
N_PATCHES, C, IH, IW = 1, 3, 8, 8
LANG = 5
STATE_DIM = 128         # padded state width
T = 1                   # state horizon
DMAX = 128              # padded (normalised) action width


class _Cfg(dict):
    """Dict that also supports attribute access (mimics OmegaConf DictConfig)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _make_env(num_envs: int = 4, cfg=None) -> IsaacLabArenaEnv:
    """Construct an IsaacLabArenaEnv without running __init__ (no AppLauncher)."""
    env = object.__new__(IsaacLabArenaEnv)
    env.num_envs = num_envs
    env.device = "cpu"
    env.seed = 123
    env.rank = 0
    env.arena_object = "power_drill"
    env.arena_embodiment = "gr1_joint"
    env.arena_env_name = "galileo_pick_and_place"
    env.task_description = "do the task"
    # set in __init__ from cfg; mirror here since _build_args reads them as attributes.
    cfg = cfg or {}
    env.kitchen_style = cfg.get("kitchen_style", 2)
    env.arena_object_set = cfg.get("object_set", None)
    env.cfg = _Cfg(cfg)
    return env


class _StubAdapter:
    """Stand-in for ``GR00TN16Adapter`` (no eagle checkpoint).

    ``build_inputs`` returns minimal model-ready tensors. ``decode_actions_flat``
    simulates a **relative-action** decode against the base ``raw_state_groups``:
    ``decoded[:, i] = base + normalized_action[:, i, :POLICY_DIM]`` (the base is
    broadcast over the whole chunk). Because the base is reconstructed from the
    groups it is fed, a test can assert the whole chunk was decoded against a
    SINGLE fixed base (the chunk-start state) rather than the per-step live state.
    """

    def __init__(self):
        self.calls = []
        # Each entry is the flat (B, 1, POLICY_DIM) base passed to one decode call;
        # used to verify chunk_step decodes the whole chunk against one fixed base.
        self.decode_bases = []
        self.state_group_dims = GR1_ARENA.spec.state_group_dims

    def build_inputs(self, full_image, state_flat, task_descriptions):
        self.calls.append("build")
        b = full_image.shape[0]
        inputs = {
            "pixel_values": [torch.randn(N_PATCHES, C, IH, IW) for _ in range(b)],
            "input_ids": torch.zeros(b, LANG, dtype=torch.long),
            "attention_mask": torch.ones(b, LANG, dtype=torch.bool),
            "state": torch.randn(b, T, STATE_DIM),
        }
        return inputs, {}

    def decode_actions_flat(self, normalized_action, raw_state_groups):
        self.calls.append("decode")
        # The env now feeds the WHOLE chunk in a single call: (B, chunk, max_action_dim).
        assert normalized_action.ndim == 3
        # Reconstruct the flat (B, 1, POLICY_DIM) base from the per-group raw state so
        # a test can verify the chunk was decoded against a single fixed base.
        base = np.concatenate(
            [np.asarray(raw_state_groups[k], dtype=np.float32) for k in self.state_group_dims],
            axis=-1,
        )  # (B, 1, POLICY_DIM)
        self.decode_bases.append(base.copy())
        # Relative-action decode: absolute = base + delta (base broadcast over chunk).
        delta = np.asarray(normalized_action[:, :, :POLICY_DIM], dtype=np.float32)
        return (base + delta).astype(np.float32)  # (B, chunk, POLICY_DIM)


def _make_io_env(num_envs: int = 2, adapter=None) -> IsaacLabArenaEnv:
    """Env wired for the obs-packing / action-decoding paths (no Isaac)."""
    env = object.__new__(IsaacLabArenaEnv)
    env.num_envs = num_envs
    env.device = "cpu"
    env.camera_name = "robot_pov_cam_rgb"
    env.task_description = "do the task"
    env.joint_map = GR1_ARENA
    env.adapter = adapter if adapter is not None else _StubAdapter()
    env.use_rel_reward = False
    env.prev_step_reward = np.zeros(num_envs)
    env.success_once = np.zeros(num_envs, dtype=bool)
    env.returns = np.zeros(num_envs)
    env._elapsed_steps = np.zeros(num_envs, dtype=np.int32)
    env.max_episode_steps = 100
    env.render_on_chunk_boundary = False
    env._chunk_render_interval_set = False
    env.video_cfg = _Cfg(save_video=False)
    env._last_state26 = np.zeros((num_envs, POLICY_DIM), dtype=np.float32)
    env._last_full_image = None
    return env


def _fake_raw_obs(num_envs: int, cam_name: str = "robot_pov_cam_rgb", h: int = 8, w: int = 8) -> dict:
    return {
        "camera_obs": {cam_name: np.zeros((num_envs, h, w, 3), dtype=np.uint8)},
        "policy": {
            "robot_joint_pos": np.arange(num_envs * STATE_FULL_DIM, dtype=np.float32).reshape(
                num_envs, STATE_FULL_DIM
            )
        },
    }


def _install_fake_arena_modules(monkeypatch, example_environments, builder_cls=None):
    """Inject fake ``isaaclab_arena*`` modules used by IsaacLabArenaEnv._init_env.

    Arena 0.2.0 exposes the example environments via the ``ExampleEnvironments``
    dict in ``isaaclab_arena_environments.cli`` (keyed by ``<EnvClass>.name``).
    """

    def fake_module(name):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child, module)
        return module

    # Parents must exist before children for ``from a.b import c`` to resolve.
    fake_module("isaaclab_arena_environments")
    cli = fake_module("isaaclab_arena_environments.cli")
    cli.ExampleEnvironments = example_environments

    fake_module("isaaclab_arena")
    fake_module("isaaclab_arena.environments")
    builder_mod = fake_module("isaaclab_arena.environments.arena_env_builder")
    if builder_cls is None:

        class _DefaultBuilder:
            def __init__(self, arena_env_, args):  # noqa: ARG002
                pass

            def build_registered(self):
                return SimpleNamespace(), SimpleNamespace()

            def make_registered(self, env_cfg=None):  # noqa: ARG002
                return SimpleNamespace(action_space=None, observation_space=None)

        builder_cls = _DefaultBuilder
    builder_mod.ArenaEnvBuilder = builder_cls


# NOTE: the GR1 ⇄ Arena-sim joint-space mapping (gather_state / scatter_action /
# extract_action) is owned by ``ArenaJointMapping`` and covered in
# ``embodiment_test.py``; the env delegates to ``self.joint_map``.


# ---------------------------------------------------------------------------
# _build_args: regression for the relation-placement / preset fields
# ---------------------------------------------------------------------------


def test_build_args_includes_relation_fields():
    args = _make_env()._build_args()
    for field in ("placement_seed", "resolve_on_reset", "presets"):
        assert hasattr(args, field), f"_build_args must set '{field}' for ArenaEnvBuilder"
    # Defaults mirror the Arena CLI parser.
    assert args.placement_seed is None
    assert args.resolve_on_reset is None
    assert args.presets is None
    assert args.solve_relations is True
    assert args.mimic is False


def test_build_args_threads_through_cfg_overrides():
    env = _make_env(cfg={"placement_seed": 7, "presets": "newton", "resolve_on_reset": True})
    args = env._build_args()
    assert args.placement_seed == 7
    assert args.presets == "newton"
    assert args.resolve_on_reset is True
    assert args.num_envs == env.num_envs
    assert args.object == env.arena_object
    assert args.embodiment == env.arena_embodiment


# ---------------------------------------------------------------------------
# _init_env: regression for the ExampleEnvironments dict lookup (Arena 0.2.0)
# ---------------------------------------------------------------------------


def test_init_env_unknown_name_raises(monkeypatch):
    env = _make_env()
    env.env = None
    env.arena_env_name = "does_not_exist"

    # Dict does not contain the requested env -> ValueError before any get_env().
    example_environments = {"galileo_pick_and_place": object}
    _install_fake_arena_modules(monkeypatch, example_environments)

    with pytest.raises(ValueError, match="does_not_exist"):
        env._init_env()


def test_init_env_uses_example_environments_dict(monkeypatch):
    # rl_success_reward=False so _init_env skips the Isaac-only RL-reward patch.
    env = _make_env(cfg={"video_cfg": _Cfg(save_video=False), "rl_success_reward": False})
    env.env = None

    calls = {}

    class FakeExample:
        def get_env(self, args):
            calls["get_env_args"] = args
            return SimpleNamespace(task=None)

    example_environments = {env.arena_env_name: FakeExample}

    class FakeBuilder:
        def __init__(self, arena_env_, args):
            calls["builder_arena_env"] = arena_env_

        def build_registered(self):
            calls["build_registered"] = True
            return SimpleNamespace(), SimpleNamespace()

        def make_registered(self, env_cfg=None):
            calls["make_registered"] = True
            return SimpleNamespace(action_space="A", observation_space="O")

    _install_fake_arena_modules(monkeypatch, example_environments, builder_cls=FakeBuilder)

    env._init_env()

    assert "get_env_args" in calls
    assert calls["build_registered"] is True
    assert calls["make_registered"] is True
    assert env.action_space == "A"
    assert env.observation_space == "O"


# ---------------------------------------------------------------------------
# Observation packing (scheme Y): env runs the processor → eagle obs slots.
# ---------------------------------------------------------------------------


def test_extract_image_and_state_gathers_26dof():
    env = _make_io_env()
    raw = _fake_raw_obs(env.num_envs)
    full_image, state26 = env._extract_image_and_state(raw)

    assert full_image.shape == (env.num_envs, 8, 8, 3)
    assert state26.shape == (env.num_envs, POLICY_DIM)
    # delegated to the real joint map (54 -> 26 name-based gather)
    expected = GR1_ARENA.gather_state(raw["policy"]["robot_joint_pos"])
    assert np.allclose(state26, expected)


def test_wrap_obs_packs_eagle_keys_and_caches_state():
    env = _make_io_env()
    obs = env._wrap_obs(_fake_raw_obs(env.num_envs))

    ias = obs["images_and_states"]
    assert set(ias.keys()) == {"images", "lang_tokens", "lang_masks", "states"}
    assert ias["images"].shape == (env.num_envs, N_PATCHES, C, IH, IW)
    assert ias["lang_tokens"].shape == (env.num_envs, LANG)
    assert ias["lang_masks"].shape == (env.num_envs, LANG)
    assert ias["states"].shape == (env.num_envs, T, STATE_DIM)

    assert len(obs["task_descriptions"]) == env.num_envs
    # full_image kept at the top level for video (NOT inside images_and_states).
    assert "full_image" in obs and "full_image" not in ias
    # raw 26-DOF joint state cached for the next step's action decode.
    assert env._last_state26.shape == (env.num_envs, POLICY_DIM)


# ---------------------------------------------------------------------------
# step: only scatters an ALREADY-DECODED 26-DOF action (decode moved to chunk_step).
# ---------------------------------------------------------------------------


def test_step_scatters_decoded_action():
    env = _make_io_env()

    class _FakeSim:
        def __init__(self):
            self.received = None

        def step(self, sim_actions):
            self.received = sim_actions
            return (
                _fake_raw_obs(env.num_envs),
                torch.zeros(env.num_envs),
                torch.zeros(env.num_envs, dtype=torch.bool),
                None,
                {},
            )

    env.env = _FakeSim()
    # step now receives an ALREADY-DECODED 26-DOF policy action (decode is owned by
    # chunk_step) and only expands 26 -> 36 before stepping the sim.
    policy_action = torch.full((env.num_envs, POLICY_DIM), 0.5)
    obs, reward, terminations, truncations, infos = env.step(policy_action)

    received = env.env.received
    assert received.shape == (env.num_envs, SIM_ACTION_DIM)
    assert torch.allclose(received, GR1_ARENA.scatter_action(policy_action))
    # the returned obs is the packed eagle obs of the next state.
    assert set(obs["images_and_states"].keys()) == {"images", "lang_tokens", "lang_masks", "states"}


def test_chunk_step_decodes_whole_chunk_against_fixed_base():
    """P0 relative-action fix: the WHOLE chunk is decoded once against the
    chunk-start state, NOT per-step against the (drifting) live state.

    A drifting sim updates ``_last_state26`` every step; a correct fixed-base
    decode must ignore those mid-chunk updates so step i yields ``base + delta_i``
    (not ``base + delta_{i-1} + delta_i``).
    """
    adapter = _StubAdapter()
    env = _make_io_env(num_envs=2, adapter=adapter)

    drift = {"i": 0}

    def _drifting_obs():
        drift["i"] += 1
        rjp = np.full((env.num_envs, STATE_FULL_DIM), float(drift["i"]) * 100.0, dtype=np.float32)
        return {
            "camera_obs": {env.camera_name: np.zeros((env.num_envs, 8, 8, 3), dtype=np.uint8)},
            "policy": {"robot_joint_pos": rjp},
        }

    class _DriftingSim:
        def __init__(self):
            self.received = []

        def step(self, sim_actions):
            self.received.append(sim_actions)
            return (
                _drifting_obs(),
                torch.zeros(env.num_envs),
                torch.zeros(env.num_envs, dtype=torch.bool),
                None,
                {},
            )

    env.env = _DriftingSim()

    # Prime the chunk-start state from a known robot_joint_pos (reset would do this).
    base_rjp = np.full((env.num_envs, STATE_FULL_DIM), 7.0, dtype=np.float32)
    env._wrap_obs(
        {
            "camera_obs": {env.camera_name: np.zeros((env.num_envs, 8, 8, 3), dtype=np.uint8)},
            "policy": {"robot_joint_pos": base_rjp},
        }
    )
    chunk_start_state = env._last_state26.copy()  # (B, 26)
    adapter.calls.clear()
    adapter.decode_bases.clear()

    chunk = 4
    chunk_actions = torch.randn(env.num_envs, chunk, DMAX)
    env.chunk_step(chunk_actions)

    # 1) decode is called EXACTLY ONCE for the whole chunk (not once per step).
    assert adapter.calls.count("decode") == 1
    # 2) the single base passed to decode is the chunk-start state (NOT a live state).
    assert len(adapter.decode_bases) == 1
    assert np.allclose(adapter.decode_bases[0][:, 0, :], chunk_start_state)

    # 3) every executed sim action == scatter(fixed_base + delta_i); i>=1 must NOT
    #    depend on the drifting live state.
    actions_np = chunk_actions.numpy()
    assert len(env.env.received) == chunk
    for i in range(chunk):
        expected_policy = chunk_start_state + actions_np[:, i, :POLICY_DIM]
        expected_sim = GR1_ARENA.scatter_action(torch.from_numpy(expected_policy.astype(np.float32)))
        assert torch.allclose(env.env.received[i], expected_sim, atol=1e-5)


def test_step_none_actions_is_noop():
    env = _make_io_env()
    assert env.step(None) == (None, None, None, None, None)


# ---------------------------------------------------------------------------
# Reward + chunk first-done semantics
# ---------------------------------------------------------------------------


def test_calc_step_reward_absolute_and_relative():
    env = _make_io_env()
    r = np.array([1.0, 2.0])
    assert np.allclose(env._calc_step_reward(r), r)  # absolute (default)

    env.use_rel_reward = True
    env.prev_step_reward = np.zeros(2)
    assert np.allclose(env._calc_step_reward(np.array([1.0, 1.0])), [1.0, 1.0])
    assert np.allclose(env._calc_step_reward(np.array([1.5, 1.0])), [0.5, 0.0])


def test_chunk_step_ever_done_is_monotonic():
    env = _make_io_env()
    rewards_seq = [
        torch.zeros(env.num_envs),
        torch.tensor([1.0, 0.0]),   # env 0 solves the task at step 1
        torch.zeros(env.num_envs),  # ... and must stay "done" afterwards
    ]
    state = {"i": 0}

    def fake_step(actions, critic_values=None):  # noqa: ARG001
        i = state["i"]
        state["i"] += 1
        return (
            {"images_and_states": {}},
            rewards_seq[i],
            torch.zeros(env.num_envs, dtype=torch.bool),
            torch.zeros(env.num_envs, dtype=torch.bool),
            {},
        )

    env.step = fake_step  # chunk_step calls self.step per sub-step
    chunk_actions = torch.zeros(env.num_envs, len(rewards_seq), DMAX)

    _obs, chunk_rewards, chunk_terminations, chunk_truncations, _infos = env.chunk_step(chunk_actions)

    assert chunk_terminations.shape == (env.num_envs, len(rewards_seq))
    # ever_done = cumulative OR of (reward > 0): env 0 latches done from step 1.
    assert chunk_terminations[0].tolist() == [False, True, True]
    assert chunk_terminations[1].tolist() == [False, False, False]


# ---------------------------------------------------------------------------
# State-ID helpers
# ---------------------------------------------------------------------------


def test_get_all_state_ids_returns_dummy_range():
    env = _make_io_env(num_envs=3)
    assert env.get_all_state_ids() == [0, 1, 2]
