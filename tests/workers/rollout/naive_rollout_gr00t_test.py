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

"""CPU-only mock tests for ``GR00TRolloutRob`` (``naive_rollout_gr00t.py``).

Runs on a **bare CPU env with only torch + numpy** — gr00t / transformers / verl
are all absent here. We therefore (mirroring ``modeling_gr00t_sac_test``):

  * stub the heavy top-level deps in ``sys.modules`` (a minimal functional
    ``verl.DataProto``, ``verl.utils.device.get_device_id``, and a stub
    ``NaiveRolloutRob`` base), then load ``naive_rollout_gr00t.py`` by file path;
  * build the rollout with ``object.__new__`` to bypass ``__init__``, hand-injecting
    a fake ``module`` (``sac_sample_actions`` / ``sac_get_critic_value``).

Coverage (Phase 4 scheme-Y contract — env owns packing + decoding):
  - ``generate_sequences`` emits ONLY the ACTION-slot keys ``action`` (normalised
    chunk), ``critic_value`` (B,), ``log_probs`` (B,);
  - it does NOT emit any ``obs.*`` key (env packs obs) nor an ``action.action`` key
    (the env loop adds the ``action.`` prefix on collation);
  - the obs handed to the model are the **un-prefixed** prompt tensors as-is (no
    rollout-side ``build_inputs``);
  - the critic is evaluated on the SAME chunk that is emitted as ``action``;
  - ``output_critic_value=False`` drops ``critic_value`` and skips the critic call;
  - ``num_action_chunks > horizon`` asserts;
  - ``register_vla_rollouts()`` registers ``("gr00t", *)`` as **string** paths.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SRC = Path(__file__).resolve().parents[3] / "src"


# --------------------------------------------------------------------------- #
# Minimal functional DataProto stub (only the surface the rollout touches).
# --------------------------------------------------------------------------- #
class _FakeDataProto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = dict(batch or {})
        self.non_tensor_batch = dict(non_tensor_batch or {})
        self.meta_info = dict(meta_info or {})

    @classmethod
    def from_dict(cls, tensors=None, non_tensors=None, meta_info=None):
        return cls(batch=tensors or {}, non_tensor_batch=non_tensors or {}, meta_info=meta_info or {})

    def to(self, _device):
        return self

    def __len__(self):
        first = next(iter(self.batch.values()))
        return first.shape[0]


def _ensure_pkg(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        # mark packages so relative imports resolve
        mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
    return mod


def _load_by_path(mod_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_rollout_module():
    """Inject stubs + load ``naive_rollout_gr00t.py`` standalone."""
    # verl.DataProto
    verl = _ensure_pkg("verl")
    verl.DataProto = _FakeDataProto
    # verl.utils.device.get_device_id
    _ensure_pkg("verl.utils")
    dev = _ensure_pkg("verl.utils.device")
    dev.get_device_id = lambda: "cpu"

    # parent packages for the relative ``.naive_rollout_rob`` import
    _ensure_pkg("verl_vla")
    _ensure_pkg("verl_vla.workers")
    _ensure_pkg("verl_vla.workers.rollout")

    # stub NaiveRolloutRob base (the real one loads an OpenVLA checkpoint on init)
    rob = _ensure_pkg("verl_vla.workers.rollout.naive_rollout_rob")

    class _StubNaiveRolloutRob:
        pass

    rob.NaiveRolloutRob = _StubNaiveRolloutRob

    return _load_by_path(
        "verl_vla.workers.rollout.naive_rollout_gr00t",
        _SRC / "verl_vla" / "workers" / "rollout" / "naive_rollout_gr00t.py",
    )


_ROLLOUT_MOD = _load_rollout_module()
GR00TRolloutRob = _ROLLOUT_MOD.GR00TRolloutRob
assert_action_horizon_invariant = _ROLLOUT_MOD.assert_action_horizon_invariant


# --------------------------------------------------------------------------- #
# Fakes: module (sample/critic). No adapter — packing/decoding live in the env.
# --------------------------------------------------------------------------- #
B = 3
HORIZON = 16        # model action horizon
DMAX = 128          # padded (normalised) action width
NUM_CHUNKS = 8      # env-facing chunk length (< HORIZON, exercises slicing)
LANG = 5            # lang token length
STATE_DIM = 128     # padded state width
T = 1               # state horizon
N_PATCHES, C, IH, IW = 1, 3, 8, 8


class _FakeModule:
    def __init__(self):
        self.sample_calls = 0
        self.critic_calls = 0
        self.last_sample_obs = None
        self.last_critic_obs = None
        self.last_critic_actions = None

    def sac_sample_actions(self, obs, tokenizer=None, validate=False):
        self.sample_calls += 1
        self.last_sample_obs = obs
        b = obs.batch["images"].shape[0]
        return {
            "action": torch.randn(b, HORIZON, DMAX),
            "log_probs": torch.randn(b),
        }

    def sac_get_critic_value(self, obs, actions, tokenizer=None):
        self.critic_calls += 1
        self.last_critic_obs = obs
        self.last_critic_actions = actions
        b = obs.batch["images"].shape[0]
        return torch.randn(b)


def _make_rollout(output_critic_value=True):
    """Build a ``GR00TRolloutRob`` via ``object.__new__`` (bypass real init)."""
    r = object.__new__(GR00TRolloutRob)
    r.module = _FakeModule()
    r.output_critic_value = output_critic_value
    r.num_action_chunks = NUM_CHUNKS
    r.action_dim = 26
    r.tokenizer = None
    return r


def _make_prompts():
    """Env-packed obs (scheme Y): un-prefixed eagle tensors in ``prompts.batch``."""
    return _FakeDataProto(
        batch={
            "images": torch.randn(B, N_PATCHES, C, IH, IW),
            "lang_tokens": torch.zeros(B, LANG, dtype=torch.long),
            "lang_masks": torch.ones(B, LANG, dtype=torch.bool),
            "states": torch.randn(B, T, STATE_DIM),
        },
        non_tensor_batch={"task_descriptions": ["pick up the bottle"] * B},
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_generate_sequences_output_keys_and_shapes():
    r = _make_rollout(output_critic_value=True)
    out = r.generate_sequences(_make_prompts())
    b = out.batch

    # ONLY the ACTION-slot keys (env loop adds the ``action.`` prefix on collation).
    assert set(b.keys()) == {"action", "critic_value", "log_probs"}

    # normalised env-facing action chunk
    assert b["action"].shape == (B, NUM_CHUNKS, DMAX)
    # critic + log probs
    assert b["critic_value"].shape == (B,)
    assert b["log_probs"].shape == (B,)


def test_no_obs_or_prefixed_action_emitted():
    r = _make_rollout()
    out = r.generate_sequences(_make_prompts())
    b = out.batch

    # the env produces obs.* and the env loop produces action.* — NOT this rollout.
    assert not any(k.startswith("obs.") for k in b)
    assert "action.action" not in b
    assert "images" not in b and "states" not in b


def test_model_obs_are_unprefixed_prompts_as_is():
    r = _make_rollout()
    prompts = _make_prompts()
    r.generate_sequences(prompts)

    # the model received the prompt DataProto directly (un-prefixed obs keys)
    model_obs = r.module.last_sample_obs
    assert model_obs is prompts
    assert set(model_obs.batch.keys()) == {"images", "lang_tokens", "lang_masks", "states"}
    # the critic saw the same un-prefixed obs object
    assert r.module.last_critic_obs is prompts


def test_critic_evaluated_on_emitted_action_chunk():
    r = _make_rollout(output_critic_value=True)
    out = r.generate_sequences(_make_prompts())
    # the critic is fed the SAME chunk that is emitted as the env-facing action.
    critic_action = r.module.last_critic_actions["action"]
    assert critic_action.shape == (B, NUM_CHUNKS, DMAX)
    assert torch.equal(critic_action, out.batch["action"])
    assert r.module.sample_calls == 1
    assert r.module.critic_calls == 1


def test_output_critic_value_false_skips_critic():
    r = _make_rollout(output_critic_value=False)
    out = r.generate_sequences(_make_prompts())
    assert "critic_value" not in out.batch
    assert r.module.critic_calls == 0
    # sampling still happens; the env-facing action is still emitted
    assert "action" in out.batch
    assert r.module.sample_calls == 1


def test_num_action_chunks_exceeding_horizon_asserts():
    r = _make_rollout()
    r.num_action_chunks = HORIZON + 1  # exceeds model action horizon
    with pytest.raises(AssertionError):
        r.generate_sequences(_make_prompts())


# --------------------------------------------------------------------------- #
# #2 invariant: critic_action_horizon <= num_action_chunks <= action_horizon
# (the __init__ fail-fast guard, extracted into a pure helper so the lower bound
# is testable on a gr00t-free CPU host).
# --------------------------------------------------------------------------- #
def test_action_horizon_invariant_ok():
    # equality on both bounds is allowed
    assert_action_horizon_invariant(num_action_chunks=16, critic_action_horizon=16, action_horizon=50)
    assert_action_horizon_invariant(num_action_chunks=10, critic_action_horizon=8, action_horizon=50)


def test_action_horizon_invariant_too_small_truncates_critic():
    # num_action_chunks < critic_action_horizon -> critic input silently truncated
    with pytest.raises(AssertionError):
        assert_action_horizon_invariant(num_action_chunks=6, critic_action_horizon=10, action_horizon=50)


def test_action_horizon_invariant_too_large_exceeds_model():
    # num_action_chunks > action_horizon -> model can't emit enough steps
    with pytest.raises(AssertionError):
        assert_action_horizon_invariant(num_action_chunks=64, critic_action_horizon=16, action_horizon=50)


# --------------------------------------------------------------------------- #
# Registry: ("gr00t", *) registered as string paths (no import triggered).
# --------------------------------------------------------------------------- #
def test_register_vla_rollouts_registers_gr00t_as_string_path():
    # stub the verl base registry module that base.py imports
    verl_base = _ensure_pkg("verl.workers.rollout.base")
    verl_base._ROLLOUT_REGISTRY = {}

    base_mod = _load_by_path(
        "verl_vla.workers.rollout._base_under_test",
        _SRC / "verl_vla" / "workers" / "rollout" / "base.py",
    )
    base_mod.register_vla_rollouts()

    registry = verl_base._ROLLOUT_REGISTRY
    for mode in ("sync", "async", "async_envloop"):
        assert ("gr00t", mode) in registry
        value = registry[("gr00t", mode)]
        assert isinstance(value, str)  # string path → lazy import, no eager load
        assert value == "verl_vla.workers.rollout.naive_rollout_gr00t.GR00TRolloutRob"
