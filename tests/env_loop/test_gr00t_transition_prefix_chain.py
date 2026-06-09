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

"""Integration test locking the GR00T scheme-Y obs/action prefix chain.

This pins the data-flow contract Phase 4 depends on, end-to-end on CPU:

  * the **env** emits the packed eagle obs (``images / lang_tokens / lang_masks /
    states``) → collated under the ``obs.`` prefix → sliced into ``t0.obs.*`` /
    ``t1.obs.*`` by ``add_transition_prefixes``;
  * the **rollout** emits ONLY the normalised ``action`` (+ critic_value /
    log_probs) → collated under the ``action.`` prefix → ``t0.action.action`` is
    the **normalised** action that the critic / replay consume.

We run the real ``stack_dataproto_with_padding`` + ``add_transition_prefixes``
(loaded by file path with a stubbed ``verl.DataProto``, since neither ``verl`` nor
``transformers`` is installed on the minimal CPU host) over a synthetic
trajectory built from a fake env-obs and the simplified rollout output.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SRC = Path(__file__).resolve().parents[1].parent / "src"

B = 2
NUM_STEPS = 3           # trajectory slots; add_transition_prefixes needs > 1
N_PATCHES, C, IH, IW = 1, 3, 8, 8
LANG = 5
STATE_DIM = 128
T = 1
NUM_CHUNKS = 8          # env-facing chunk length
DMAX = 128              # padded (normalised) action width


class _FakeDataProto:
    """Minimal DataProto surface used by data.py (dict-backed batch)."""

    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = dict(batch) if batch is not None else {}
        self.non_tensor_batch = dict(non_tensor_batch or {})
        self.meta_info = dict(meta_info or {})

    @classmethod
    def from_dict(cls, tensors=None, non_tensors=None, meta_info=None):
        return cls(batch=tensors or {}, non_tensor_batch=non_tensors or {}, meta_info=meta_info or {})


def _ensure_pkg(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
    return mod


def _load_by_path(mod_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_data_module():
    """Load ``verl_vla.utils.data`` by path with a stubbed ``verl.DataProto``.

    ``verl_vla.utils.keys`` (pure constants, import-clean) is left to import for
    real — we deliberately do NOT stub the ``verl_vla`` namespace so later tests
    that import the real package are unaffected.
    """
    verl = _ensure_pkg("verl")
    verl.DataProto = _FakeDataProto

    return _load_by_path("_gr00t_data_under_test", _SRC / "verl_vla" / "utils" / "data.py")


_DATA = _load_data_module()
stack_dataproto_with_padding = _DATA.stack_dataproto_with_padding
add_transition_prefixes = _DATA.add_transition_prefixes


def _env_obs_slot():
    """One env-emitted obs slot (the packed eagle tensors, un-prefixed)."""
    return _FakeDataProto(
        batch={
            "images": torch.randn(B, N_PATCHES, C, IH, IW),
            "lang_tokens": torch.zeros(B, LANG, dtype=torch.long),
            "lang_masks": torch.ones(B, LANG, dtype=torch.bool),
            "states": torch.randn(B, T, STATE_DIM),
        }
    )


def _rollout_action_slot():
    """One rollout-emitted action slot (simplified scheme-Y output)."""
    return _FakeDataProto(
        batch={
            "action": torch.randn(B, NUM_CHUNKS, DMAX),  # normalised chunk
            "critic_value": torch.randn(B),
            "log_probs": torch.randn(B),
        }
    )


def test_obs_and_action_prefix_chain_to_t0_t1():
    obs_slots = [_env_obs_slot() for _ in range(NUM_STEPS)]
    action_slots = [_rollout_action_slot() for _ in range(NUM_STEPS)]

    # env loop collation: env obs → ``obs.*``; rollout output → ``action.*``.
    merged = {}
    merged.update(stack_dataproto_with_padding(obs_slots, "obs"))
    merged.update(stack_dataproto_with_padding(action_slots, "action"))

    # Collated keys carry exactly the env/rollout provenance prefixes.
    assert "obs.images" in merged and merged["obs.images"].shape == (B, NUM_STEPS, N_PATCHES, C, IH, IW)
    assert "action.action" in merged and merged["action.action"].shape == (B, NUM_STEPS, NUM_CHUNKS, DMAX)
    # the rollout did NOT emit obs.* nor a doubly-nested action.action.
    assert "action.obs.images" not in merged
    assert "action.action.action" not in merged

    data = _FakeDataProto(batch=merged)
    add_transition_prefixes(data)
    batch = data.batch

    # obs slots (from the env) → t0.obs.* / t1.obs.*
    assert batch["t0.obs.images"].shape == (B, NUM_STEPS - 1, N_PATCHES, C, IH, IW)
    assert batch["t1.obs.images"].shape == (B, NUM_STEPS - 1, N_PATCHES, C, IH, IW)
    assert "t0.obs.states" in batch and "t1.obs.states" in batch
    assert "t0.obs.lang_tokens" in batch and "t1.obs.lang_tokens" in batch

    # rollout action (normalised) → t0.action.action (the replay / critic space)
    assert batch["t0.action.action"].shape == (B, NUM_STEPS - 1, NUM_CHUNKS, DMAX)
    # value matches the first (t0) view of the collated normalised action.
    assert torch.equal(batch["t0.action.action"], merged["action.action"][:, :-1])
    assert torch.equal(batch["t1.action.action"], merged["action.action"][:, 1:])
    assert torch.equal(batch["t0.obs.images"], merged["obs.images"][:, :-1])
