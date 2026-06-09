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

"""CPU-only test for the SAC actor-loss BC-anchor branch in ``training_worker.py``.

Covers P1-1: the fixed-coefficient BC anchor (``bc_loss_coef``) added alongside the
adaptive TD3+BC path. ``verl`` is not installed on the CPU dev host, so (mirroring
``naive_rollout_gr00t_test``) we stub the heavy top-level deps in ``sys.modules`` and
load ``training_worker.py`` by file path, then drive ``_forward_actor`` with a fully
mocked engine module.

Asserts the three mutually-exclusive branches:
  * ``bc_loss_coef == 0.0`` (default) → actor_loss == sac_loss, ``bc_loss`` NOT called
    (proves the existing pi05/libero path is byte-for-byte unchanged);
  * ``bc_loss_coef > 0`` (and td3 off) → actor_loss == sac_loss + coef * bc_loss;
  * ``td3_enabled`` → adaptive TD3+BC weight wins and ``bc_loss_coef`` is ignored.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SRC = Path(__file__).resolve().parents[4] / "src"


class _FakeDataProto:
    def __init__(self, batch=None):
        self.batch = dict(batch or {})


def _fresh_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    return mod


def _load_training_worker_module():
    """Inject minimal stubs for every top-level import, load by file path, then
    RESTORE ``sys.modules`` so the fakes never leak into other test modules.

    ``training_worker.py`` binds the stubbed callables into its own module globals
    at exec time (``from verl_vla.utils.data import ...``), so cleaning up the
    sys.modules entries afterward does not affect the loaded worker.
    """
    # verl.single_controller.base.decorator: register / make_nd... used as decorators
    dec = _fresh_module("verl.single_controller.base.decorator")
    dec.register = lambda *a, **k: (lambda f: f)
    dec.make_nd_compute_dataproto_dispatch_fn = lambda *a, **k: None

    dev = _fresh_module("verl.utils.device")
    dev.get_device_id = lambda: "cpu"
    dev.get_device_name = lambda: "cpu"

    wcfg = _fresh_module("verl.workers.config")
    wcfg.TrainingWorkerConfig = object

    eng = _fresh_module("verl.workers.engine_workers")

    class _StubTrainingWorker:
        def __init__(self, *a, **k):
            pass

    eng.TrainingWorker = _StubTrainingWorker

    verl = _fresh_module("verl")
    verl.DataProto = _FakeDataProto

    data = _fresh_module("verl_vla.utils.data")

    def _get_dataproto_from_prefix(mb, prefix):
        if prefix == "t0.action.":
            return _FakeDataProto(batch={"action": torch.zeros(2, 4)})
        return _FakeDataProto(batch={"obs": "sentinel"})

    data.get_dataproto_from_prefix = _get_dataproto_from_prefix
    data.split_nested_dicts_or_tuples = lambda *a, **k: a
    data.valid_mean = lambda x, valids: x.float().mean()

    rp = _fresh_module("verl_vla.utils.replay_pool")
    rp.SACReplayPool = object

    vwcfg = _fresh_module("verl_vla.workers.config")
    vwcfg.ActorConfig = object

    # Pure namespace placeholders needed so the dotted imports resolve.
    placeholders = [
        "verl.single_controller",
        "verl.single_controller.base",
        "verl.utils",
        "verl.workers",
        "verl_vla",
        "verl_vla.utils",
        "verl_vla.workers",
    ]
    fakes = {
        "verl": verl,
        "verl.single_controller.base.decorator": dec,
        "verl.utils.device": dev,
        "verl.workers.config": wcfg,
        "verl.workers.engine_workers": eng,
        "verl_vla.utils.data": data,
        "verl_vla.utils.replay_pool": rp,
        "verl_vla.workers.config": vwcfg,
    }
    for name in placeholders:
        fakes[name] = _fresh_module(name)

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(
            "verl_vla.workers.engine.sac._training_worker_under_test",
            _SRC / "verl_vla" / "workers" / "engine" / "sac" / "training_worker.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_TW_MOD = _load_training_worker_module()
SACTrainingWorker = _TW_MOD.SACTrainingWorker

SAC_LOSS = 2.0
BC_LOSS = 5.0
COEF = 0.05


class _FakeModule:
    def __init__(self):
        self.bc_loss_calls = 0

    def sac_forward_state_features(self, obs, tokenizer):
        return {"sf": torch.zeros(2, 3)}

    def sac_forward_actor(self, sf, task_ids=None, is_first_micro_batch=False):
        return torch.zeros(2, 4), torch.zeros(2), {}

    def sac_forward_critic(self, a, sf, task_ids=None, use_target_network=False, method="min", requires_grad=False):
        return torch.full((2,), 1.0)

    def bc_loss(self, obs, tokenizer, actions, valids):
        self.bc_loss_calls += 1
        return torch.tensor(BC_LOSS)


class _FakeEngine:
    def __init__(self):
        self.module = _FakeModule()


def _make_worker(*, td3_enabled: bool, bc_loss_coef: float) -> SACTrainingWorker:
    w = object.__new__(SACTrainingWorker)
    w.td3_enabled = td3_enabled
    w.td3_bc_alpha = 2.5
    w.bc_loss_coef = bc_loss_coef
    w.tokenizer = None
    w.engine = _FakeEngine()
    # Fix sac_loss so the BC composition is the only variable under test.
    w._calculate_actor_loss = lambda log_probs, q_values, valids: torch.tensor(SAC_LOSS)
    return w


def _micro_batch():
    return _FakeDataProto(
        batch={
            "info.task_ids": torch.zeros(2, dtype=torch.long),
            "info.valids": torch.ones(2),
        }
    )


def test_default_coef_zero_is_pure_sac_no_bc_call():
    """Default bc_loss_coef == 0.0 (and td3 off): actor_loss == sac_loss, bc_loss NOT called."""
    w = _make_worker(td3_enabled=False, bc_loss_coef=0.0)
    actor_loss, _, _, metrics = w._forward_actor(_micro_batch(), is_first_micro_batch=False)
    assert pytest.approx(actor_loss.item(), abs=1e-6) == SAC_LOSS
    assert w.engine.module.bc_loss_calls == 0  # pi05/libero path untouched
    assert "bc_anchor_bc_loss" not in metrics


def test_fixed_coef_adds_bc_term():
    """bc_loss_coef > 0 (td3 off): actor_loss == sac_loss + coef * bc_loss."""
    w = _make_worker(td3_enabled=False, bc_loss_coef=COEF)
    actor_loss, _, _, metrics = w._forward_actor(_micro_batch(), is_first_micro_batch=False)
    assert pytest.approx(actor_loss.item(), abs=1e-5) == SAC_LOSS + COEF * BC_LOSS
    assert w.engine.module.bc_loss_calls == 1
    assert pytest.approx(metrics["bc_loss_coef"], abs=1e-9) == COEF
    assert pytest.approx(metrics["bc_anchor_bc_loss"], abs=1e-5) == BC_LOSS


def test_td3_takes_precedence_over_fixed_coef():
    """td3_enabled wins even if bc_loss_coef > 0 (mutually exclusive): TD3+BC metrics emitted."""
    w = _make_worker(td3_enabled=True, bc_loss_coef=COEF)
    _, _, _, metrics = w._forward_actor(_micro_batch(), is_first_micro_batch=False)
    assert "td3_bc_weight" in metrics
    assert "bc_anchor_bc_loss" not in metrics  # the fixed-coef branch did NOT run
