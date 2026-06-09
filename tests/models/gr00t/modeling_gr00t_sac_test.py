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

"""CPU-only mock shape tests for ``Gr00tN1d6ForSAC`` (``modeling_gr00t_sac.py``).

Runs on a **bare CPU env with only torch + numpy** — gr00t / transformers / verl
are all absent here. We therefore:

  * stub the heavy top-level imports of ``modeling_gr00t_sac`` in ``sys.modules``
    (``gr00t.* → Gr00tN1d6 = object``, ``transformers.feature_extraction_utils``,
    ``verl_vla.models.base``), and load the **real** ``utils.py`` / ``data.py`` by
    file path so the SAC dims, ``CriticMLP`` and ``split_nested_dicts_or_tuples``
    are exercised for real;
  * build the model with ``object.__new__`` to bypass ``Gr00tN1d6.__init__`` (which
    needs a real checkpoint), hand-injecting scalar dims + fake ``action_head`` /
    ``backbone`` sub-modules (small ``nn.Module`` shims returning the right shapes),
    while the critic heads are the **real** ``CriticMLP`` (deep-copied to targets).

Coverage (Phase 2 contract):
  - ``_run_flow`` / ``_denoise`` → ``(B, H, Dmax)``; flow-SDE log-prob → ``(B,)``;
  - two-pass action mask: frozen dims follow the noise-free **base** trajectory;
  - ``_critic_input`` width == ``critic_input_dim``; critic ``cat→(B,heads)`` /
    ``min→(B,)``; ``task_ids`` accepted-and-ignored;
  - ``sac_update_target_network`` mutates the target heads;
  - ``bc_loss`` reads the single ``action`` key (KeyError on the old ``full_action``);
  - **None-free contract**: a ``None`` backbone ``image_mask`` is dropped from the
    state-features dict and ``split_nested_dicts_or_tuples(sf, 2)`` does not raise;
  - new rollout entry points ``sac_sample_actions`` / ``sac_get_critic_value``.
"""

import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

_SRC = Path(__file__).resolve().parents[3] / "src"


# --------------------------------------------------------------------------- #
# 1. Stub the heavy top-level deps + load the real leaf modules by file path.
# --------------------------------------------------------------------------- #
def _ensure_pkg(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _load_by_path(mod_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_modeling_module():
    """Inject stubs + load ``modeling_gr00t_sac.py`` standalone (no gr00t/verl)."""

    # --- fake transformers.feature_extraction_utils.BatchFeature ---
    class _FakeBatchFeature:
        def __init__(self, data=None, **kw):
            self.data = dict(data or {})

        def __getitem__(self, key):
            return self.data[key]

        def get(self, key, default=None):
            return self.data.get(key, default)

    tf = _ensure_pkg("transformers")
    tf_feat = _ensure_pkg("transformers.feature_extraction_utils")
    tf_feat.BatchFeature = _FakeBatchFeature
    tf.feature_extraction_utils = tf_feat

    # --- fake gr00t.model.gr00t_n1d6.gr00t_n1d6.Gr00tN1d6 = object ---
    _ensure_pkg("gr00t")
    _ensure_pkg("gr00t.model")
    _ensure_pkg("gr00t.model.gr00t_n1d6")
    g_leaf = _ensure_pkg("gr00t.model.gr00t_n1d6.gr00t_n1d6")
    # A fresh empty class (NOT ``object``): the SAC class also mixes in the
    # base-class shims below, and using ``object`` as a base alongside its own
    # subclasses would make the MRO inconsistent.
    g_leaf.Gr00tN1d6 = type("Gr00tN1d6", (), {})

    # --- parent package stubs so absolute imports never run real __init__ ---
    _ensure_pkg("verl_vla")
    _ensure_pkg("verl_vla.models")
    _ensure_pkg("verl_vla.models.gr00t")
    _ensure_pkg("verl_vla.utils")

    # --- fake verl_vla.models.base (plain mixin bases) ---
    base = _ensure_pkg("verl_vla.models.base")

    class SupportSACTraining:  # noqa: D401 - minimal stand-in
        pass

    class SupportSFTTraining:
        pass

    class ModelOutput:
        pass

    base.SupportSACTraining = SupportSACTraining
    base.SupportSFTTraining = SupportSFTTraining
    base.ModelOutput = ModelOutput

    # --- real utils.py (numpy-only) injected at its package path ---
    _load_by_path("verl_vla.models.gr00t.utils", _SRC / "verl_vla" / "models" / "gr00t" / "utils.py")

    # --- real split helper from data.py: stub verl + keys, then load by path ---
    verl_stub = _ensure_pkg("verl")
    verl_stub.DataProto = object
    keys = _ensure_pkg("verl_vla.utils.keys")
    keys.OBS_KEY = "obs"
    keys.ACTION_KEY = "action"
    keys.FEEDBACK_KEY = "feedback"
    keys.INTERVENTION_INFO_KEY = "intervention_info"
    data_mod = _load_by_path("verl_vla.utils.data", _SRC / "verl_vla" / "utils" / "data.py")

    # --- finally, the module under test ---
    modeling = _load_by_path(
        "verl_vla.models.gr00t.modeling_gr00t_sac",
        _SRC / "verl_vla" / "models" / "gr00t" / "modeling_gr00t_sac.py",
    )
    return modeling, data_mod


_MODELING, _DATA = _load_modeling_module()
Gr00tN1d6ForSAC = _MODELING.Gr00tN1d6ForSAC
CriticMLP = _MODELING.CriticMLP
split_nested_dicts_or_tuples = _DATA.split_nested_dicts_or_tuples


# --------------------------------------------------------------------------- #
# 2. Fake GR00T sub-modules (small nn.Modules with the right output shapes).
# --------------------------------------------------------------------------- #
# tiny dims so the test is fast
B = 4
H = 4            # action_horizon
DMAX = 8         # max_action_dim
STATE_DIM = 6    # max_state_dim
T = 1            # state_horizon
BB_DIM = 5       # backbone_feature_dim
E = 7            # action-head internal embed dim
S = 4            # backbone seq len
NUM_STEPS = 2    # num_inference_timesteps
NUM_HEADS = 3    # critic heads
ACTION_DIM = 6   # real (unpadded) action width


class _FakeBackbone(nn.Module):
    def __init__(self, return_image_mask: bool = True):
        super().__init__()
        self.dummy = nn.Linear(1, 1)  # gives parameters() a device/dtype anchor
        self.register_buffer("feat", torch.randn(S, BB_DIM))
        self.return_image_mask = return_image_mask

    def forward(self, inputs):
        b = inputs["input_ids"].shape[0]
        device = self.dummy.weight.device
        dtype = self.dummy.weight.dtype
        feats = self.feat.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1).contiguous()
        attn = torch.ones(b, S, dtype=torch.bool, device=device)
        out = {"backbone_features": feats, "backbone_attention_mask": attn}
        # Deliberately include the key with a None value when the backbone yields
        # no image_mask (matches the real backbone) — exercises the None-free path.
        out["image_mask"] = torch.ones(b, S, dtype=torch.bool, device=device) if self.return_image_mask else None
        return out


class _FakeStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(STATE_DIM, E)

    def forward(self, state, emb_id):
        return self.proj(state)  # (B, T, E)


class _FakeActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DMAX, E)

    def forward(self, x, timesteps, emb_id):
        return self.proj(x)  # (B, H, E) — depends on x so the actor grad flows


class _FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(E, E)

    def forward(self, hidden_states, encoder_hidden_states, timestep, image_mask=None, backbone_attention_mask=None):
        return self.proj(hidden_states)  # (B, T+H, E)


class _FakeActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(E, DMAX)

    def forward(self, model_output, emb_id):
        return self.proj(model_output)  # (B, T+H, DMAX); caller slices last H


class _FakeActionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlln = nn.Identity()
        self.state_encoder = _FakeStateEncoder()
        self.action_encoder = _FakeActionEncoder()
        self.position_embedding = nn.Embedding(H, E)
        self.model = _FakeDiT()
        self.action_decoder = _FakeActionDecoder()


class _FakeObs:
    """Stand-in for a ``DataProto`` — only ``.batch`` (a dict) is read."""

    def __init__(self, batch):
        self.batch = batch


def _state_dict(batch=B):
    return {
        "images": torch.randn(batch, 1, 3, 8, 8),  # (B, n_patches, C, H, W)
        "lang_tokens": torch.zeros(batch, 5, dtype=torch.long),
        "lang_masks": torch.ones(batch, 5, dtype=torch.bool),
        "states": torch.randn(batch, T, STATE_DIM),
    }


def _make_model(
    *,
    train_dims=None,
    flow_sde=False,
    return_image_mask=True,
    build_heads=True,
    critic_pooling="mean",
    critic_use_encoded_state=False,
    critic_prefix_attn_heads=1,
):
    """Construct a ``Gr00tN1d6ForSAC`` via ``object.__new__`` (bypass real init)."""
    torch.manual_seed(1234)  # deterministic fake weights
    m = object.__new__(Gr00tN1d6ForSAC)

    m.action_horizon = H
    m.max_action_dim = DMAX
    m.max_state_dim = STATE_DIM
    m.state_horizon = T
    m.backbone_feature_dim = BB_DIM
    m.num_inference_timesteps = NUM_STEPS
    m.num_timestep_buckets = 100
    m.add_pos_embed = True
    m.use_alternate_vl_dit = True
    m.action_dim = ACTION_DIM
    m.embodiment_id = 20
    m.num_critic_heads = NUM_HEADS
    m.critic_action_dim = ACTION_DIM
    m.critic_action_horizon = H

    # --- critic representation options (legacy defaults: mean-pool + raw state) ---
    m.critic_pooling = critic_pooling
    m.critic_use_encoded_state = critic_use_encoded_state
    m.critic_prefix_attn_heads = critic_prefix_attn_heads
    m._state_feature_dim = E  # state_encoder output width (fake action-head embed dim)
    # Asymmetric-AC privileged critic obs: OFF for these shape tests (matches production
    # default), so critic_input_dim is unchanged and the priv_obs code paths are no-ops.
    m.critic_privileged_obs = False
    m.critic_privileged_obs_dim = 0
    critic_state_width = E if critic_use_encoded_state else STATE_DIM
    m.critic_input_dim = BB_DIM + T * critic_state_width + H * ACTION_DIM + m.critic_privileged_obs_dim

    # flow-SDE config + step buffer
    m.flow_sde_enable = flow_sde
    m.flow_sde_noise_level = 0.065
    m.flow_sde_rollout_noise_scale = 1.0
    m.flow_sde_train_noise_scale = 1.0
    m.flow_sde_initial_beta = 1.0
    m.flow_sde_beta_min = 0.02
    m.flow_sde_beta_schedule_T = 4000
    m.flow_sde_step = torch.zeros((), dtype=torch.long)

    # action train mask
    mask = torch.zeros(DMAX, dtype=torch.bool)
    if train_dims is None:
        mask[:] = True
    else:
        for s, e in train_dims:
            mask[s:e] = True
    m.sac_action_train_mask = mask
    m.sac_action_train_all = bool(mask.all().item())

    m.backbone = _FakeBackbone(return_image_mask=return_image_mask).eval()
    m.action_head = _FakeActionHead().eval()

    m.critic_heads = None
    m.target_critic_heads = None
    m.critic_state_token = None
    m.target_state_token = None
    m.critic_prefix_cross_attn = None
    m.target_prefix_cross_attn = None
    if build_heads:
        m.critic_heads = nn.ModuleList([CriticMLP(m.critic_input_dim) for _ in range(NUM_HEADS)]).eval()
        m.target_critic_heads = copy.deepcopy(m.critic_heads)
        for p in m.target_critic_heads.parameters():
            p.requires_grad_(False)
        if critic_pooling == "attn":
            d = BB_DIM
            # Mirror production: the cross-attn query token is an nn.Embedding (NOT a bare
            # nn.Parameter), so from_pretrained's _fast_init initializes it instead of leaving NaN.
            m.critic_state_token = nn.Embedding(1, d)
            m.target_state_token = nn.Embedding(1, d)
            nn.init.normal_(m.critic_state_token.weight, mean=0.0, std=0.02)
            m.target_state_token.load_state_dict(m.critic_state_token.state_dict())
            for p in m.target_state_token.parameters():
                p.requires_grad_(False)
            m.critic_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=critic_prefix_attn_heads, batch_first=True
            )
            m.target_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=critic_prefix_attn_heads, batch_first=True
            )
            m.target_prefix_cross_attn.load_state_dict(m.critic_prefix_cross_attn.state_dict())
            for p in m.target_prefix_cross_attn.parameters():
                p.requires_grad_(False)
    return m


# --------------------------------------------------------------------------- #
# 3. Tests
# --------------------------------------------------------------------------- #
def test_run_flow_and_denoise_shapes():
    m = _make_model(train_dims=None, flow_sde=False)
    sf = m._state_features_impl(_state_dict())

    x, lp = m._run_flow(
        sf, torch.randn(B, H, DMAX), noise_scale=0.0, requires_grad=False, return_log_prob=False
    )
    assert x.shape == (B, H, DMAX)
    assert lp is None  # deterministic ODE → no log-prob

    x2, lp2 = m._denoise(sf, noise_scale=0.0, requires_grad=False, return_log_prob=False)
    assert x2.shape == (B, H, DMAX)
    assert lp2 is None


def test_flow_sde_logprob_shape():
    m = _make_model(train_dims=None, flow_sde=True)
    sf = m._state_features_impl(_state_dict())
    x, lp = m._denoise(sf, noise_scale=1.0, requires_grad=False, return_log_prob=True)
    assert x.shape == (B, H, DMAX)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_two_pass_mask_frozen_dims_follow_base_trajectory():
    # right_arm-like subset trainable: dims [1,4); the rest are frozen.
    train_dims = [[1, 4]]
    m = _make_model(train_dims=train_dims, flow_sde=True)
    assert not m.sac_action_train_all
    sf = m._state_features_impl(_state_dict())

    torch.manual_seed(0)
    x, _ = m._denoise(sf, noise_scale=1.0, requires_grad=False, return_log_prob=False)

    # Reproduce the internal noise-free *base* pass from the same initial noise.
    torch.manual_seed(0)
    x0 = torch.randn((B, H, DMAX))
    x_base, _ = m._run_flow(sf, x0, noise_scale=0.0, requires_grad=False, return_log_prob=False)

    frozen = ~m.sac_action_train_mask
    trainable = m.sac_action_train_mask
    assert torch.allclose(x[..., frozen], x_base[..., frozen], atol=1e-6), "frozen dims must equal the base trajectory"
    # the explored (trainable) dims carry the injected noise → must differ
    assert not torch.allclose(x[..., trainable], x_base[..., trainable])


def test_critic_input_width_and_forward_shapes():
    m = _make_model(train_dims=None)
    sf = m._state_features_impl(_state_dict())
    a = {"action": torch.randn(B, H, DMAX)}

    ci = m._critic_input(a, sf)
    assert ci.shape == (B, m.critic_input_dim)

    q_cat = m.sac_forward_critic(a, sf, method="cat")
    assert q_cat.shape == (B, NUM_HEADS)
    q_min = m.sac_forward_critic(a, sf, method="min")
    assert q_min.shape == (B,)

    # task_ids accepted but ignored — same result.
    q_cat_tid = m.sac_forward_critic(a, sf, task_ids=torch.zeros(B, dtype=torch.long), method="cat")
    assert q_cat_tid.shape == (B, NUM_HEADS)


def test_critic_input_reads_action_key_not_full_action():
    m = _make_model(train_dims=None)
    sf = m._state_features_impl(_state_dict())
    with pytest.raises(KeyError):
        m._critic_input({"full_action": torch.randn(B, H, DMAX)}, sf)


def test_sac_update_target_network_mutates_targets():
    m = _make_model(train_dims=None)
    with torch.no_grad():
        for p in m.critic_heads.parameters():
            p.add_(1.0)
    before = [p.clone() for p in m.target_critic_heads.parameters()]
    m.sac_update_target_network(0.5)
    after = list(m.target_critic_heads.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))

    # tau == 1.0 copies the online weights exactly.
    m.sac_update_target_network(1.0)
    for po, pt in zip(m.critic_heads.parameters(), m.target_critic_heads.parameters(), strict=True):
        assert torch.allclose(po, pt)


def test_bc_loss_reads_action_key():
    m = _make_model(train_dims=None)
    obs = _FakeObs(_state_dict())
    valids = torch.ones(B)

    loss = m.bc_loss(obs, None, {"action": torch.randn(B, H, DMAX)}, valids)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # the legacy "full_action" key must no longer be accepted.
    with pytest.raises(KeyError):
        m.bc_loss(obs, None, {"full_action": torch.randn(B, H, DMAX)}, valids)


def test_none_free_state_features_contract():
    m = _make_model(train_dims=None, return_image_mask=False)
    sf = m._state_features_impl(_state_dict())

    assert "image_mask" not in sf, "None image_mask must be dropped (None-free contract)"
    assert all(isinstance(v, torch.Tensor) for v in sf.values()), "state_features must be None-free"

    # The worker runs this on the returned dict; it raises TypeError on any None.
    parts = split_nested_dicts_or_tuples(sf, 2)
    assert len(parts) == 2
    for part in parts:
        assert "image_mask" not in part
        assert part["pooled"].shape[0] == B // 2


def test_state_features_with_image_mask_is_still_none_free():
    m = _make_model(train_dims=None, return_image_mask=True)
    sf = m._state_features_impl(_state_dict())
    assert "image_mask" in sf
    # must not raise even with the optional key present
    split_nested_dicts_or_tuples(sf, 2)


def test_sac_forward_state_features_unpacks_obs_and_ignores_tokenizer():
    m = _make_model(train_dims=None)
    obs = _FakeObs(_state_dict())
    sf = m.sac_forward_state_features(obs, tokenizer="ignored-on-purpose")
    assert sf["pooled"].shape == (B, BB_DIM)
    assert sf["state"].shape == (B, T, STATE_DIM)
    assert sf["embodiment_id"].shape == (B,)


def test_sac_sample_actions_returns_action_and_logprobs_dict():
    m = _make_model(train_dims=None, flow_sde=False)
    obs = _FakeObs(_state_dict())
    out = m.sac_sample_actions(obs)
    assert set(out.keys()) == {"action", "log_probs"}
    assert out["action"].shape == (B, H, DMAX)
    assert out["log_probs"].shape == (B,)
    # flow-SDE off → zero log-probs placeholder
    assert torch.allclose(out["log_probs"], torch.zeros(B))


def test_sac_get_critic_value_shape_and_input_forms():
    m = _make_model(train_dims=None, flow_sde=False)
    obs = _FakeObs(_state_dict())
    out = m.sac_sample_actions(obs)

    q = m.sac_get_critic_value(obs, out)  # dict form
    assert q.shape == (B,)
    assert q.dtype == torch.float32

    class _ActionObj:
        action = out["action"]

    q2 = m.sac_get_critic_value(obs, _ActionObj())  # attribute form
    assert q2.shape == (B,)


def test_sac_forward_actor_grad_enabled_and_task_ids_ignored():
    m = _make_model(train_dims=None, flow_sde=True)
    sf = m._state_features_impl(_state_dict())
    actions, log_probs, metrics = m.sac_forward_actor(
        sf, task_ids=torch.zeros(B, dtype=torch.long), is_first_micro_batch=True
    )
    assert actions.shape == (B, H, DMAX)
    assert actions.requires_grad, "actor sampling must be differentiable"
    assert log_probs.shape == (B,)
    assert isinstance(metrics, dict)
    # is_first_micro_batch advanced the beta schedule step.
    assert int(m.flow_sde_step.item()) == 1


# --------------------------------------------------------------------------- #
# 4. Source-commit 09b0f07 sync: vision freeze + critic cross-attn pooling +
#    encoded-state critic input (all config-gated; default = legacy regression).
# --------------------------------------------------------------------------- #
def test_default_path_is_legacy_mean_pool_no_attn_params():
    """Default critic_pooling != 'attn' ⇒ no cross-attn submodules, mean-pool input."""
    m = _make_model(train_dims=None)
    assert m.critic_pooling == "mean"
    assert m.critic_prefix_cross_attn is None
    assert m.critic_state_token is None
    # critic params are exactly the head params — no attn params leak in.
    assert len(m.sac_get_critic_parameters()) == len(list(m.critic_heads.parameters()))

    sf = m._state_features_impl(_state_dict())
    a = {"action": torch.randn(B, H, DMAX)}
    ci = m._critic_input(a, sf)
    # mean-pool input width == backbone + raw-state + action
    assert ci.shape == (B, BB_DIM + T * STATE_DIM + H * ACTION_DIM)


def test_attn_pool_output_shape_and_targets_initialised():
    m = _make_model(train_dims=None, critic_pooling="attn")
    assert m.critic_prefix_cross_attn is not None
    assert m.target_prefix_cross_attn is not None
    sf = m._state_features_impl(_state_dict())

    pooled = m._cross_attention_pool(sf["backbone_features"], sf["backbone_attention_mask"], False)
    assert pooled.shape == (B, BB_DIM)

    # online == target at init (deepcopy-equivalent load_state_dict + token copy).
    for po, pt in zip(
        m.critic_prefix_cross_attn.parameters(), m.target_prefix_cross_attn.parameters(), strict=True
    ):
        assert torch.allclose(po, pt)
    assert torch.allclose(m.critic_state_token.weight, m.target_state_token.weight)

    # full critic input still has the unchanged width (attn output dim == backbone dim).
    a = {"action": torch.randn(B, H, DMAX)}
    ci = m._critic_input(a, sf)
    assert ci.shape == (B, BB_DIM + T * STATE_DIM + H * ACTION_DIM)


def test_attn_critic_params_include_cross_attn_and_state_token():
    m = _make_model(train_dims=None, critic_pooling="attn")
    params = m.sac_get_critic_parameters()
    n_heads = len(list(m.critic_heads.parameters()))
    n_attn = len(list(m.critic_prefix_cross_attn.parameters()))
    # heads + cross-attn params + the single state_token (nn.Embedding -> one .weight param)
    assert len(params) == n_heads + n_attn + 1
    assert any(p is m.critic_state_token.weight for p in params)


def test_attn_target_polyak_updates_attn_and_state_token():
    m = _make_model(train_dims=None, critic_pooling="attn")
    with torch.no_grad():
        for p in m.critic_prefix_cross_attn.parameters():
            p.add_(1.0)
        m.critic_state_token.weight.add_(1.0)
    before_attn = [p.clone() for p in m.target_prefix_cross_attn.parameters()]
    before_tok = m.target_state_token.weight.clone()

    m.sac_update_target_network(0.5)

    after_attn = list(m.target_prefix_cross_attn.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before_attn, after_attn, strict=True))
    assert not torch.allclose(before_tok, m.target_state_token.weight)

    # tau == 1.0 hard-copies online → target for both attn params and the token.
    m.sac_update_target_network(1.0)
    for po, pt in zip(
        m.critic_prefix_cross_attn.parameters(), m.target_prefix_cross_attn.parameters(), strict=True
    ):
        assert torch.allclose(po, pt)
    assert torch.allclose(m.critic_state_token.weight, m.target_state_token.weight)


def test_attn_forward_critic_uses_target_pool_and_shapes():
    m = _make_model(train_dims=None, critic_pooling="attn")
    sf = m._state_features_impl(_state_dict())
    a = {"action": torch.randn(B, H, DMAX)}
    q_cat = m.sac_forward_critic(a, sf, method="cat")
    assert q_cat.shape == (B, NUM_HEADS)
    q_min_t = m.sac_forward_critic(a, sf, method="min", use_target_network=True)
    assert q_min_t.shape == (B,)
    assert torch.isfinite(q_cat).all() and torch.isfinite(q_min_t).all()


def test_critic_use_encoded_state_changes_input_width():
    m = _make_model(train_dims=None, critic_use_encoded_state=True)
    sf = m._state_features_impl(_state_dict())
    a = {"action": torch.randn(B, H, DMAX)}
    ci = m._critic_input(a, sf)
    # encoded-state width E replaces the raw max_state_dim
    assert ci.shape == (B, BB_DIM + T * E + H * ACTION_DIM)
    assert ci.shape[1] == m.critic_input_dim
    q = m.sac_forward_critic(a, sf, method="cat")
    assert q.shape == (B, NUM_HEADS)


def test_init_weights_multihead_attention_no_nan():
    m = _make_model(train_dims=None, critic_pooling="attn")
    attn = m.critic_prefix_cross_attn
    # Zero out the packed in-proj params to simulate uninitialised meta-device memory,
    # then run _init_weights and confirm it repopulates them finitely.
    with torch.no_grad():
        attn.in_proj_weight.zero_()
        if attn.in_proj_bias is not None:
            attn.in_proj_bias.fill_(float("nan"))
    m._init_weights(attn)
    assert torch.isfinite(attn.in_proj_weight).all()
    assert attn.in_proj_weight.abs().sum() > 0  # xavier_uniform_ wrote something
    if attn.in_proj_bias is not None:
        assert torch.isfinite(attn.in_proj_bias).all()


def test_freeze_vision_tower_sets_requires_grad_false():
    m = _make_model(train_dims=None)

    class _VisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)

    class _Eagle(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = _VisionModel()
            self.mlp1 = nn.Linear(3, 3)

    eagle = _Eagle()
    m.backbone.eagle_model = eagle
    assert all(p.requires_grad for p in eagle.vision_model.parameters())

    m.freeze_vision_tower()

    assert all(not p.requires_grad for p in eagle.vision_model.parameters())
    assert all(not p.requires_grad for p in eagle.mlp1.parameters())
    assert not eagle.vision_model.training  # eval() mode
    assert not eagle.mlp1.training


def test_freeze_vision_tower_noop_when_absent():
    # Default fake backbone has no eagle_model → freeze is a defensive no-op (no raise).
    m = _make_model(train_dims=None)
    assert not hasattr(m.backbone, "eagle_model")
    m.freeze_vision_tower()  # must not raise


def test_cross_attention_pool_nan_guard_on_padding():
    """Padded VL positions carrying NaN/inf must NOT poison the cross-attn pool.

    nn.MultiheadAttention projects the *value* of every key position; key_padding_mask only
    zeros the softmax weight, so without zeroing the padded values first ``0 * NaN = NaN``
    propagates into the pooled output. Regression: with the guard the output is finite; we
    also assert the valid (mask=True) positions fully drive the result (zeroing the masked
    positions ourselves yields the identical pooled vector).
    """
    m = _make_model(train_dims=None, critic_pooling="attn", critic_prefix_attn_heads=1)

    torch.manual_seed(7)
    vl = torch.randn(B, S, BB_DIM)
    # Mark the LAST token of every row as padding (invalid) and poison it with NaN/inf.
    attn_mask = torch.ones(B, S, dtype=torch.bool)
    attn_mask[:, -1] = False
    vl[:, -1, 0] = float("nan")
    vl[:, -1, 1] = float("inf")

    pooled = m._cross_attention_pool(vl, attn_mask, use_target_network=False)
    assert pooled.shape == (B, BB_DIM)
    assert torch.isfinite(pooled).all(), "NaN guard failed: padded NaN/inf leaked into pool"

    # Valid positions alone must determine the output: pre-zeroing the masked positions
    # (no NaN present) must give the SAME pooled vector the guard produces internally.
    vl_clean = vl.clone()
    vl_clean[:, -1, :] = 0.0
    pooled_clean = m._cross_attention_pool(vl_clean, attn_mask, use_target_network=False)
    assert torch.allclose(pooled, pooled_clean, atol=1e-6)


def test_attn_critic_input_finite_with_padded_nan():
    """End-to-end through sac_forward_critic: padded NaN VL tokens → finite Q (attn path)."""
    m = _make_model(train_dims=None, critic_pooling="attn", critic_prefix_attn_heads=1)
    sf = m._state_features_impl(_state_dict())
    # Inject NaN/inf at a padding position and mark it invalid in the attention mask.
    sf["backbone_features"][:, -1, 0] = float("nan")
    sf["backbone_features"][:, -1, 1] = float("inf")
    sf["backbone_attention_mask"][:, -1] = False

    a = {"action": torch.randn(B, H, DMAX)}
    ci = m._critic_input(a, sf)
    assert torch.isfinite(ci).all()
    q = m.sac_forward_critic(a, sf, method="cat")
    assert q.shape == (B, NUM_HEADS)
    assert torch.isfinite(q).all()
