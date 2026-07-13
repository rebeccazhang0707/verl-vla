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

"""Unit tests for the gr00t-free SAC config surface (``configuration_gr00t``).

``cfg_get`` treats explicit ``None`` as unset so checkpoint configs without SAC
fields fall through to defaults. Runtime SAC settings now live on
``Gr00tAdapterConfig``; this module remains the gr00t-free helper/defaults table.
"""

import types

import pytest

gr00t_cfg = pytest.importorskip("verl_vla.models.gr00t_n1d6.configuration_gr00t")


def _ns(**kwargs):
    """A bare attribute holder standing in for an HF PretrainedConfig."""
    return types.SimpleNamespace(**kwargs)


def test_cfg_get_missing_attribute_returns_default():
    cfg = _ns()
    assert gr00t_cfg.cfg_get(cfg, "critic_head_num", 10) == 10


def test_cfg_get_explicit_none_treated_as_unset():
    # A gr00t config that carries the key but leaves it None must fall through to
    # the default (this is the crux of the override-merge contract).
    cfg = _ns(critic_action_dim=None)
    assert gr00t_cfg.cfg_get(cfg, "critic_action_dim", 26) == 26


def test_cfg_get_present_value_wins_over_default():
    cfg = _ns(action_dim=7)
    assert gr00t_cfg.cfg_get(cfg, "action_dim", 26) == 7


def test_cfg_get_falsy_but_not_none_is_preserved():
    # 0 / False / "" are legitimate configured values, NOT "unset"; they must be
    # returned verbatim rather than replaced by the default.
    assert gr00t_cfg.cfg_get(_ns(critic_pool_proj_dim=0), "critic_pool_proj_dim", 128) == 0
    assert gr00t_cfg.cfg_get(_ns(flow_sde_enable=False), "flow_sde_enable", True) is False
    assert gr00t_cfg.cfg_get(_ns(policy_type=""), "policy_type", "arena") == ""


def test_cfg_get_default_none_when_unset():
    assert gr00t_cfg.cfg_get(_ns(), "sac_action_train_dims") is None


def test_sac_config_defaults_populated_from_table():
    config = gr00t_cfg.Gr00tSACConfig()
    for key, default in gr00t_cfg.GR00T_SAC_CONFIG_DEFAULTS.items():
        assert getattr(config, key) == default, f"{key} default mismatch"


def test_sac_config_kwargs_override_defaults():
    config = gr00t_cfg.Gr00tSACConfig(action_dim=7, critic_type="mean_pool")
    assert config.action_dim == 7
    assert config.critic_type == "mean_pool"
    # Untouched keys keep their table default.
    assert config.embodiment_tag == gr00t_cfg.GR00T_SAC_CONFIG_DEFAULTS["embodiment_tag"]


def test_sac_config_defaults_match_gr1_contract():
    # The default embodiment is GR1 (26-DOF, projector index 20); the critic action
    # width defaults to that same real action width.
    config = gr00t_cfg.Gr00tSACConfig()
    assert config.action_dim == 26
    assert config.embodiment_id == 20
    assert config.policy_type == "arena"
    assert config.critic_type == "cross_attn"
    # critic_action_horizon stays None in the table so modeling resolves it to
    # num_action_chunks (executed chunk), not the full checkpoint action_horizon.
    assert config.critic_action_horizon is None
    assert config.num_action_chunks == 16
