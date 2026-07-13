# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Unit tests for GR00T adapter / critic config (gr00t-package-free)."""

from verl_vla.models.gr00t_n1d6.adapter_config import Gr00tAdapterConfig, Gr00tCriticConfig


def test_critic_defaults_disabled():
    critic = Gr00tCriticConfig()
    assert critic.enabled is False
    assert critic.type == "cross_attn"
    assert critic.pooling == "attn"
    assert critic.head_num == 10


def test_sac_enable_legacy_alias_enables_critic():
    cfg = Gr00tAdapterConfig(sac_enable=True, critic_type="mean_pool", critic_head_num=4)
    assert cfg.critic.enabled is True
    assert cfg.sac_enable is True
    assert cfg.critic.type == "mean_pool"
    assert cfg.critic.head_num == 4
    assert cfg.critic.pooling == "mean"


def test_nested_critic_can_enable_without_legacy_alias():
    cfg = Gr00tAdapterConfig(
        critic={"enabled": True, "type": "cross_attn", "head_num": 8},
    )
    assert cfg.critic.enabled is True
    assert cfg.critic.head_num == 8


def test_legacy_sac_enable_overrides_nested_enabled():
    cfg = Gr00tAdapterConfig(
        sac_enable=False,
        critic={"enabled": True, "type": "cross_attn", "head_num": 8},
    )
    # Flat legacy fields are applied after nested critic (same as pi0).
    assert cfg.critic.enabled is False
    assert cfg.critic.head_num == 8


def test_adapter_to_dict_includes_critic():
    cfg = Gr00tAdapterConfig(policy_type="arena", action_dim=26)
    payload = cfg.to_dict()
    assert payload["policy_type"] == "arena"
    assert payload["action_dim"] == 26
    assert "critic" in payload
    assert payload["critic"]["enabled"] is False


def test_libero_sft_defaults_via_overrides():
    cfg = Gr00tAdapterConfig(
        policy_type="libero",
        embodiment_tag="libero_panda",
        action_dim=7,
        embodiment_id=2,
        num_action_chunks=8,
        override_modality_configs=True,
        use_relative_action=True,
        critic={"enabled": False},
    )
    assert cfg.policy_type == "libero"
    assert cfg.embodiment_tag == "libero_panda"
    assert cfg.action_dim == 7
    assert cfg.override_modality_configs is True
    assert cfg.use_relative_action is True
    assert cfg.critic.enabled is False
    assert "norm_stats_path" in cfg.to_dict()
