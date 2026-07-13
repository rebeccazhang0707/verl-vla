# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Unit tests for GR00T adapter state-dict helpers (no gr00t runtime required)."""

from __future__ import annotations

import torch

from verl_vla.models.gr00t_n1d6.utils import (
    extract_critic_state_dict,
    normalize_adapter_state_dict,
)


def test_normalize_adapter_state_dict_remaps_legacy_critic_prefixes():
    state = {
        "policy.backbone.weight": torch.ones(1),
        "critic_backend.critic_heads.0.weight": torch.ones(2),
        "auxiliary_modules.critic.target_critic_heads.0.weight": torch.ones(3),
        "critic.already_ok": torch.ones(4),
    }
    normalized = normalize_adapter_state_dict(state)
    assert "policy.backbone.weight" in normalized
    assert "critic.critic_heads.0.weight" in normalized
    assert "critic.target_critic_heads.0.weight" in normalized
    assert "critic.already_ok" in normalized
    assert "critic_backend.critic_heads.0.weight" not in normalized
    assert "auxiliary_modules.critic.target_critic_heads.0.weight" not in normalized


def test_extract_critic_state_dict_strips_prefix():
    state = {
        "policy.x": torch.ones(1),
        "critic.heads.0.weight": torch.ones(2),
        "critic.target_heads.0.weight": torch.ones(3),
    }
    critic = extract_critic_state_dict(state)
    assert set(critic) == {"heads.0.weight", "target_heads.0.weight"}
    assert "policy.x" not in critic
