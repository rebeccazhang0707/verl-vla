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

"""Unit tests for the GR00T DSRL Transformer sequence critic."""

from __future__ import annotations

import torch

from verl_vla.models.gr00t_n1d6.critic import Gr00tTransformerCriticGroup

BATCH = 4
HEADS = 3
ACTION_HORIZON = 6
ACTION_DIM = 26
FEATURE_DIM = 64
STATE_HORIZON = 2
STATE_DIM = 32
OBS_DIM = FEATURE_DIM + STATE_HORIZON * STATE_DIM


def _make_critic(**overrides) -> Gr00tTransformerCriticGroup:
    values = {
        "head_num": HEADS,
        "obs_dim": OBS_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "d_model": 32,
        "nhead": 4,
    }
    values.update(overrides)
    return Gr00tTransformerCriticGroup(**values)


def _state_features() -> dict[str, torch.Tensor]:
    return {"pooled": torch.randn(BATCH, FEATURE_DIM), "state": torch.randn(BATCH, STATE_HORIZON, STATE_DIM)}


def _actions(key: str = "action") -> dict[str, torch.Tensor]:
    return {key: torch.randn(BATCH, ACTION_HORIZON, ACTION_DIM)}


def test_forward_shapes_for_cat_and_min():
    critic = _make_critic()
    sf, a = _state_features(), _actions()
    assert critic(a, sf).shape == (BATCH, HEADS)
    assert critic(a, sf, method="min").shape == (BATCH,)


def test_target_network_starts_as_a_copy_and_tracks_via_polyak():
    critic = _make_critic()
    sf, a = _state_features(), _actions()
    target_q = critic(a, sf, use_target_network=True)
    torch.testing.assert_close(critic(a, sf), target_q)

    # Shift only the output bias so the change survives the input LayerNorm.
    for member in critic.critic_members:
        member.value_head.bias.data.add_(1.0)
    torch.testing.assert_close(critic(a, sf), target_q + 1.0)

    critic.update_target_network(0.5)
    torch.testing.assert_close(critic(a, sf, use_target_network=True), target_q + 0.5)

    # Online optimization must not accumulate gradients on the target copy.
    assert all(not p.requires_grad for p in critic.target_members.parameters())
    critic(a, sf, requires_grad=True).sum().backward()
    assert critic.critic_members[0].value_head.weight.grad is not None
    assert all(p.grad is None for p in critic.target_members.parameters())


def test_replay_full_action_key_is_accepted():
    """Replay transitions carry the steering noise under ``full_action``."""
    critic = _make_critic()
    assert critic(_actions("full_action"), _state_features()).shape == (BATCH, HEADS)
