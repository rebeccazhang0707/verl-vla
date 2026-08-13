# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from torch import nn

from verl_vla.models.pi0_torch.critic import PI0CNNCritic, PI0CNNCriticBackend
from verl_vla.models.pi0_torch.trainable_model import PI0TrainableModel


def test_pi0_adapts_state_features_for_dsrl_cnn_critic() -> None:
    model = PI0TrainableModel.__new__(PI0TrainableModel)
    nn.Module.__init__(model)
    model.critic_type = "cnn"
    model.critic_api = PI0CNNCriticBackend()
    model.dsrl = None
    model.critic = PI0CNNCritic(
        head_num=2,
        state_dim=8,
        action_dim=32,
        hidden_dims=[16],
        image_size=64,
        cnn_features=[8, 8, 8, 8],
        cnn_strides=[2, 1, 1, 1],
        latent_dim=12,
    )

    pixels = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    states = torch.randn(2, 8)
    actions = torch.randn(2, 1, 32)
    state_features = ((), states, (pixels, states))

    q_values = model.sac_forward_critic(
        {"action": actions},
        state_features,
        method="cat",
        requires_grad=True,
    )

    assert q_values.shape == (2, 2)
    assert model.sac_get_critic_parameters() == model.critic.trainable_parameters()

    with torch.no_grad():
        model.critic.critic_heads[0].network[-1].bias.fill_(2.0)
    model.sac_update_target_network(0.25)
    target_bias = model.critic.target_network_heads[0].network[-1].bias
    torch.testing.assert_close(target_bias, torch.full_like(target_bias, 0.5))
