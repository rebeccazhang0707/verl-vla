# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import SimpleNamespace

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from verl import DataProto

from verl_vla.models.builder import build_vla_model
from verl_vla.models.gaussian_actor import GaussianActorConfig, GaussianActorPolicy, GaussianActorTrainableModel
from verl_vla.models.gaussian_actor.configuration import ActorNetworkConfig, PolicyConfig
from verl_vla.models.gaussian_actor.modeling import TanhMultivariateNormalDiag


def _config() -> GaussianActorConfig:
    return GaussianActorConfig(
        device="cpu",
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.images.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        image_encoder_hidden_dim=8,
        image_embedding_pooling_dim=2,
        latent_dim=16,
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[16]),
        policy_kwargs=PolicyConfig(std_min=1e-5, std_max=2.0),
    )


def _observations() -> DataProto:
    return DataProto.from_dict(
        tensors={
            "observation.images.image": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
            "observation.images.wrist_image": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
            "observation.state": torch.randn(2, 8),
        }
    )


def _model() -> GaussianActorTrainableModel:
    policy = GaussianActorPolicy(_config())
    preprocessor, postprocessor = make_pre_post_processors(policy.config, dataset_stats=_stats())
    return GaussianActorTrainableModel(
        policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        adapter_config={"critic": {"enabled": True, "head_num": 2, "hidden_dims": [16]}},
    )


def _stats() -> dict:
    return {
        "observation.images.image": {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
        },
        "observation.images.wrist_image": {
            "mean": [[[0.485]], [[0.456]], [[0.406]]],
            "std": [[[0.229]], [[0.224]], [[0.225]]],
        },
        "observation.state": {"min": [-1.0] * 8, "max": [1.0] * 8},
        "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
    }


def test_gaussian_actor_native_artifact_round_trip(tmp_path) -> None:
    model = _model()
    expected = {name: value.detach().clone() for name, value in model._native_policy_state_dict(None).items()}

    model.export_policy(tmp_path, state_dict=model.state_dict())
    reloaded = GaussianActorPolicy.from_pretrained(tmp_path, strict=True)

    assert reloaded.state_dict().keys() == expected.keys()
    assert all(torch.equal(reloaded.state_dict()[name], value) for name, value in expected.items())

    built = build_vla_model(
        SimpleNamespace(
            native_architecture="gaussian_actor",
            local_path=str(tmp_path),
            override_config={},
            adapter={
                "processor_dataset_root": None,
                "critic": {"enabled": True, "head_num": 2, "hidden_dims": [16]},
            },
        ),
        torch_dtype=torch.float32,
    )
    assert isinstance(built.policy, GaussianActorPolicy)


def test_tanh_gaussian_uses_scale_as_standard_deviation() -> None:
    loc = torch.zeros(2, 3)
    scale = torch.tensor([[0.1, 0.2, 0.4], [0.3, 0.5, 0.7]], requires_grad=True)

    distribution = TanhMultivariateNormalDiag(loc=loc, scale_diag=scale)
    distribution.rsample().sum().backward()

    torch.testing.assert_close(distribution.base_dist.scale_tril, torch.diag_embed(scale.detach()))
    torch.testing.assert_close(distribution.base_dist.variance, scale.detach().square())
    assert scale.grad is not None


def test_gaussian_actor_sac_contract() -> None:
    model = _model()
    actor_encoder_parameter_ids = {id(parameter) for parameter in model.policy.actor.encoder.parameters()}
    critic_parameter_ids = {id(parameter) for parameter in model.sac_get_critic_parameters()}
    features = model.sac_forward_state_features(_observations(), None)
    actions, log_prob, metrics = model.sac_forward_actor(features)
    q_values = model.sac_forward_critic({"action": actions}, features, method="cat", requires_grad=True)

    assert set(features) == {
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
    }
    assert actions.shape == (2, 1, 7)
    assert log_prob.shape == (2,)
    assert metrics.keys() == {"pre_tanh_std_mean"}
    assert metrics["pre_tanh_std_mean"] > 0
    assert q_values.shape == (2, 2)
    assert torch.all(actions.abs() < 1)
    assert actor_encoder_parameter_ids.isdisjoint(critic_parameter_ids)
    assert model.critic_encoder is not model.policy.actor.encoder

    _, actor_parameter = next(iter(model.sac_get_named_actor_parameters()))
    actor_before = actor_parameter.detach().clone()
    (-q_values.mean() + log_prob.mean()).backward()
    assert actor_parameter.grad is not None
    assert torch.equal(actor_parameter, actor_before)

    target_before = next(model.critic.target_heads.parameters()).detach().clone()
    source = next(model.critic.heads.parameters())
    with torch.no_grad():
        source.add_(1.0)
    model.sac_update_target_network(0.5)
    torch.testing.assert_close(next(model.critic.target_heads.parameters()), target_before + 0.5)


def test_gaussian_actor_scales_sac_exploration_around_mean_policy() -> None:
    model = _model()
    model.eval()
    observations = _observations()
    mean_action = model.sac_sample_actions(observations, eval=True).action

    torch.manual_seed(7)
    unit_scale_action = model.sac_sample_actions(observations).action
    model.sac_std_scale = 0.1
    torch.manual_seed(7)
    controlled_action = model.sac_sample_actions(observations).action

    assert torch.linalg.vector_norm(controlled_action - mean_action) < torch.linalg.vector_norm(
        unit_scale_action - mean_action
    )


def test_gaussian_actor_sft_uses_one_step_action_mse_and_freezes_std() -> None:
    model = _model()
    model.sft_init()
    actions = {"action": torch.zeros(2, 1, 7)}

    loss = model.sft_loss(
        _observations(),
        tokenizer=None,
        actions=actions,
        valids=torch.ones(2),
        action_mask=torch.ones(2, 1),
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.policy.actor.mean_layer.weight.grad is not None
    assert model.policy.actor.std_layer.weight.grad is None
    assert not model.policy.actor.std_layer.weight.requires_grad

    model.sac_init()
    assert model.policy.actor.std_layer.weight.requires_grad
