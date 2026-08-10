# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

# Adapted from LeRobot
# src/lerobot/policies/gaussian_actor/configuration_gaussian_actor.py at
# commit 22bd7a2f489b367d8df42de803b1e8c4ca63a3f9. Imports and compatibility
# hooks are adjusted for verl-vla's pinned LeRobot release.

"""Compatibility backport of LeRobot's native Gaussian actor configuration."""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import MultiAdamConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def is_image_feature(key: str) -> bool:
    return key.startswith(OBS_IMAGE)


@dataclass
class ConcurrencyConfig:
    actor: str = "threads"
    learner: str = "threads"
    multiprocessing_context: str | None = "spawn"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None


@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("gaussian_actor")
@dataclass
class GaussianActorConfig(PreTrainedConfig):
    """LeRobot GaussianActorConfig kept artifact-compatible with upstream."""

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            OBS_IMAGE: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            OBS_STATE: {"min": [0.0, 0.0], "max": [1.0, 1.0]},
            ACTION: {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        }
    )
    storage_device: str = "cpu"
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    shared_encoder: bool = True
    num_discrete_actions: int | None = None
    image_embedding_pooling_dim: int = 8
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 256
    online_steps: int = 1_000_000
    online_buffer_capacity: int = 100_000
    offline_buffer_capacity: int = 100_000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": 3e-4},
                "critic": {"lr": 3e-4},
                "temperature": {"lr": 3e-4},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not (OBS_STATE in self.input_features or any(is_image_feature(key) for key in self.input_features)):
            raise ValueError("Gaussian actor requires observation.state or an image observation")
        if ACTION not in self.output_features:
            raise ValueError("Gaussian actor requires an action output feature")

    def validate_feature_names(self) -> None:
        return

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None


def make_gaussian_actor_pre_post_processors(config, dataset_stats=None):
    """LeRobot 0.4 plugin-discovery entrypoint for the native processors."""
    from .processor import make_gaussian_actor_pre_post_processors as make_processors

    return make_processors(config, dataset_stats=dataset_stats)
