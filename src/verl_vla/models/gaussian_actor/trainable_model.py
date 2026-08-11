# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""verl-vla SAC adapter around LeRobot's native GaussianActorPolicy."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from torch import Tensor, nn
from torch.distributed.fsdp import register_fsdp_forward_method
from verl import DataProto

from ..base import ModelOutput, SupportSACTraining, SupportSFTTraining, TrainableVLAModelBase
from .modeling import MLP, GaussianActorPolicy, orthogonal_init
from .policy import GaussianActorOutput


class GaussianCriticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.net = MLP(input_dim=input_dim, hidden_dims=hidden_dims, activate_final=True)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        orthogonal_init()(self.output_layer.weight)

    def forward(self, value: Tensor) -> Tensor:
        return self.output_layer(self.net(value))


class GaussianCriticEnsemble(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], head_num: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([GaussianCriticHead(input_dim, hidden_dims) for _ in range(head_num)])
        self.target_heads = deepcopy(self.heads).requires_grad_(False)

    def forward(self, inputs: Tensor, *, target: bool, method: Literal["cat", "min"]) -> Tensor:
        heads = self.target_heads if target else self.heads
        values = torch.cat([head(inputs) for head in heads], dim=-1)
        if method == "cat":
            return values
        if method == "min":
            return values.min(dim=-1).values
        raise ValueError(f"Unsupported critic reduction: {method}")

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        for target, source in zip(self.target_heads.parameters(), self.heads.parameters(), strict=True):
            target.lerp_(source, tau)


class GaussianActorTrainableModel(TrainableVLAModelBase, SupportSACTraining, SupportSFTTraining):
    def __init__(
        self,
        policy: GaussianActorPolicy,
        *,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        adapter_config: dict | None = None,
    ) -> None:
        super().__init__(policy=policy)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        config = dict(adapter_config or {})
        config.pop("processor_dataset_root", None)
        self.sac_std_scale = float(config.pop("sac_std_scale", 1.0))
        if self.sac_std_scale <= 0.0:
            raise ValueError(f"sac_std_scale must be positive, got {self.sac_std_scale}")
        critic_config = dict(config.pop("critic", {}))
        if config:
            raise ValueError(f"Unsupported Gaussian actor adapter fields: {sorted(config)}")
        critic_enabled = bool(critic_config.pop("enabled", True))
        hidden_dims = [int(dim) for dim in critic_config.pop("hidden_dims", [256, 256])]
        head_num = int(critic_config.pop("head_num", 2))
        if critic_config:
            raise ValueError(f"Unsupported Gaussian critic fields: {sorted(critic_config)}")
        action_dim = int(policy.config.output_features[ACTION].shape[0])
        # The upstream policy may share its actor and critic encoders. SAC critic
        # updates must not mutate the pretrained actor's observation features, so
        # the verl-vla critic owns an independent encoder alongside its Q heads.
        self.critic_encoder = deepcopy(policy.encoder_critic) if critic_enabled else None
        self.critic = (
            GaussianCriticEnsemble(self.critic_encoder.output_dim + action_dim, hidden_dims, head_num)
            if critic_enabled
            else None
        )
        self._canonicalize_encoder_registration()
        SupportSFTTraining.__init__(self, adapter_config or {})
        self.config = policy.config

    def _canonicalize_encoder_registration(self) -> None:
        """Keep one registered path per shared encoder for FSDP2 DTensor state loading."""
        actor_encoder = self.policy.actor.encoder
        if self.policy._modules.get("encoder_actor") is actor_encoder:
            self.policy._modules.pop("encoder_actor")
            object.__setattr__(self.policy, "encoder_actor", actor_encoder)
        if self.policy._modules.get("encoder_critic") is actor_encoder:
            self.policy._modules.pop("encoder_critic")
            object.__setattr__(self.policy, "encoder_critic", actor_encoder)

    def _native_policy_state_dict(self, state_dict) -> dict[str, Tensor]:
        canonical = self.policy.state_dict() if state_dict is None else self.extract_policy_state_dict(state_dict)
        native_template = GaussianActorPolicy(deepcopy(self.policy.config)).state_dict()
        native = {}
        for name in native_template:
            source_name = name
            if name.startswith("encoder_actor."):
                source_name = name.replace("encoder_actor.", "actor.encoder.", 1)
            elif self.policy.shared_encoder and name.startswith("encoder_critic."):
                source_name = name.replace("encoder_critic.", "actor.encoder.", 1)
            native[name] = canonical[source_name]
        return native

    @staticmethod
    def _normalize_image(image: Tensor) -> Tensor:
        return image.float().div(255.0) if image.dtype == torch.uint8 else image.float()

    def _policy_batch(self, obs: DataProto) -> dict[str, Tensor]:
        batch = {}
        for key in self.policy.config.input_features:
            value = obs.batch[key]
            if key in self.policy.config.image_features:
                if value.ndim == 5:
                    value = value[:, -1]
                if value.shape[-1] == 3:
                    value = value.permute(0, 3, 1, 2)
                value = self._normalize_image(value)
            else:
                value = value.float()
            batch[key] = value
        return self.preprocessor(batch)

    def _normalize_action(self, action: Tensor) -> Tensor:
        shape = action.shape
        flattened = action.reshape(-1, shape[-1])
        normalizer = next(step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep))
        normalized = normalizer._normalize_action(flattened, inverse=False)
        return normalized.reshape(shape)

    def _environment_action(self, action: Tensor) -> Tensor:
        return self.postprocessor(action).to(device=action.device, dtype=action.dtype)

    def reset(self) -> None:
        self.policy.reset()

    def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs) -> None:
        del args
        self.export_policy(save_directory, state_dict=state_dict, **kwargs)

    def can_generate(self) -> bool:
        return False

    def export_policy(self, output_dir: str | Path, *, state_dict=None, **kwargs) -> None:
        if kwargs:
            raise TypeError(f"Unsupported Gaussian actor export options: {sorted(kwargs)}")
        policy_state = self._native_policy_state_dict(state_dict)
        export_policy = GaussianActorPolicy(deepcopy(self.policy.config))
        export_policy.load_state_dict(policy_state, strict=True)
        export_policy.save_pretrained(output_dir)
        self.preprocessor.save_pretrained(output_dir)
        self.postprocessor.save_pretrained(output_dir)

    def sft_init(self) -> None:
        self.policy.actor.std_layer.requires_grad_(False)
        SupportSFTTraining.sft_init(self)

    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: nn.Module,
        actions: dict[str, Tensor],
        valids: Tensor,
        action_mask: Tensor | None = None,
        target_values: Tensor | None = None,
    ) -> Tensor:
        del tokenizer, target_values
        action = actions[ACTION]
        if action.ndim != 3 or action.shape[1] != 1:
            raise ValueError(f"GaussianActor SFT actions must have shape (batch, 1, action_dim), got {action.shape}")
        if action_mask is not None and tuple(action_mask.shape) != tuple(action.shape[:2]):
            raise ValueError(
                f"GaussianActor SFT action mask must have shape {tuple(action.shape[:2])}, "
                f"got {tuple(action_mask.shape)}"
            )

        observations = self.sac_forward_state_features(obs, None)
        target_action = self._normalize_action(action[:, 0])
        observation_features = self.policy.actor.encoder(observations)
        hidden = self.policy.actor.network(observation_features)
        predicted_action = torch.tanh(self.policy.actor.mean_layer(hidden))
        sample_loss = F.mse_loss(predicted_action, target_action, reduction="none").mean(dim=-1)

        valids = valids.to(device=sample_loss.device, dtype=sample_loss.dtype)
        if action_mask is not None:
            valids = valids * action_mask[:, 0].to(device=sample_loss.device, dtype=sample_loss.dtype)
        loss = (sample_loss * valids).sum() / valids.sum().clamp_min(1.0)
        self.sft_metrics["action_mse"] = loss.detach()
        return loss

    def sac_init(self) -> None:
        self.policy.actor.std_layer.requires_grad_(True)
        methods = ["sac_sample_actions", "sac_forward_actor", "sac_forward_state_features"]
        if self.critic is not None:
            methods.append("sac_forward_critic")
        for method in methods:
            register_fsdp_forward_method(self, method)

    @torch.no_grad()
    def sac_sample_actions(
        self, obs: DataProto, tokenizer: Optional[nn.Module] = None, eval: bool = False
    ) -> ModelOutput:
        observations = self.sac_forward_state_features(obs, tokenizer)
        normalized_action, log_prob, mean, _ = self.policy.actor(observations, std_scale=self.sac_std_scale)
        if eval:
            normalized_action = torch.tanh(mean)
            log_prob = None
        action = self._environment_action(normalized_action)
        return GaussianActorOutput(action.unsqueeze(1), log_prob)

    @torch.no_grad()
    def sac_get_critic_value(
        self, obs: DataProto, actions: ModelOutput, tokenizer: Optional[nn.Module] = None
    ) -> Tensor:
        observations = self.sac_forward_state_features(obs, tokenizer)
        return self.sac_forward_critic(
            {"action": actions.action},
            observations,
            use_target_network=False,
            method="min",
            requires_grad=False,
        ).float()

    def sac_get_critic_parameters(self) -> list[nn.Parameter]:
        if self.critic is None or self.critic_encoder is None:
            raise RuntimeError("GaussianActor critic is disabled")
        parameters = list(self.critic.heads.parameters())
        parameters.extend(parameter for parameter in self.critic_encoder.parameters() if parameter.requires_grad)
        return parameters

    def sac_get_named_actor_parameters(self) -> list[tuple[str, nn.Parameter]]:
        actor_parameter_ids = {id(parameter) for parameter in self.policy.get_optim_params()["actor"]}
        return [
            (name, parameter)
            for name, parameter in self.policy.named_parameters()
            if id(parameter) in actor_parameter_ids
        ]

    def sac_forward_critic(
        self,
        a: dict[str, Tensor],
        state_features: dict[str, Tensor],
        task_ids: Optional[Tensor] = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> Tensor:
        del task_ids
        if self.critic is None or self.critic_encoder is None:
            raise RuntimeError("GaussianActor critic is disabled")
        if requires_grad:
            encoded = self.critic_encoder(state_features)
        else:
            with torch.no_grad():
                encoded = self.critic_encoder(state_features)
        actions = self._normalize_action(a["action"]).reshape(a["action"].shape[0], -1)
        critic_inputs = torch.cat([encoded, actions], dim=-1)
        parameters = tuple(self.critic.heads.parameters())
        previous = tuple(parameter.requires_grad for parameter in parameters)
        if not requires_grad:
            self.critic.heads.requires_grad_(False)
        try:
            return self.critic(critic_inputs, target=use_target_network, method=method)
        finally:
            for parameter, value in zip(parameters, previous, strict=True):
                parameter.requires_grad_(value)

    def sac_forward_actor(
        self,
        state_features: dict[str, Tensor],
        task_ids: Optional[Tensor] = None,
        is_first_micro_batch: bool = False,
        noise_scale: Optional[float] = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        del task_ids, is_first_micro_batch, noise_scale
        normalized_action, log_prob, _, std = self.policy.actor(state_features, std_scale=self.sac_std_scale)
        metrics = {"pre_tanh_std_mean": std.detach().mean().item()}
        return self._environment_action(normalized_action).unsqueeze(1), log_prob, metrics

    def sac_forward_state_features(self, obs: DataProto, tokenizer: nn.Module | None) -> dict[str, Tensor]:
        del tokenizer
        processed = self._policy_batch(obs)
        return {key: processed[key] for key in self.policy.config.input_features}

    @torch.no_grad()
    def sac_update_target_network(self, tau: float) -> None:
        if self.critic is None:
            raise RuntimeError("GaussianActor critic is disabled")
        self.critic.update_target(tau)
