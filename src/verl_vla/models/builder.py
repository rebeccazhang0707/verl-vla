# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Explicit VLA model construction without Transformers AutoClass registration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch


def _apply_overrides(config, overrides: Mapping) -> None:
    for name, value in overrides.items():
        setattr(config, name, value)


def build_vla_model(model_config, *, torch_dtype: torch.dtype):
    architecture = model_config.native_architecture
    path = model_config.local_path
    overrides = dict(model_config.override_config)
    if "model_config" in overrides:
        overrides = dict(overrides["model_config"])

    if architecture == "pi0":
        from .pi0_torch import PI0TrainableModel

        return PI0TrainableModel.from_pretrained(
            path,
            adapter_config=dict(model_config.adapter),
            policy_config_overrides=overrides,
            torch_dtype=torch_dtype,
        )

    if architecture == "act":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.pretrained import SAFETENSORS_SINGLE_FILE

        from .act_torch import ACTTrainableModel
        from .act_torch.processor import load_act_processors

        if overrides:
            raise ValueError("ACT architecture is checkpoint-owned; model.override_config must be empty")
        model_path = Path(path)
        weights_path = model_path / SAFETENSORS_SINGLE_FILE
        initialization_path = model_path / "initialization.json"
        if weights_path.is_file():
            policy = ACTPolicy.from_pretrained(path)
        else:
            if not initialization_path.is_file():
                raise FileNotFoundError(
                    f"Native ACT weights are missing at {weights_path}. Config-only initialization requires "
                    f"an explicit {initialization_path.name} sidecar."
                )
            with initialization_path.open(encoding="utf-8") as file:
                initialization = json.load(file)
            if initialization != {"type": "act_config"}:
                raise ValueError(f"Unsupported ACT initialization metadata in {initialization_path}")
            config = PreTrainedConfig.from_pretrained(path)
            if not isinstance(config, ACTConfig):
                raise TypeError(f"Expected a native ACT config at {path}, got {type(config).__name__}")
            policy = ACTPolicy(config)
        adapter_config = dict(model_config.adapter)
        processor_dataset_root = adapter_config.pop("processor_dataset_root", None)
        preprocessor, postprocessor = load_act_processors(
            policy.config,
            model_path,
            dataset_root=processor_dataset_root,
        )
        policy.to(dtype=torch_dtype)
        return ACTTrainableModel(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            adapter_config=adapter_config,
            model_path=path,
        )

    if architecture == "gr00t_n1d6":
        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

        from .gr00t_n1d6.adapter_config import Gr00tAdapterConfig
        from .gr00t_n1d6.trainable_model import Gr00tN1d6TrainableModel, load_gr00t_n1d6_policy

        config = Gr00tN1d6Config.from_pretrained(path)
        _apply_overrides(config, overrides)
        adapter_config = Gr00tAdapterConfig(model_path=path, **dict(model_config.adapter))

        policy = load_gr00t_n1d6_policy(path, config=config, torch_dtype=torch_dtype)
        return Gr00tN1d6TrainableModel(policy, adapter_config=adapter_config)

    if architecture == "openvla_oft":
        from .openvla_oft.configuration_prismatic import OpenVLAConfig
        from .openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        from .openvla_oft.trainable_model import OpenVLATrainableModel

        config = OpenVLAConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        policy = OpenVLAForActionPrediction.from_pretrained(path, config=config, torch_dtype=torch_dtype)
        return OpenVLATrainableModel(policy)

    if architecture == "recap_value_critic":
        from .recap_value_critic import ReCapValueCriticConfig, ReCapValueCriticTrainableModel

        config = ReCapValueCriticConfig.from_pretrained(path)
        _apply_overrides(config, overrides)
        return ReCapValueCriticTrainableModel.from_pretrained(path, config=config, torch_dtype=torch_dtype)

    raise ValueError(f"Unsupported VLA architecture: {architecture!r}")


__all__ = ["build_vla_model"]
