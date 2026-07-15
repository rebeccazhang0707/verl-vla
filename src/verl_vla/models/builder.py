# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Explicit VLA model construction without Transformers AutoClass registration."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def _apply_overrides(config, overrides: Mapping) -> None:
    for name, value in overrides.items():
        setattr(config, name, value)


def _split_gr00t_overrides(overrides: Mapping) -> tuple[dict, dict]:
    """Split legacy override_config into native HF overrides vs adapter fields."""
    from .gr00t_n1d6.adapter_config import GR00T_ADAPTER_OVERRIDE_KEYS

    native: dict = {}
    adapter: dict = {}
    for key, value in overrides.items():
        if key in GR00T_ADAPTER_OVERRIDE_KEYS:
            adapter[key] = value
        else:
            native[key] = value
    return native, adapter


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

    if architecture == "gr00t_n1d6":
        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

        from .gr00t_n1d6.adapter_config import Gr00tAdapterConfig
        from .gr00t_n1d6.trainable_model import Gr00tN1d6TrainableModel, load_gr00t_n1d6_policy

        config = Gr00tN1d6Config.from_pretrained(path)
        native_overrides, legacy_adapter = _split_gr00t_overrides(overrides)
        _apply_overrides(config, native_overrides)

        # Dual-write migration: prefer ``model/adapter=gr00t``; fold leftover SAC
        # keys from ``override_config``. Ignore the default pi0 adapter group when
        # a GR00T run still only sets ``model/override=gr00t``.
        adapter_raw = dict(model_config.adapter or {})
        looks_like_default_pi0 = (
            adapter_raw.get("embodiment") == "libero"
            and "embodiment_tag" not in adapter_raw
            and "policy_type" not in adapter_raw
            and "action_dim" not in adapter_raw
        )
        if looks_like_default_pi0:
            adapter_values = dict(legacy_adapter)
        else:
            adapter_values = {**legacy_adapter, **adapter_raw}
        adapter_config = Gr00tAdapterConfig(model_path=path, **adapter_values)

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
