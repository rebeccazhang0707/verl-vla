# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared DSRL (latent-noise steering) configuration.

DSRL (Diffusion Steering via Reinforcement Learning, arXiv:2506.15799) keeps
the whole VLA frozen and trains a small SAC policy over the *initial noise*
``x0`` of the flow-matching action head. This config is model-agnostic and is
embedded by both the GR00T and pi0/pi05 adapter configs under the ``dsrl`` key;
model-derived dimensions (feature/state/noise widths, action horizon) are
resolved by each trainable model at build time, not stored here.

``actor_type`` selects the noise-actor architecture. Each implementation owns a
typed child config under its architecture name, while sampling bounds and
model-derived dimensions remain shared at this boundary.
"""

from __future__ import annotations

from typing import Any


class DSRLMLPActorConfig:
    DEFAULTS = {
        "hidden_dims": [256, 256, 256],
        "feature_latent_dim": 128,
        "state_latent_dim": 64,
    }

    def __init__(self, **values: Any) -> None:
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class DSRLTransformerActorConfig:
    DEFAULTS = {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 1,
        "dropout": 0.0,
        "activation": "gelu",
        "positional_dropout": 0.0,
        "log_std_init": 0.0,
    }

    def __init__(self, **values: Any) -> None:
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class DSRLSteeringConfig:
    _FLAT_ACTOR_FIELDS = {
        *DSRLMLPActorConfig.DEFAULTS,
        *DSRLTransformerActorConfig.DEFAULTS,
        "transformer_dropout",
        "transformer_activation",
    }
    DEFAULTS = {
        # Master switch. When True the VLA policy is fully frozen and SAC
        # trains only the noise actor (+ critic).
        "enabled": False,
        # Optional overrides for the model-derived actor input widths. None
        # resolves from the model (backbone feature dim / processor state dim).
        "feature_dim": None,
        "state_dim": None,
        # Noise-actor architecture: "mlp" (default) or "transformer" (posttrain
        # reference chunking actor).
        "actor_type": "mlp",
        # False (RLinf parity): one noise vector shared by every step of the
        # action chunk. True: an independent noise vector per horizon step.
        "noise_per_step": False,
        # tanh output bound; x0 lives in [-noise_bound, noise_bound]^d.
        "noise_bound": 1.0,
        # Pre-tanh Gaussian log-std clamp range.
        "log_std_min": -20.0,
        "log_std_max": 2.0,
    }

    def __init__(self, **values: Any) -> None:
        flat_actor_fields = self._FLAT_ACTOR_FIELDS.intersection(values)
        if flat_actor_fields:
            fields = ", ".join(sorted(flat_actor_fields))
            raise ValueError(f"DSRL actor settings must be nested under dsrl.mlp or dsrl.transformer: {fields}")
        mlp_values = dict(values.pop("mlp", {}) or {})
        transformer_values = dict(values.pop("transformer", {}) or {})
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)
        self.mlp = DSRLMLPActorConfig(**mlp_values)
        self.transformer = DSRLTransformerActorConfig(**transformer_values)

    def to_dict(self) -> dict[str, Any]:
        config = {name: value for name, value in vars(self).items() if name not in ("mlp", "transformer")}
        config["mlp"] = self.mlp.to_dict()
        config["transformer"] = self.transformer.to_dict()
        return config


__all__ = ["DSRLMLPActorConfig", "DSRLSteeringConfig", "DSRLTransformerActorConfig"]
