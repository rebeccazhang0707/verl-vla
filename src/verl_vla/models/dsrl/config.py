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

The noise actor is a Transformer chunking policy that mirrors the posttrain
reference DSRL actor (``modules/transformer/actor.py``); the fields below are
its transformer geometry plus the tanh-Gaussian output settings.
"""

from __future__ import annotations

from typing import Any


class DSRLSteeringConfig:
    DEFAULTS = {
        # Master switch. When True the VLA policy is fully frozen and SAC
        # trains only the noise actor (+ critic).
        "enabled": False,
        # Optional overrides for the model-derived actor input widths. None
        # resolves from the model (backbone feature dim / processor state dim).
        "feature_dim": None,
        "state_dim": None,
        # Observation feature source for the noise actor + Transformer critic:
        #   "gr00t" (default, recommended): the frozen VLA's own mean-pooled VL
        #       prefix, aligned with the flow head the noise steers.
        #   "dino" (posttrain parity): an independent frozen DINOv2 embedding of
        #       the camera frame. Implemented for the GR00T integration only.
        "feature_source": "gr00t",
        "dino_model": "facebook/dinov2-base",
        "dino_image_size": 224,
        # Transformer geometry of the noise actor (posttrain reference actor).
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 1,
        # posttrain uses 0.1, but relies on eval() mode to disable it; the online
        # rollout here samples the actor without a guaranteed eval() toggle, so we
        # default to 0.0 to keep deterministic (eval) steering noise reproducible.
        "transformer_dropout": 0.0,
        "transformer_activation": "gelu",
        "positional_dropout": 0.0,
        # True (default, GR00T/posttrain parity): an independent latent per flow
        # horizon step. False (RLinf / pi0 broadcast): one noise vector shared by
        # every step of the action chunk (a single-token transformer).
        "noise_per_step": True,
        # True (default, posttrain parity): steer only the real action DOF
        # (action_dim); the padding columns [action_dim:max_action_dim] of x0 are
        # not free SAC dims. False (RLinf): steer the full padded max_action_dim.
        # Only the GR00T integration implements the real-dims x0 build.
        "noise_real_dims_only": True,
        # How the padding columns of x0 are filled when noise_real_dims_only is
        # set (GR00T only). True (default): tile the steered block across the full
        # max_action_dim so no dim is random -> deterministic decode, the actor's
        # noise fully determines the action (posttrain tile_dims). False: draw the
        # padding fresh N(0, 1) each call (posttrain fresh_base) -> the base
        # policy's native behavioral noise leaks into the decode (non-reproducible
        # when max_action_dim >> action_dim, e.g. GR1's 128 vs 26).
        "tile_dims": True,
        # tanh output bound; x0 lives in [-noise_bound, noise_bound]^d.
        "noise_bound": 1.0,
        # Initial bias of the (zero-weight) log-std head -> initial std.
        "log_std_init": 0.0,
        # Pre-tanh Gaussian log-std clamp range.
        "log_std_min": -20.0,
        "log_std_max": 2.0,
    }

    def __init__(self, **values: Any) -> None:
        for name, value in {**self.DEFAULTS, **values}.items():
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


__all__ = ["DSRLSteeringConfig"]
