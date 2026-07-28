# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Trainable composition around the native OpenVLA-OFT policy."""

from __future__ import annotations

from verl_vla.models.base import TrainableVLAModelBase

from .modeling_prismatic import OpenVLAForActionPrediction


class OpenVLATrainableModel(TrainableVLAModelBase):
    def __init__(self, policy: OpenVLAForActionPrediction):
        super().__init__(policy=policy)
        self.config = policy.config

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def can_generate(self) -> bool:
        return False

    def predict_action(self, *args, **kwargs):
        return self.policy.predict_action(*args, **kwargs)

    def generate_action_verl(self, *args, **kwargs):
        return self.policy.generate_action_verl(*args, **kwargs)

    def export_policy(self, output_dir, *, state_dict=None):
        policy_state = self.extract_policy_state_dict(state_dict) if state_dict is not None else None
        self.native_policy.save_pretrained(output_dir, state_dict=policy_state, safe_serialization=True)


__all__ = ["OpenVLATrainableModel"]
