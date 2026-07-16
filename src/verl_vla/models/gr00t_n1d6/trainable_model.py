# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Trainable verl-vla composition around the official GR00T N1.6 policy."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import torch
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from torch import nn
from torch.distributed.fsdp import register_fsdp_forward_method
from transformers.modeling_utils import no_init_weights
from verl import DataProto

from verl_vla.models.base import SupportSFTTraining, TrainableVLAModelMixin

from .policy.libero_policy import LiberoGr00tInput, LiberoGr00tOutput, load_gr00t_processor


class _CpuBeta(torch.distributions.Beta):
    def __init__(self, concentration1: float, concentration0: float):
        super().__init__(
            torch.tensor(float(concentration1), dtype=torch.float32, device="cpu"),
            torch.tensor(float(concentration0), dtype=torch.float32, device="cpu"),
        )


_BETA_PATCH_LOCK = Lock()


def load_gr00t_n1d6_policy(path, *, config, torch_dtype):
    """Load the official policy while handling its meta-init Beta distribution."""
    import gr00t.model.gr00t_n1d6.gr00t_n1d6 as upstream_model

    with _BETA_PATCH_LOCK, no_init_weights():
        original_beta = upstream_model.Beta
        upstream_model.Beta = _CpuBeta
        try:
            return Gr00tN1d6.from_pretrained(path, config=config, torch_dtype=torch_dtype)
        finally:
            upstream_model.Beta = original_beta


def _rec_to_device_dtype(value: Any, *, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype if torch.is_floating_point(value) else value.dtype)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _rec_to_device_dtype(item, device=device, dtype=dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_rec_to_device_dtype(item, device=device, dtype=dtype) for item in value]
    return value


class Gr00tN1d6TrainableModel(nn.Module, TrainableVLAModelMixin, SupportSFTTraining):
    def __init__(self, policy: Gr00tN1d6):
        super().__init__()
        self.config = policy.config
        SupportSFTTraining.__init__(self, self.config)
        self.init_trainable_model(policy=policy)
        self._verl_processor = None
        self._verl_processor_training: bool | None = None

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def forward(self, *args, **kwargs):
        return self.policy(*args, **kwargs)

    def can_generate(self) -> bool:
        return False

    def sft_init(self):
        self.sft_metrics = {}
        register_fsdp_forward_method(self, "sft_loss")

    def sft_loss(self, obs, tokenizer, actions, valids, action_mask=None, target_values=None):
        del tokenizer, target_values
        processor = self._get_processor(training=True)
        policy_input = LiberoGr00tInput.from_data_proto(obs, actions=actions["action"])
        collated = policy_input.collate(processor, action_valid_mask=action_mask)
        inputs = _rec_to_device_dtype(collated["inputs"], device=self.device, dtype=torch.bfloat16)
        output = self.policy(inputs)
        element_loss = output["action_loss"]
        element_mask = output["action_mask"].to(element_loss.dtype)
        valid_weights = valids.to(device=element_loss.device, dtype=element_loss.dtype)
        valid_weights = valid_weights.reshape(-1, *([1] * (element_loss.ndim - 1)))
        weighted_mask = element_mask * valid_weights
        loss = (element_loss * valid_weights).sum() / weighted_mask.sum().clamp_min(1e-6)
        self.sft_metrics = {
            "sft/action_loss": loss.detach(),
            "sft/valid_action_fraction": element_mask.float().mean().detach(),
        }
        return loss

    def sac_init(self):
        register_fsdp_forward_method(self, "sac_sample_actions")

    def _get_processor(self, *, training: bool):
        if self._verl_processor is None:
            processor_path = getattr(self.config, "verl_processor_path", None) or getattr(
                self.config, "_name_or_path", None
            )
            if not processor_path:
                raise ValueError("GR00T requires config.verl_processor_path or a checkpoint processor.")
            self._verl_processor = load_gr00t_processor(
                str(processor_path), getattr(self.config, "norm_stats_path", None), training=training
            )
        elif self._verl_processor_training != training:
            self._verl_processor.train() if training else self._verl_processor.eval()
        self._verl_processor_training = training
        return self._verl_processor

    @torch.no_grad()
    def sac_sample_actions(self, obs: DataProto, tokenizer=None, eval: bool = False):
        del tokenizer, eval
        processor = self._get_processor(training=False)
        policy_input = LiberoGr00tInput.from_data_proto(obs)
        collated = _rec_to_device_dtype(policy_input.collate(processor), device=self.device, dtype=torch.bfloat16)
        model_pred = self.policy.get_action(**collated)
        return LiberoGr00tOutput.from_model_output(
            model_pred,
            processor=processor,
            policy_input=policy_input,
            action_chunk_size=int(getattr(self.config, "verl_action_chunk_size", 8)),
            device=self.device,
        )

    def export_policy(self, output_dir, *, state_dict=None):
        processor = self._get_processor(training=False)
        policy_state = self.extract_policy_state_dict(state_dict) if state_dict is not None else None
        original_processor_path = getattr(self.config, "verl_processor_path", None)
        original_norm_stats_path = getattr(self.config, "norm_stats_path", None)
        original_architectures = getattr(self.config, "architectures", None)
        self.config.verl_processor_path = None
        self.config.norm_stats_path = None
        self.config.architectures = ["Gr00tN1d6"]
        try:
            self.native_policy.save_pretrained(output_dir, state_dict=policy_state, safe_serialization=True)
            processor.save_pretrained(Path(output_dir))
        finally:
            self.config.verl_processor_path = original_processor_path
            self.config.norm_stats_path = original_norm_stats_path
            self.config.architectures = original_architectures


__all__ = ["Gr00tN1d6TrainableModel", "load_gr00t_n1d6_policy"]
