# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import glob
import os
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers import PreTrainedModel, ResNetModel
from verl import DataProto

from verl_vla.models.base import SupportSFTTraining
from verl_vla.utils.models.value_head import DistributionalValueHead

from .configuration_recap_resnet18_value_critic import ReCapResNet18ValueCriticConfig


class ReCapResNet18ValueFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ReCapResNet18ValueCriticTrainableModel(PreTrainedModel, SupportSFTTraining):
    config_class = ReCapResNet18ValueCriticConfig
    base_model_prefix = "recap_resnet18_value_critic"

    def __init__(self, config: ReCapResNet18ValueCriticConfig):
        PreTrainedModel.__init__(self, config)
        SupportSFTTraining.__init__(self, config)
        self.vision_encoder = ResNetModel.from_pretrained(config.resnet18_path)
        vision_dim = self.vision_encoder.config.hidden_sizes[-1]
        self.fusion = ReCapResNet18ValueFusion(2 * vision_dim + config.state_dim, config.fusion_hidden_dim)
        self.value_head = DistributionalValueHead(
            input_dim=config.fusion_hidden_dim,
            hidden_dim=config.value_head_hidden_dim,
            num_bins=config.value_head_num_bins,
            v_min=config.value_head_v_min,
            v_max=config.value_head_v_max,
            dropout=config.value_head_dropout,
        )
        if config.freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
            self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vision_encoder.eval()
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        del model_args
        config = kwargs.pop("config", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        kwargs.pop("trust_remote_code", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)
        state_dict = cls._load_optional_state_dict(pretrained_model_name_or_path)
        if state_dict:
            model.load_state_dict(state_dict, strict=True)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        return model

    @staticmethod
    def _load_optional_state_dict(path_or_repo: str | os.PathLike) -> dict[str, torch.Tensor]:
        path = Path(path_or_repo)
        if not path.is_dir():
            return {}
        safetensor_paths = sorted(glob.glob(str(path / "*.safetensors")))
        if safetensor_paths:
            state_dict: dict[str, torch.Tensor] = {}
            for weight_path in safetensor_paths:
                state_dict.update(safetensors.torch.load_file(weight_path, device="cpu"))
            return state_dict
        return {}

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected image batch shaped (B, 3, H, W), got {tuple(images.shape)}.")
        images = images.float()
        if images.shape[-2:] != (self.config.image_size, self.config.image_size):
            images = F.interpolate(
                images,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (images - mean) / std

    def value_model_forward_features(self, obs: DataProto) -> torch.Tensor:
        image_features = []
        for image_key in self.config.image_keys:
            images = self._normalize_images(obs.batch[image_key])
            if self.config.freeze_vision_encoder:
                with torch.no_grad():
                    features = self.vision_encoder(images).pooler_output.flatten(1)
            else:
                features = self.vision_encoder(images).pooler_output.flatten(1)
            image_features.append(features)

        state = obs.batch["observation.state"]
        if state.ndim != 2 or state.shape[1] != self.config.state_dim:
            raise ValueError(
                f"Expected observation.state shaped (B, {self.config.state_dim}), got {tuple(state.shape)}."
            )
        features = torch.cat([*image_features, state.to(dtype=image_features[0].dtype)], dim=-1)
        return self.fusion(features)

    def forward(self, obs: DataProto, tokenizer: torch.nn.Module | None = None) -> torch.Tensor:
        del tokenizer
        values, _, _ = self.value_head(self.value_model_forward_features(obs))
        return values

    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del tokenizer, actions, action_mask
        if target_values is None:
            raise ValueError("ReCap value-model SFT requires target_values.")
        _, logits, _ = self.value_head(self.value_model_forward_features(obs))
        per_sample_loss, metrics = self.value_head.loss(
            logits,
            target_values.to(device=logits.device).float(),
            reduction="none",
        )
        self.sft_metrics = {f"value/{name}": value for name, value in metrics.items()}
        valid_mask = valids.float().view(-1).to(device=per_sample_loss.device)
        return (per_sample_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
