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

from dataclasses import dataclass
from typing import Any

from transformers import PretrainedConfig


@dataclass(frozen=True)
class GemmaExpertConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_expert_config(variant: str) -> GemmaExpertConfig:
    if variant == "gemma_1m":
        return GemmaExpertConfig(width=128, depth=4, mlp_dim=448, num_heads=1, num_kv_heads=1, head_dim=256)
    if variant == "gemma_50m":
        return GemmaExpertConfig(width=384, depth=18, mlp_dim=1536, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_100m":
        return GemmaExpertConfig(width=512, depth=18, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma3_100m":
        return GemmaExpertConfig(width=512, depth=18, mlp_dim=2048, num_heads=4, num_kv_heads=1, head_dim=256)
    if variant == "gemma_150m":
        return GemmaExpertConfig(width=640, depth=18, mlp_dim=2560, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_300m":
        return GemmaExpertConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return GemmaExpertConfig(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
    raise ValueError(f"Unknown ReCap value expert variant: {variant}.")


class ReCapValueCriticConfig(PretrainedConfig):
    model_type = "recap_value_critic"

    def __init__(
        self,
        siglip_path: str = "google/siglip-so400m-patch14-384",
        gemma3_path: str = "google/gemma-3-270m",
        tokenizer_path: str | None = None,
        critic_expert_variant: str = "gemma3_100m",
        image_size: int = 384,
        max_token_len: int = 200,
        include_state_in_prompt: bool = False,
        max_state_dim: int = 32,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        stop_gradient_to_vlm: bool = False,
        precision: str = "bfloat16",
        value_head_num_bins: int = 201,
        value_head_v_min: float = -1.0,
        value_head_v_max: float = 0.0,
        value_head_dropout: float = 0.0,
        value_head_hidden_dim: int = 0,
        sft_type: str = "value_model",
        **kwargs,
    ):
        kwargs["architectures"] = ["ReCapValueCriticTrainableModel"]
        super().__init__(**kwargs)
        self.siglip_path = siglip_path
        self.gemma3_path = gemma3_path
        self.tokenizer_path = tokenizer_path or gemma3_path
        self.critic_expert_variant = critic_expert_variant
        self.image_size = image_size
        self.max_token_len = max_token_len
        self.include_state_in_prompt = include_state_in_prompt
        self.max_state_dim = max_state_dim
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_vlm = freeze_vlm
        self.stop_gradient_to_vlm = stop_gradient_to_vlm
        self.precision = precision
        self.sft_type = sft_type
        self.value_head_hidden_dim = value_head_hidden_dim
        self.value_head_num_bins = value_head_num_bins
        self.value_head_v_min = value_head_v_min
        self.value_head_v_max = value_head_v_max
        self.value_head_dropout = value_head_dropout
        self.refresh_derived_fields()

    def to_dict(self) -> dict[str, Any]:
        return _sanitize_json_keys(super().to_dict())

    def to_diff_dict(self) -> dict[str, Any]:
        return _sanitize_json_keys(super().to_diff_dict())

    def refresh_derived_fields(self):
        expert_config = get_gemma_expert_config(self.critic_expert_variant)
        self.value_head_input_dim = expert_config.width
        self.num_attention_heads = expert_config.num_heads
        self.num_key_value_heads = expert_config.num_kv_heads
        return self


def _sanitize_json_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json_keys(item) for key, item in value.items() if key is not None}
    if isinstance(value, list):
        return [_sanitize_json_keys(item) for item in value]
    return value
