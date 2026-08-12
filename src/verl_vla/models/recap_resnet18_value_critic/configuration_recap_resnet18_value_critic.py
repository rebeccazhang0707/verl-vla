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

from transformers import PretrainedConfig


class ReCapResNet18ValueCriticConfig(PretrainedConfig):
    model_type = "recap_resnet18_value_critic"

    def __init__(
        self,
        image_keys: tuple[str, str] | list[str] = (
            "observation.images.image",
            "observation.images.wrist_image",
        ),
        image_size: int = 224,
        resnet18_path: str = "microsoft/resnet-18",
        state_dim: int = 8,
        fusion_hidden_dim: int = 256,
        freeze_vision_encoder: bool = True,
        value_head_num_bins: int = 201,
        value_head_v_min: float = -1.0,
        value_head_v_max: float = 0.0,
        value_head_dropout: float = 0.0,
        value_head_hidden_dim: int = 0,
        sft_type: str = "value_model",
        **kwargs,
    ):
        kwargs["architectures"] = ["ReCapResNet18ValueCriticTrainableModel"]
        super().__init__(**kwargs)
        if len(image_keys) != 2:
            raise ValueError(f"ReCap ResNet-18 value critic requires exactly two image keys, got {image_keys}.")
        self.image_keys = list(image_keys)
        self.image_size = image_size
        self.resnet18_path = resnet18_path
        self.state_dim = state_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.freeze_vision_encoder = freeze_vision_encoder
        self.value_head_num_bins = value_head_num_bins
        self.value_head_v_min = value_head_v_min
        self.value_head_v_max = value_head_v_max
        self.value_head_dropout = value_head_dropout
        self.value_head_hidden_dim = value_head_hidden_dim
        self.sft_type = sft_type
