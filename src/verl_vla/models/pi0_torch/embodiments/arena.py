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

import torch
from typing_extensions import override
from verl.protocol import DataProto

from verl_vla.utils.image import image_to_float01, preprocess_image_batch_to_uint8

from .base import Pi0Input, Pi0Output

ARENA_CAMERA_KEY = "observation.images.robot_head_cam_rgb"
ARENA_PI0_STATE_DIM = 32
ARENA_ACTION_DIM = 50
ARENA_IMAGE_CROP_SIZE = 480
ARENA_IMAGE_RESIZE_SIZE = (224, 224)


def _image_batch_to_bchw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected Arena image batch with shape BCHW or BHWC, got {tuple(images.shape)}")
    if images.shape[1] in (3, 4):
        return images[:, :3]
    if images.shape[-1] in (3, 4):
        return images[..., :3].permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Expected Arena image batch with 3 or 4 channels, got {tuple(images.shape)}")


class ArenaPi0Input(Pi0Input):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "ArenaPi0Input":
        input = cls()

        raw_images = env_obs.batch[ARENA_CAMERA_KEY]
        device = raw_images.device
        images = preprocess_image_batch_to_uint8(
            _image_batch_to_bchw(raw_images),
            crop_size=ARENA_IMAGE_CROP_SIZE,
            resize_size=ARENA_IMAGE_RESIZE_SIZE,
        )
        images = image_to_float01(images).to(device=device)

        batch_size = images.shape[0]
        empty_images = torch.zeros(
            (batch_size, 3, images.shape[2], images.shape[3]),
            device=device,
            dtype=torch.bfloat16,
        )

        input.images = {
            "observation.images.cam_high": images.to(torch.bfloat16),
            "observation.images.cam_left_wrist": empty_images,
            "observation.images.cam_right_wrist": empty_images,
        }
        input.img_masks = [
            torch.ones((batch_size,), device=device, dtype=torch.bool),
            torch.zeros((batch_size,), device=device, dtype=torch.bool),
            torch.zeros((batch_size,), device=device, dtype=torch.bool),
        ]

        input.task = list(env_obs.non_tensor_batch["task"])
        state = env_obs.batch["observation.state"]
        state = torch.nn.functional.pad(
            state[..., :ARENA_PI0_STATE_DIM],
            (0, max(0, ARENA_PI0_STATE_DIM - state.shape[-1])),
        )
        input.state = state.to(device=device, dtype=torch.float32)
        return input


class ArenaPi0Output(Pi0Output):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "ArenaPi0Output":
        output = cls()
        action_chunk_size = int(model_output.get("action_chunk_size", model_output["full_action"].shape[1]))
        action = model_output["full_action"][:, :action_chunk_size]
        if action.shape[-1] < ARENA_ACTION_DIM:
            action = torch.nn.functional.pad(action, (0, ARENA_ACTION_DIM - action.shape[-1]))
        output.action = action[:, :, :ARENA_ACTION_DIM]
        output.log_prob = model_output.get("log_probs")
        return output


__all__ = ["ArenaPi0Input", "ArenaPi0Output"]
