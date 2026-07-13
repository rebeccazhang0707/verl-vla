# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Arena x GR00T policy IO adapter.

``ArenaGr00tInput`` reads arena obs keys (``observation.images.*`` /
``observation.state`` / ``task``) into the raw tensors :class:`GR00TN16Adapter`
consumes. Images are kept as raw ``uint8`` HWC -- the GR00T processor does its
own crop / resize / normalise.

``ArenaGr00tOutput`` slices the decoded action chunk for the env and carries the
normalised model action + log-prob for the SAC replay/critic.
"""

import numpy as np
import torch
from typing_extensions import override
from verl import DataProto

from .sac_io import Gr00tInput, Gr00tOutput


def _image_batch_to_bhwc_uint8(images: torch.Tensor) -> torch.Tensor:
    """Return an arena image batch as (B, H, W, C=3) ``uint8`` (processor-ready).

    Accepts BCHW or BHWC, 3 or 4 channels, ``uint8`` or float (``0..1`` or
    ``0..255``). Extra (alpha) channels are dropped and float frames are scaled to
    the ``0..255`` byte range the GR00T processor expects.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected Arena image batch with shape BCHW or BHWC, got {tuple(images.shape)}")
    if images.shape[1] in (3, 4) and images.shape[-1] not in (3, 4):
        images = images.permute(0, 2, 3, 1)  # BCHW -> BHWC
    if images.shape[-1] not in (3, 4):
        raise ValueError(f"Expected Arena image batch with 3 or 4 channels, got {tuple(images.shape)}")
    images = images[..., :3]
    if images.dtype != torch.uint8:
        images = images.float()
        if images.numel() > 0 and float(images.max()) <= 1.0:
            images = images * 255.0
        images = images.clamp(0.0, 255.0).round().to(torch.uint8)
    return images.contiguous()


class ArenaGr00tInput(Gr00tInput):
    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> "ArenaGr00tInput":
        model_input = cls()

        if env_obs.batch is not None:
            for key, tensor in env_obs.batch.items():
                if key.startswith("observation.images."):
                    model_input.images[key] = _image_batch_to_bhwc_uint8(tensor)

        state = env_obs.batch["observation.state"]
        model_input.state = state.to(dtype=torch.float32)
        model_input.task = list(env_obs.non_tensor_batch["task"])
        return model_input


class ArenaGr00tOutput(Gr00tOutput):
    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> "ArenaGr00tOutput":
        output = cls()

        full_action = model_output["full_action"]
        decoded = model_output.get("decoded_action")
        if decoded is None:
            # No decode available (e.g. actor-side forward): fall back to the
            # normalised action so callers still get a well-formed chunk.
            decoded = full_action
        if not torch.is_tensor(decoded):
            decoded = torch.as_tensor(np.asarray(decoded, dtype=np.float32))

        chunk = int(model_output.get("num_action_chunks", decoded.shape[1]))
        chunk = min(chunk, decoded.shape[1])

        output.action = decoded[:, :chunk]
        output.full_action = full_action
        output.log_prob = model_output.get("log_probs")
        return output
