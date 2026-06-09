# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Arena × GR00T **N1.6** (``Gr00tN1d6``) input/output adapter.

All preprocessing and de-normalisation is delegated to the checkpoint's own
``Gr00tN1d6Processor`` (loaded with ``AutoProcessor.from_pretrained``), mirroring
the canonical inference path in ``gr00t.policy.gr00t_policy.Gr00tPolicy``:

    raw obs ─► VLAStepData ─► processor(messages) ─► collator ─► model inputs
    model action_pred (normalised) ─► processor.decode_action ─► sim joints

This replaces the N1.5 hand-rolled Eagle/state code. Dimensions (action_horizon,
max_state_dim, embodiment id, sin/cos, relative actions, ...) are NOT hard-coded;
they come from the loaded model/processor.

Embodiment specs + the gr00t-free state helpers live in ``utils.py``.

"""

import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

from verl_vla.models.gr00t.utils import (
    GR1_STATE_GROUP_DIMS,
    load_embodiment_id,
    split_flat_state_to_groups,
)

logger = logging.getLogger(__name__)


class GR00TN16Adapter:
    """Builds GR00T N1.6 model inputs and decodes actions via the checkpoint processor.

    Mirrors ``Gr00tPolicy`` (gr00t.policy.gr00t_policy) but exposes the collated
    ``inputs`` dict so the SAC wrapper can run backbone/action-head sub-modules
    directly (Gr00tPolicy itself is inference-only / no-grad).
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "gr1",
        state_group_dims: "Optional[OrderedDict[str, int]]" = None,
    ):
        # Registers Gr00tN1d6Processor with AutoProcessor.
        import gr00t.model  # noqa: F401
        from gr00t.data.embodiment_tags import EmbodimentTag
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.eval()

        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        # Authoritative projector index from this checkpoint's embodiment_id.json
        # (falls back to the copied table in utils.py if the file is absent).
        self.embodiment_id = load_embodiment_id(embodiment_tag, model_path)
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator

        self.video_keys = list(self.modality_configs["video"].modality_keys)
        self.state_keys = list(self.modality_configs["state"].modality_keys)
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self.state_group_dims = state_group_dims or GR1_STATE_GROUP_DIMS

        logger.info(
            "GR00TN16Adapter: tag=%s embodiment_id=%d video=%s state=%s lang=%s",
            self.embodiment_tag.value, self.embodiment_id,
            self.video_keys, self.state_keys, self.language_key,
        )

    # -- input building -------------------------------------------------

    def _to_vla_step_data(self, image_hwc: np.ndarray, state_groups: dict, task: str):
        """Build a single-sample VLAStepData (T=1)."""
        from gr00t.data.types import VLAStepData

        images = {vk: [image_hwc] for vk in self.video_keys}
        states = {k: state_groups[k].reshape(1, -1).astype(np.float32) for k in self.state_keys}
        return VLAStepData(
            images=images,
            states=states,
            actions={},
            text=task,
            embodiment=self.embodiment_tag,
        )

    def build_inputs(
        self,
        full_image: torch.Tensor | np.ndarray,   # (B, H, W, C) uint8
        state_flat: torch.Tensor | np.ndarray,   # (B, 26) policy-order joints (raw)
        task_descriptions: list[str],
    ) -> tuple[dict, "OrderedDict[str, np.ndarray]"]:
        """Run the processor + collator → model-ready ``inputs`` dict.

        Returns:
            inputs:          dict of tensors for ``Gr00tN1d6`` (eagle_* + state +
                             embodiment_id), ready for ``model.get_action(inputs)``
                             or ``model.backbone`` / ``model.action_head``.
            raw_state_groups: {group: (B, 1, d)} raw (un-normalised) states, needed
                             by ``decode_action`` for relative→absolute conversion.
        """
        from gr00t.data.types import MessageType

        image_np = full_image.cpu().numpy() if isinstance(full_image, torch.Tensor) else np.asarray(full_image)
        state_np = state_flat.cpu().numpy() if isinstance(state_flat, torch.Tensor) else np.asarray(state_flat)
        B = image_np.shape[0]

        grouped = split_flat_state_to_groups(state_np, self.state_group_dims)

        processed_inputs = []
        for i in range(B):
            task = task_descriptions[i] if i < len(task_descriptions) else task_descriptions[-1]
            sample_groups = {k: grouped[k][i] for k in self.state_keys}
            vla = self._to_vla_step_data(image_np[i], sample_groups, task)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
            processed_inputs.append(self.processor(messages))

        collated = self.collate_fn(processed_inputs)
        inputs = collated["inputs"] if "inputs" in collated else collated

        # raw states keyed by modality, shaped (B, T, d) for decode_action
        raw_state_groups: OrderedDict[str, np.ndarray] = OrderedDict(
            (k, grouped[k].reshape(B, 1, -1).astype(np.float32)) for k in self.state_keys
        )
        return inputs, raw_state_groups

    # -- output decoding ------------------------------------------------

    def decode_actions(
        self,
        normalized_action: np.ndarray,                     # (B, horizon, max_action_dim)
        raw_state_groups: "OrderedDict[str, np.ndarray]",  # {group: (B, T, d)}
    ) -> dict[str, np.ndarray]:
        """Un-normalise (and convert relative→absolute) model actions → per-group joints."""
        return self.processor.decode_action(
            normalized_action, self.embodiment_tag, raw_state_groups
        )

    def decode_actions_flat(
        self,
        normalized_action: np.ndarray,
        raw_state_groups: "OrderedDict[str, np.ndarray]",
    ) -> np.ndarray:
        """Decoded actions concatenated back to flat (B, horizon, 26) policy order."""
        decoded = self.decode_actions(normalized_action, raw_state_groups)
        return np.concatenate([decoded[k] for k in self.state_group_dims], axis=-1)
