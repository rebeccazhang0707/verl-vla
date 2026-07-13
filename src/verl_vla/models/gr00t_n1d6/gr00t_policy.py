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

"""Arena x GR00T **N1.6** (``Gr00tN1d6``) input/output adapter.

All preprocessing and de-normalisation is delegated to the checkpoint's own
``Gr00tN1d6Processor`` (loaded with ``AutoProcessor.from_pretrained``), mirroring
the canonical inference path in ``gr00t.policy.gr00t_policy.Gr00tPolicy``::

    raw obs -> VLAStepData -> processor(messages) -> collator -> model inputs
    model action_pred (normalised) -> processor.decode_action -> sim joints

This replaces the N1.5 hand-rolled Eagle/state code. Dimensions (action_horizon,
max_state_dim, embodiment id, sin/cos, relative actions, ...) are NOT hard-coded;
they come from the loaded model/processor.

Embodiment specs + the gr00t-free state helpers live in ``utils.py``. The gr00t
package is imported lazily inside ``__init__`` / methods so this module stays
importable (for typing / registration) without gr00t installed.
"""

import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

from .utils import GR1, GR00TDim, load_embodiment_id, split_flat_state_to_groups

logger = logging.getLogger(__name__)


def _to_numpy_batch(image) -> np.ndarray:
    return image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)


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
        self.action_keys = list(self.modality_configs["action"].modality_keys)
        self.language_key = self.modality_configs["language"].modality_keys[0]
        # Per-key (group) widths. Authoritative source = the checkpoint's own
        # StateActionProcessor norm_params (computed from statistics.json), so the
        # adapter is embodiment-agnostic (gr1: left_arm/right_arm/...; libero_panda:
        # eef pos/quat/gripper; etc). Falls back to the explicit arg, then GR1.
        # NOTE: state and action key sets can DIFFER (e.g. libero state carries a
        # 4-d quaternion but the action carries a 3-d rotation), so they are derived
        # independently and the action decode is concatenated in ACTION-key order.
        self.state_group_dims = state_group_dims or self._derive_group_dims("state") or GR1.state_group_dims
        self.action_group_dims = self._derive_group_dims("action") or GR1.state_group_dims

        logger.info(
            "GR00TN16Adapter: tag=%s embodiment_id=%d video=%s state=%s(%s) action=%s(%s) lang=%s",
            self.embodiment_tag.value, self.embodiment_id, self.video_keys,
            self.state_keys, dict(self.state_group_dims),
            self.action_keys, dict(self.action_group_dims), self.language_key,
        )

    def _derive_group_dims(self, modality: str) -> "Optional[OrderedDict[str, int]]":
        """Per-key widths for ``modality`` ('state'|'action') from the checkpoint.

        Reads the raw (pre sin/cos-encoding) per-group dimension that the processor
        computed from this checkpoint's ``statistics.json``. Returns ``None`` (so the
        caller can fall back) if the processor does not expose norm_params.
        """
        keys = self.state_keys if modality == "state" else self.action_keys
        try:
            norm = self.processor.state_action_processor.norm_params[self.embodiment_tag.value][modality]
            return OrderedDict((k, int(np.asarray(norm[k]["dim"]).item())) for k in keys)
        except Exception as exc:  # noqa: BLE001 - never let stats introspection break loading
            logger.warning(
                "GR00TN16Adapter: could not derive %s group dims from processor (%s); "
                "falling back to GR1 layout", modality, exc,
            )
            return None

    @property
    def action_dim(self) -> int:
        """Real (unpadded) action width = sum of the per-group action dims."""
        return sum(self.action_group_dims.values())

    # -- input building -------------------------------------------------

    def _to_vla_step_data(self, images_by_view: dict, state_groups: dict, task: str):
        """Build a single-sample VLAStepData (T=1).

        ``images_by_view`` maps every key in ``self.video_keys`` to that sample's
        (H, W, C) image.
        """
        from gr00t.data.types import VLAStepData

        images = {vk: [images_by_view[vk]] for vk in self.video_keys}
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
        images: dict[str, torch.Tensor | np.ndarray],
        state_flat: torch.Tensor | np.ndarray,
        task_descriptions: list[str],
    ) -> tuple[dict, "OrderedDict[str, np.ndarray]"]:
        """Run the processor + collator -> model-ready ``inputs`` dict.

        ``images`` maps ``observation.images.<name>`` -> ``(B, H, W, C) uint8`` in
        camera order. Cameras are mapped onto the checkpoint's ``video_keys`` BY
        POSITION (camera 0 -> ``video_keys[0]``, ...); if fewer cameras than video
        keys are supplied the first (primary) camera fills the remainder, so a
        single-view env still drives a multi-view checkpoint.
        """
        from gr00t.data.types import MessageType

        if not images:
            raise KeyError("No observation.images.* frames provided")
        image_batches = [_to_numpy_batch(v) for v in images.values()]
        state_np = _to_numpy_batch(state_flat)
        B = image_batches[0].shape[0]

        grouped = split_flat_state_to_groups(state_np, self.state_group_dims)

        processed_inputs = []
        for i in range(B):
            task = task_descriptions[i] if i < len(task_descriptions) else task_descriptions[-1]
            sample_groups = {k: grouped[k][i] for k in self.state_keys}
            images_by_view = {
                vk: (image_batches[vi] if vi < len(image_batches) else image_batches[0])[i]
                for vi, vk in enumerate(self.video_keys)
            }
            vla = self._to_vla_step_data(images_by_view, sample_groups, task)
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
        """Un-normalise (and convert relative->absolute) model actions -> per-group joints."""
        return self.processor.decode_action(
            normalized_action, self.embodiment_tag, raw_state_groups
        )

    def decode_actions_flat(
        self,
        normalized_action: np.ndarray,
        raw_state_groups: "OrderedDict[str, np.ndarray]",
    ) -> np.ndarray:
        """Decoded actions concatenated back to flat (B, horizon, action_dim) policy order.

        Iterates the ACTION modality keys (not the state keys): for embodiments where
        the two differ (e.g. libero: 8-d state vs 7-d action) the decoded dict is keyed
        by action groups, so concatenating in action-key order yields the env action.
        """
        decoded = self.decode_actions(normalized_action, raw_state_groups)
        return np.concatenate([decoded[k] for k in self.action_keys], axis=-1)

    def encode_actions_flat(
        self,
        env_action: np.ndarray,                            # (B, horizon, action_dim)
        raw_state_groups: "OrderedDict[str, np.ndarray]",  # {group: (B, T, d)}
        *,
        max_action_dim: int,
        max_action_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalise env-space flat actions -> padded model space (inverse of decode).

        Mirrors ``Gr00tN1d6Processor.__call__`` action path: split by action keys ->
        ``state_action_processor.apply_action`` (absolute->relative + norm) -> pad to
        ``(max_action_horizon, max_action_dim)`` with an ``action_mask``.

        Returns:
            normalized: ``(B, max_action_horizon, max_action_dim)`` float32
            action_mask: same shape, 1 on real (horizon, action_dim) entries
        """
        env_action = np.asarray(env_action, dtype=np.float32)
        if env_action.ndim != 3:
            raise ValueError(f"env_action must be (B, horizon, action_dim), got {env_action.shape}")
        B, horizon, action_dim = env_action.shape
        expected = self.action_dim
        if action_dim < expected:
            raise ValueError(
                f"env_action last dim {action_dim} < adapter action_dim {expected}; "
                "cannot encode a truncated demo action."
            )
        if action_dim > expected:
            env_action = env_action[..., :expected]
            action_dim = expected

        # Split flat policy-order joints into per-group dicts (same order as decode).
        action_groups: OrderedDict[str, np.ndarray] = OrderedDict()
        start = 0
        for key in self.action_keys:
            d = int(self.action_group_dims[key])
            action_groups[key] = env_action[..., start : start + d]
            start += d

        sap = self.processor.state_action_processor
        tag = self.embodiment_tag.value
        normalized_batches: list[np.ndarray] = []
        for i in range(B):
            sample_action = {k: v[i] for k, v in action_groups.items()}  # (H, d_k)
            sample_state = {k: v[i] for k, v in raw_state_groups.items()}  # (T, d_k)
            processed = sap.apply_action(sample_action, tag, state=sample_state)
            flat = np.concatenate([processed[k] for k in self.action_keys], axis=-1)  # (H, D)
            # Pad width then horizon (matches processor training collation).
            if flat.shape[-1] < max_action_dim:
                flat = np.concatenate(
                    [flat, np.zeros((flat.shape[0], max_action_dim - flat.shape[-1]), dtype=np.float32)],
                    axis=-1,
                )
            h = flat.shape[0]
            if h < max_action_horizon:
                flat = np.concatenate(
                    [flat, np.zeros((max_action_horizon - h, max_action_dim), dtype=np.float32)],
                    axis=0,
                )
            elif h > max_action_horizon:
                flat = flat[:max_action_horizon]
                h = max_action_horizon
            normalized_batches.append(flat.astype(np.float32, copy=False))

        normalized = np.stack(normalized_batches, axis=0)
        action_mask = np.zeros_like(normalized, dtype=np.float32)
        real_h = min(horizon, max_action_horizon)
        action_mask[:, :real_h, :action_dim] = 1.0
        return normalized, action_mask
