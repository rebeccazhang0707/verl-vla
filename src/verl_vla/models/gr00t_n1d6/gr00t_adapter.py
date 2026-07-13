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
``Gr00tN1d6Processor`` (loaded via :meth:`GR00TN16Adapter.load_processor`),
mirroring the canonical inference path in
``gr00t.policy.gr00t_policy.Gr00tPolicy``::

    raw obs -> VLAStepData -> processor(messages) -> collator -> model inputs
    model action_pred (normalised) -> processor.decode_action -> sim joints

Dimensions (action_horizon, max_state_dim, embodiment id, sin/cos, relative
actions, ...) are NOT hard-coded; they come from the loaded model/processor.

Embodiment specs + the gr00t-free state helpers live in ``utils.py``. The gr00t
package is imported lazily inside ``__init__`` / methods so this module stays
importable (for typing / registration) without gr00t installed.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import torch

from .utils import GR1, load_embodiment_id, split_flat_state_to_groups

logger = logging.getLogger(__name__)


def _to_numpy_batch(image) -> np.ndarray:
    return image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)


def _numpy_float32(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


class GR00TN16Adapter:
    """Builds GR00T N1.6 model inputs and decodes actions via the checkpoint processor.

    Mirrors ``Gr00tPolicy`` (gr00t.policy.gr00t_policy) but exposes the collated
    ``inputs`` dict so the SAC wrapper can run backbone/action-head sub-modules
    directly (Gr00tPolicy itself is inference-only / no-grad).

    Also the single factory for processor loading used by SFT and SAC.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "gr1",
        state_group_dims: "Optional[OrderedDict[str, int]]" = None,
        *,
        processor: Any | None = None,
        norm_stats_path: str | None = None,
        statistics: Mapping[str, Any] | None = None,
        override_modality_configs: bool | None = None,
        use_relative_action: bool | None = None,
        training: bool = False,
    ):
        if processor is None:
            processor = self.load_processor(
                model_path,
                embodiment_tag=embodiment_tag,
                norm_stats_path=norm_stats_path,
                statistics=statistics,
                override_modality_configs=override_modality_configs,
                use_relative_action=use_relative_action,
                training=training,
            )
        else:
            processor.train() if training else processor.eval()

        self.processor = processor
        self._training = bool(training)

        from gr00t.data.embodiment_tags import EmbodimentTag

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

    # -- processor factory ------------------------------------------------

    @staticmethod
    def resolve_modality_configs(embodiment_tag: str) -> dict[str, Any]:
        """Build ``{tag: modality_config}`` from official ``MODALITY_CONFIGS``.

        Fills missing ``action_configs`` with ABSOLUTE / NON_EEF / DEFAULT so the
        processor matches validated upstream LIBERO (and similar) metadata.
        """
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType

        if embodiment_tag not in MODALITY_CONFIGS:
            known = sorted(MODALITY_CONFIGS)
            raise KeyError(f"Unknown embodiment_tag {embodiment_tag!r} for modality override; known: {known}")

        modality_config = deepcopy(MODALITY_CONFIGS[embodiment_tag])
        action_config = modality_config.get("action")
        if action_config is not None and action_config.action_configs is None:
            action_config.action_configs = [
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in action_config.modality_keys
            ]
        return {embodiment_tag: modality_config}

    @classmethod
    def load_processor(
        cls,
        model_path: str,
        *,
        embodiment_tag: str | None = None,
        modality_configs: Mapping[str, Any] | None = None,
        override_modality_configs: bool | None = None,
        use_relative_action: bool | None = None,
        norm_stats_path: str | None = None,
        statistics: Mapping[str, Any] | None = None,
        training: bool = False,
    ):
        """Single processor factory for SFT and SAC.

        Prefer ``AutoProcessor.from_pretrained`` when the checkpoint already
        carries the right modalities. Pass ``override_modality_configs=True``
        (or an explicit ``modality_configs``) when fine-tuning LIBERO from a
        base GR1 checkpoint.
        """
        # Registers Gr00tN1d6Processor with AutoProcessor.
        import gr00t.model  # noqa: F401
        from transformers import AutoProcessor

        load_kwargs: dict[str, Any] = {}
        tag = embodiment_tag
        should_override = override_modality_configs
        if modality_configs is not None:
            load_kwargs["modality_configs"] = dict(modality_configs)
            should_override = False
        elif should_override is None and tag is not None and tag != "gr1":
            # Auto-override only for official MODALITY_CONFIGS packs (e.g. libero_panda
            # when fine-tuning from a base GR1 checkpoint). Custom / Arena tags such as
            # ``new_embodiment`` are expected to already live in the checkpoint
            # processor and must not be remapped via MODALITY_CONFIGS.
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            should_override = tag in MODALITY_CONFIGS
        if should_override:
            if not tag:
                raise ValueError("override_modality_configs requires embodiment_tag")
            load_kwargs["modality_configs"] = cls.resolve_modality_configs(tag)
        if use_relative_action is not None:
            load_kwargs["use_relative_action"] = bool(use_relative_action)
        elif should_override or modality_configs is not None:
            # Match historical load_gr00t_processor default for overridden packs.
            load_kwargs["use_relative_action"] = True

        if load_kwargs:
            from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor

            processor = Gr00tN1d6Processor.from_pretrained(model_path, **load_kwargs)
        else:
            processor = AutoProcessor.from_pretrained(model_path)

        if statistics is not None:
            processor.set_statistics(dict(statistics), override=True)
        elif norm_stats_path:
            from .policy import get_statistics_loader

            loader = get_statistics_loader(tag)
            if loader is not None:
                processor.set_statistics(loader(norm_stats_path), override=True)
            else:
                import json

                with open(norm_stats_path, encoding="utf-8") as file:
                    processor.set_statistics(json.load(file), override=True)

        processor.train() if training else processor.eval()
        return processor

    def set_training(self, training: bool) -> None:
        """Toggle processor train/eval (image augs etc.) without reloading."""
        training = bool(training)
        if training == self._training:
            return
        self.processor.train() if training else self.processor.eval()
        self._training = training

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

    @property
    def state_dim(self) -> int:
        """Real (unpadded) state width = sum of the per-group state dims."""
        return sum(self.state_group_dims.values())

    # -- input building -------------------------------------------------

    def _split_flat_action(self, action_array: np.ndarray) -> dict[str, np.ndarray]:
        """Split ``(T, action_dim)`` flat actions into per-key groups."""
        expected = self.action_dim
        if action_array.ndim != 2 or action_array.shape[-1] != expected:
            raise ValueError(f"Expected actions shaped [T, {expected}], got {action_array.shape}.")
        out: dict[str, np.ndarray] = {}
        start = 0
        for key in self.action_keys:
            width = int(self.action_group_dims[key])
            out[key] = action_array[:, start : start + width]
            start += width
        return out

    def _to_vla_step_data(
        self,
        images_by_view: dict,
        state_groups: dict,
        task: str,
        actions: dict[str, np.ndarray] | None = None,
    ):
        """Build a single-sample VLAStepData (T=1 for state; action horizon from ``actions``)."""
        from gr00t.data.types import VLAStepData

        images = {vk: [images_by_view[vk]] for vk in self.video_keys}
        states = {k: state_groups[k].reshape(1, -1).astype(np.float32) for k in self.state_keys}
        return VLAStepData(
            images=images,
            states=states,
            actions=actions or {},
            text=task,
            embodiment=self.embodiment_tag,
        )

    def _apply_action_valid_mask(
        self,
        processed: dict[str, Any],
        action_valid_mask: torch.Tensor | np.ndarray | None,
    ) -> dict[str, Any]:
        if action_valid_mask is None or "action_mask" not in processed:
            return processed
        valid = torch.as_tensor(action_valid_mask, dtype=torch.bool).reshape(-1)
        mask = torch.as_tensor(processed["action_mask"]).clone()
        length = min(mask.shape[0], valid.shape[0])
        invalid_indices = torch.nonzero(~valid[:length], as_tuple=False).flatten()
        mask[invalid_indices] = 0
        processed = dict(processed)
        processed["action_mask"] = mask
        return processed

    def _process_to_numpy(self, processed: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for key, value in processed.items()
        }

    def build_collated(
        self,
        images: dict[str, torch.Tensor | np.ndarray],
        state_flat: torch.Tensor | np.ndarray,
        task_descriptions: list[str],
        *,
        actions: torch.Tensor | np.ndarray | None = None,
        action_valid_mask: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[Any, "OrderedDict[str, np.ndarray]"]:
        """Run processor + collator -> full collated batch (for ``get_action`` / SFT).

        ``images`` maps ``observation.images.<name>`` -> ``(B, H, W, C) uint8`` in
        camera order. Cameras are mapped onto the checkpoint's ``video_keys`` BY
        POSITION. Optional ``actions`` are ``(B, T, action_dim)`` already in the
        processor's action space (embodiment IO must convert gripper etc. first).
        """
        from gr00t.data.types import MessageType

        if not images:
            raise KeyError("No observation.images.* frames provided")
        image_batches = [_to_numpy_batch(v) for v in images.values()]
        state_np = _to_numpy_batch(state_flat)
        B = image_batches[0].shape[0]

        grouped = split_flat_state_to_groups(state_np, self.state_group_dims)

        action_np = None
        if actions is not None:
            action_np = _numpy_float32(actions)
            if action_np.ndim == 2:
                action_np = action_np[None, ...]
            if action_np.shape[0] != B:
                raise ValueError(f"actions batch {action_np.shape[0]} != image batch {B}")

        processed_inputs = []
        for i in range(B):
            task = task_descriptions[i] if i < len(task_descriptions) else task_descriptions[-1]
            sample_groups = {k: grouped[k][i] for k in self.state_keys}
            images_by_view = {
                vk: (image_batches[vi] if vi < len(image_batches) else image_batches[0])[i]
                for vi, vk in enumerate(self.video_keys)
            }
            sample_actions = None
            if action_np is not None:
                sample_actions = self._split_flat_action(action_np[i])
            vla = self._to_vla_step_data(images_by_view, sample_groups, task, actions=sample_actions)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla}]
            processed = self.processor(messages)
            sample_mask = None if action_valid_mask is None else action_valid_mask[i]
            processed = self._apply_action_valid_mask(processed, sample_mask)
            processed_inputs.append(self._process_to_numpy(processed))

        collated = self.collate_fn(processed_inputs)
        raw_state_groups: OrderedDict[str, np.ndarray] = OrderedDict(
            (k, grouped[k].reshape(B, 1, -1).astype(np.float32)) for k in self.state_keys
        )
        return collated, raw_state_groups

    def build_inputs(
        self,
        images: dict[str, torch.Tensor | np.ndarray],
        state_flat: torch.Tensor | np.ndarray,
        task_descriptions: list[str],
        *,
        actions: torch.Tensor | np.ndarray | None = None,
        action_valid_mask: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[dict, "OrderedDict[str, np.ndarray]"]:
        """Run the processor + collator -> model-ready ``inputs`` dict.

        Same as :meth:`build_collated` but returns the inner ``inputs`` payload
        used by ``policy.forward`` / backbone feature extraction.
        """
        collated, raw_state_groups = self.build_collated(
            images,
            state_flat,
            task_descriptions,
            actions=actions,
            action_valid_mask=action_valid_mask,
        )
        inputs = collated["inputs"] if isinstance(collated, Mapping) and "inputs" in collated else collated
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
