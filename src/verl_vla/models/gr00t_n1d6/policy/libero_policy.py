# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LIBERO adapter for the external GR00T N1.6 policy.

Embodiment-specific concerns live here (obs key mapping, gripper semantics,
flat LeRobot stats conversion). Processor bridging and SAC encode/decode live
in :class:`~verl_vla.models.gr00t_n1d6.gr00t_adapter.GR00TN16Adapter`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from typing_extensions import override
from verl import DataProto

from .arena_policy import _image_batch_to_bhwc_uint8
from .base import Gr00tInput, Gr00tOutput

# Flat LeRobot LIBERO statistics layout (dataset format, not processor keys).
# Runtime group dims come from the checkpoint processor via GR00TN16Adapter.
LIBERO_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
LIBERO_STATE_GROUP_DIMS: OrderedDict[str, int] = OrderedDict(x=1, y=1, z=1, roll=1, pitch=1, yaw=1, gripper=2)
LIBERO_ACTION_GROUP_DIMS: OrderedDict[str, int] = OrderedDict(x=1, y=1, z=1, roll=1, pitch=1, yaw=1, gripper=1)


def _slices_from_group_dims(group_dims: OrderedDict[str, int]) -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for key, width in group_dims.items():
        end = start + int(width)
        slices[key] = (start, end)
        start = end
    return slices


LIBERO_STATE_SLICES = _slices_from_group_dims(LIBERO_STATE_GROUP_DIMS)
LIBERO_ACTION_SLICES = _slices_from_group_dims(LIBERO_ACTION_GROUP_DIMS)

# Preferred DataProto camera order for LIBERO (adapter maps onto video_keys by position).
LIBERO_IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")


def libero_gripper_to_gr00t(action: np.ndarray) -> np.ndarray:
    """Map LIBERO's ``-1=open, +1=close`` command to GR00T's ``1=open, 0=close``.

    The LeRobot LIBERO dataset stores the simulator command, while the official
    GR00T LIBERO policy and its normalization statistics use the standardized
    ``[0, 1]`` representation. Keep this conversion on the policy boundary so
    the shared LeRobot dataloader remains model agnostic.
    """
    converted = np.asarray(action, dtype=np.float32).copy()
    converted[..., -1] = (1.0 - converted[..., -1]) * 0.5
    return converted


def _libero_gripper_stats_to_gr00t(stats: dict[str, list[float]]) -> dict[str, list[float]]:
    """Apply ``y = (1 - x) / 2`` to scalar summary statistics."""
    return {
        "min": [(1.0 - float(stats["max"][0])) * 0.5],
        "max": [(1.0 - float(stats["min"][0])) * 0.5],
        "mean": [(1.0 - float(stats["mean"][0])) * 0.5],
        "std": [float(stats["std"][0]) * 0.5],
        # The transform is decreasing, so the quantile endpoints swap.
        "q01": [(1.0 - float(stats["q99"][0])) * 0.5],
        "q99": [(1.0 - float(stats["q01"][0])) * 0.5],
    }


def _stats_for_slice(flat_stats: dict[str, Any], start: int, end: int) -> dict[str, list[float]]:
    required = ("min", "max", "mean", "std", "q01", "q99")
    missing = [name for name in required if name not in flat_stats]
    if missing:
        raise ValueError(f"Normalization statistics are missing {missing}.")
    return {name: [float(value) for value in flat_stats[name][start:end]] for name in required}


def load_libero_statistics(path: str | Path) -> dict[str, Any]:
    """Load official nested stats or convert flat raw-LIBERO statistics.

    Official nested GR00T statistics already contain a standardized ``[0, 1]``
    gripper and are returned unchanged. Flat statistics produced by the shared
    LeRobot script describe LIBERO's raw ``[-1, 1]`` simulator command, so its
    gripper summaries are transformed together with the training actions.
    """
    stats_path = Path(path).expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"NORM_STATS_PATH does not exist: {stats_path}")
    with stats_path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if "libero_panda" in raw:
        return {"libero_panda": raw["libero_panda"]}

    modality_slices = {
        "state": LIBERO_STATE_SLICES,
        "action": LIBERO_ACTION_SLICES,
    }
    for modality, slices in modality_slices.items():
        if modality not in raw:
            raise ValueError(f"Normalization statistics do not contain '{modality}': {stats_path}")
        lengths = {name: len(raw[modality].get(name, [])) for name in ("min", "max", "mean", "std", "q01", "q99")}
        expected_dim = max(end for _, end in slices.values())
        if any(length != expected_dim for length in lengths.values()):
            raise ValueError(f"Expected {expected_dim} {modality} statistics in {stats_path}, got {lengths}.")

    converted = {
        "libero_panda": {
            modality: {
                key: _stats_for_slice(raw[modality], start, end)
                for key, (start, end) in modality_slices[modality].items()
            }
            for modality in modality_slices
        }
    }
    converted["libero_panda"]["action"]["gripper"] = _libero_gripper_stats_to_gr00t(
        converted["libero_panda"]["action"]["gripper"]
    )
    return converted


def image_to_uint8_hwc(value: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert one CHW/HWC image in [0, 1] or [0, 255] to uint8 HWC."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if torch.is_floating_point(value):
            value = value.float()
        image = value.numpy()
    else:
        image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected one image with three dimensions, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image in CHW or HWC layout, got {image.shape}.")
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def prepare_libero_gripper_action(action: np.ndarray) -> np.ndarray:
    """Convert GR00T's [0, 1] gripper value to LIBERO's inverted {-1, 1} command."""
    prepared = np.asarray(action, dtype=np.float32).copy()
    prepared[..., -1] = -np.sign(2.0 * prepared[..., -1] - 1.0)
    return prepared


def load_gr00t_processor(model_path: str, norm_stats_path: str | None, *, training: bool):
    """Load the official LIBERO processor via the shared Adapter factory.

    Flat LeRobot statistics conversion is handled inside
    :meth:`GR00TN16Adapter.load_processor` for ``libero_panda``.
    """
    from ..gr00t_adapter import GR00TN16Adapter

    return GR00TN16Adapter.load_processor(
        model_path,
        embodiment_tag="libero_panda",
        override_modality_configs=True,
        use_relative_action=True,
        norm_stats_path=norm_stats_path,
        training=training,
    )


class LiberoGr00tInput(Gr00tInput):
    """LIBERO obs → raw tensors for :class:`GR00TN16Adapter`.

    Cameras are exposed in ``LIBERO_IMAGE_KEYS`` order; the adapter maps them onto
    checkpoint ``video_keys`` by position. Demo actions (if provided) are converted
    to GR00T gripper space here; Adapter then normalises.
    """

    @override
    @classmethod
    def from_env_obs(cls, env_obs: DataProto) -> LiberoGr00tInput:
        model_input = cls()
        for key in LIBERO_IMAGE_KEYS:
            if key not in env_obs.batch:
                raise KeyError(f"LIBERO obs is missing required image key {key!r}")
            model_input.images[key] = _image_batch_to_bhwc_uint8(env_obs.batch[key])
        model_input.state = env_obs.batch["observation.state"].to(dtype=torch.float32)
        model_input.task = list(env_obs.non_tensor_batch["task"])
        return model_input

    @classmethod
    def actions_to_processor_space(cls, actions: torch.Tensor) -> torch.Tensor:
        action_np = libero_gripper_to_gr00t(actions.detach().to(device="cpu", dtype=torch.float32).numpy())
        return torch.from_numpy(action_np)


class LiberoGr00tOutput(Gr00tOutput):
    """Decode GR00T actions and apply the official LIBERO gripper semantics."""

    @override
    @classmethod
    def from_model_output(cls, model_output: dict) -> LiberoGr00tOutput:
        output = cls()

        full_action = model_output.get("full_action")
        decoded = model_output.get("decoded_action")
        if decoded is None:
            decoded = full_action
        if decoded is None:
            raise KeyError("LiberoGr00tOutput requires decoded_action or full_action")
        if not torch.is_tensor(decoded):
            decoded = torch.as_tensor(np.asarray(decoded, dtype=np.float32))

        # Gripper: GR00T [0, 1] -> LIBERO inverted {-1, +1}.
        decoded_np = prepare_libero_gripper_action(decoded.detach().float().cpu().numpy())
        decoded = torch.from_numpy(decoded_np).to(device=decoded.device, dtype=torch.float32)

        chunk = int(model_output.get("num_action_chunks", decoded.shape[1]))
        chunk = min(chunk, decoded.shape[1])
        output.action = decoded[:, :chunk]
        output.full_action = full_action
        output.log_prob = model_output.get("log_probs")
        return output


__all__ = [
    "LIBERO_ACTION_GROUP_DIMS",
    "LIBERO_ACTION_SLICES",
    "LIBERO_IMAGE_KEYS",
    "LIBERO_KEYS",
    "LIBERO_STATE_GROUP_DIMS",
    "LIBERO_STATE_SLICES",
    "LiberoGr00tInput",
    "LiberoGr00tOutput",
    "image_to_uint8_hwc",
    "libero_gripper_to_gr00t",
    "load_gr00t_processor",
    "load_libero_statistics",
    "prepare_libero_gripper_action",
]
