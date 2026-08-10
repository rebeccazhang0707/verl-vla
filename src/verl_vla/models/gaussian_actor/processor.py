# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

# Adapted from LeRobot
# src/lerobot/policies/gaussian_actor/processor_gaussian_actor.py at
# commit 22bd7a2f489b367d8df42de803b1e8c4ca63a3f9. The explicit processor steps
# preserve the upstream pipeline on verl-vla's pinned LeRobot release.

"""Load and initialize native LeRobot Gaussian actor processors."""

from __future__ import annotations

from pathlib import Path

from lerobot.datasets.utils import load_stats
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_gaussian_actor_pre_post_processors(config, dataset_stats=None):
    """Build the default normalization pipelines used by LeRobot GaussianActor."""
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline(steps=input_steps, name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def load_gaussian_actor_processors(config, model_path: str | Path, *, dataset_root: str | Path | None):
    model_path = Path(model_path)
    preprocessor_path = model_path / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    postprocessor_path = model_path / f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    processor_files = (preprocessor_path.is_file(), postprocessor_path.is_file())
    if any(processor_files) and not all(processor_files):
        raise FileNotFoundError(
            f"Incomplete native Gaussian actor processor artifacts in {model_path}: both "
            f"{preprocessor_path.name} and {postprocessor_path.name} are required"
        )
    if all(processor_files):
        return make_pre_post_processors(config, pretrained_path=str(model_path))
    if dataset_root is None:
        raise FileNotFoundError(
            f"Native Gaussian actor processor artifacts are missing from {model_path}. "
            "Set model.adapter.processor_dataset_root when initializing from config."
        )
    stats = load_stats(Path(dataset_root))
    if stats is None:
        raise FileNotFoundError(f"LeRobot dataset statistics are missing from {Path(dataset_root) / 'meta/stats.json'}")
    return make_pre_post_processors(config, dataset_stats=stats)


__all__ = ["load_gaussian_actor_processors", "make_gaussian_actor_pre_post_processors"]
