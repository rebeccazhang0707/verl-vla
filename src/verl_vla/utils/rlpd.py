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

import math
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor

from verl_vla.utils.dataloader.lerobot import (
    RLPDTransitionDataset,
)


def pad_dataproto_to_divisor_with_valid_mask(
    batch: DataProto,
    size_divisor: int,
    valid_key: str,
) -> DataProto:
    padded_batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
    if pad_size <= 0:
        return padded_batch

    valid_tensor = padded_batch.batch[valid_key].clone()
    if valid_tensor.dtype == torch.bool:
        valid_tensor[-pad_size:] = False
    else:
        valid_tensor[-pad_size:] = 0
    padded_batch.batch[valid_key] = valid_tensor
    return padded_batch


def iter_rlpd_replay_prefill_batches(config, global_steps: int):
    rlpd_config = OmegaConf.select(config, "data.rlpd")
    if not rlpd_config or not rlpd_config.get("enable", False):
        return

    action_chunk_steps = int(rlpd_config.get("action_chunk_steps", 10))
    max_prefill_transitions = int(rlpd_config.get("max_transitions", 0))
    submit_max_transitions = int(rlpd_config.get("submit_max_transitions", 0))
    num_workers = int(rlpd_config.get("num_workers", 0))
    prefetch_factor = int(rlpd_config.get("prefetch_factor", 2))
    requested_episodes = OmegaConf.to_container(rlpd_config.get("episodes")) if rlpd_config.get("episodes") else None

    transition_dataset = RLPDTransitionDataset(
        repo_id=rlpd_config.repo_id,
        action_chunk_steps=action_chunk_steps,
        root=rlpd_config.get("root"),
        revision=rlpd_config.get("revision"),
        video_backend=rlpd_config.get("video_backend"),
        episodes=requested_episodes,
        max_transitions=max_prefill_transitions,
    )
    if len(transition_dataset) == 0:
        return

    rlpd_config_dict = OmegaConf.to_container(rlpd_config, resolve=True)
    if submit_max_transitions > 0:
        batch_size = submit_max_transitions
    elif num_workers > 1:
        batch_size = max(1, math.ceil(len(transition_dataset) / num_workers))
    else:
        batch_size = len(transition_dataset)
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(
        transition_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(
            rlpd_transition_samples_to_actor_input,
            dataset=transition_dataset.dataset,
            rlpd_config=rlpd_config_dict,
            global_steps=global_steps,
        ),
        **dataloader_kwargs,
    )

    yield from tqdm(loader, total=len(loader), desc="RLPD replay prefill")


def extract_lerobot_step_actions(action: torch.Tensor) -> torch.Tensor:
    # TODO(remove after datasets are fixed): old LeRobot datasets stored an action chunk
    # inside each action entry. Standard RLPD data should be per-step action [T, D].
    if action.ndim >= 3:
        action = action[:, 0, :]
    # End legacy compatibility block.
    if action.ndim != 2:
        raise ValueError(f"Expected LeRobot actions with shape [T, D], got {tuple(action.shape)}")
    return action


def stack_lerobot_hf_rows(dataset, key: str, indices: list[int]) -> torch.Tensor:
    values = dataset.hf_dataset[key][indices]
    return torch.stack([value if torch.is_tensor(value) else torch.as_tensor(value) for value in values], dim=0)


def read_lerobot_action_chunks(dataset, start_indices: torch.Tensor, chunk_size: int) -> torch.Tensor:
    offsets = torch.arange(chunk_size, dtype=torch.long)
    chunk_indices = (start_indices[:, None] + offsets[None, :]).reshape(-1).tolist()
    raw_actions = stack_lerobot_hf_rows(dataset, "action", chunk_indices)
    step_actions = extract_lerobot_step_actions(raw_actions)
    return step_actions.reshape(len(start_indices), chunk_size, step_actions.shape[-1])


def stack_transition_observations(items: list[dict], prefix: str) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    fields = {}
    obs_keys = [
        key
        for key, value in items[0].items()
        if key.startswith("observation.") and torch.is_tensor(value) and not key.endswith("_is_pad")
    ]
    for key in obs_keys:
        value = torch.stack([item[key] for item in items], dim=0)
        fields[f"{prefix}.obs.{key}"] = value

    task_descriptions = np.asarray([item.get("task", "") for item in items], dtype=object)
    return fields, task_descriptions


def rlpd_transition_samples_to_actor_input(
    samples: list[dict],
    dataset,
    rlpd_config,
    global_steps: int,
) -> DataProto:
    action_chunk_steps = int(rlpd_config.get("action_chunk_steps", 10))
    start_indices = torch.as_tensor([sample["start"] for sample in samples], dtype=torch.long)
    next_indices = torch.as_tensor([sample["next"] for sample in samples], dtype=torch.long)
    terminal_mask = torch.as_tensor([sample["terminal"] for sample in samples], dtype=torch.bool)

    t0_items = [sample["t0_item"] for sample in samples]
    t1_items = [sample["t1_item"] for sample in samples]
    t0_obs_fields, t0_task_descriptions = stack_transition_observations(t0_items, "t0")
    t1_obs_fields, t1_task_descriptions = stack_transition_observations(t1_items, "t1")

    t0_action_chunks = read_lerobot_action_chunks(dataset, start_indices, action_chunk_steps)
    t1_action_chunks = read_lerobot_action_chunks(dataset, next_indices, action_chunk_steps)
    rewards = terminal_mask.to(torch.float32) * float(rlpd_config.get("terminal_reward", 1.0))
    task_ids = torch.as_tensor(
        [item["task_index"].item() if torch.is_tensor(item["task_index"]) else item["task_index"] for item in t0_items],
        dtype=torch.long,
    )

    return DataProto.from_dict(
        tensors={
            **t0_obs_fields,
            **t1_obs_fields,
            "t0.action.action": t0_action_chunks,
            "t1.action.action": t1_action_chunks,
            "info.rewards": rewards,
            "info.dones": terminal_mask.to(torch.float32),
            "info.valids": torch.ones(len(samples), dtype=torch.float32),
            "info.positive_sample_mask": torch.ones(len(samples), dtype=torch.float32),
            "info.task_ids": task_ids,
        },
        non_tensors={
            "t0.obs.task_descriptions": t0_task_descriptions,
            "t1.obs.task_descriptions": t1_task_descriptions,
        },
        meta_info={"global_steps": global_steps, "global_token_num": [0]},
    )
