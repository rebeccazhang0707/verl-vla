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

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_vla.utils.data import add_transition_prefixes, flatten_trajectories
from verl_vla.utils.dataloader.lerobot import iter_lerobot_episodes
from verl_vla.utils.image import preprocess_image_batch_to_uint8

LEROBOT_IMAGE_CROP_SIZE = 480
LEROBOT_IMAGE_RESIZE_SIZE = (224, 224)
LEROBOT_METADATA_KEYS = {
    "done",
    "episode_index",
    "frame_index",
    "index",
    "reward",
    "task_index",
    "timestamp",
}


def iter_rlpd_replay_prefill_batches(config, global_steps: int):
    rlpd_config = OmegaConf.select(config, "data.rlpd")
    if not rlpd_config or not rlpd_config.get("enable", False):
        return

    prefill_batches = []

    for episode in iter_lerobot_episodes(
        repo_id=rlpd_config.repo_id,
        root=rlpd_config.get("root"),
        episodes=OmegaConf.to_container(rlpd_config.get("episodes")) if rlpd_config.get("episodes") else None,
        revision=rlpd_config.get("revision"),
        video_backend=rlpd_config.get("video_backend"),
        max_episodes=int(rlpd_config.get("max_episodes", 0)),
        num_workers=int(rlpd_config.get("num_workers", 0)),
        pin_memory=bool(rlpd_config.get("pin_memory", True)),
        desc="RLPD replay prefill",
    ):
        actor_input = rlpd_episode_to_actor_input(episode, rlpd_config, global_steps)

        prefill_batches.append(actor_input)

        max_transitions = int(rlpd_config.get("submit_max_transitions", 0))
        if max_transitions > 0 and sum(len(batch) for batch in prefill_batches) >= max_transitions:
            yield DataProto.concat(prefill_batches)
            prefill_batches = []

    if prefill_batches:
        yield DataProto.concat(prefill_batches)


def patch_lerobot_image_batch(image_batch: torch.Tensor) -> torch.Tensor:
    # This implementation is ugly, but matches the LeRobot env logic that compresses data before returning it.
    return preprocess_image_batch_to_uint8(
        image_batch,
        crop_size=LEROBOT_IMAGE_CROP_SIZE,
        resize_size=LEROBOT_IMAGE_RESIZE_SIZE,
    )


def collect_episode_tensors(episode: dict) -> dict[str, torch.Tensor]:
    fields = {}
    for key, value in episode.items():
        if not torch.is_tensor(value) or key in LEROBOT_METADATA_KEYS:
            continue

        output_key = key
        if key == "action":
            output_key = "action.action"
        elif key.startswith("observation."):
            output_key = f"obs.{key}"

        if key.startswith("observation.images."):
            value = patch_lerobot_image_batch(value)
        fields[output_key] = value.unsqueeze(0)
    return fields


def rlpd_episode_to_actor_input(episode: dict, rlpd_config, global_steps: int) -> DataProto:
    tensor_fields = collect_episode_tensors(episode)
    actions = tensor_fields["action.action"].squeeze(0)
    num_steps = int(actions.shape[0])
    if num_steps <= 1:
        raise ValueError("RLPD episodes must contain at least two frames to form transitions.")

    task_descriptions = episode.get("task", [""] * num_steps)
    if not isinstance(task_descriptions, list):
        task_descriptions = list(task_descriptions)
    task_ids = episode.get("task_index", torch.zeros(num_steps, dtype=torch.long))
    if torch.is_tensor(task_ids):
        task_ids = task_ids.to(torch.long)
    else:
        task_ids = torch.as_tensor(task_ids, dtype=torch.long)

    rewards = torch.zeros(num_steps, dtype=torch.float32)
    if "reward" in episode:
        rewards = (
            episode["reward"].to(torch.float32)
            if torch.is_tensor(episode["reward"])
            else torch.as_tensor(episode["reward"], dtype=torch.float32)
        )
    else:
        rewards[-2] = float(rlpd_config.get("terminal_reward", 1.0))
    rewards[-2] = torch.maximum(
        rewards[-2],
        torch.as_tensor(float(rlpd_config.get("terminal_reward", 1.0)), dtype=rewards.dtype),
    )

    dones = torch.zeros(num_steps, dtype=torch.float32)
    done_key = "done" if "done" in episode else None
    if done_key is not None:
        dones = (
            episode[done_key].to(torch.float32)
            if torch.is_tensor(episode[done_key])
            else torch.as_tensor(episode[done_key], dtype=torch.float32)
        )
    dones[-2] = 1.0

    trajectory = DataProto.from_dict(
        tensors={
            **tensor_fields,
            "info.rewards": rewards.unsqueeze(0),
            "info.dones": dones.unsqueeze(0),
            "info.valids": torch.ones(1, num_steps, dtype=torch.float32),
            "info.positive_sample_mask": torch.ones(1, num_steps, dtype=torch.float32),
            "info.task_ids": task_ids[:1],
        },
        non_tensors={
            "obs.task_descriptions": np.asarray([task_descriptions], dtype=object),
        },
        meta_info={"global_steps": global_steps, "global_token_num": [0]},
    )
    return flatten_trajectories(add_transition_prefixes(trajectory))
