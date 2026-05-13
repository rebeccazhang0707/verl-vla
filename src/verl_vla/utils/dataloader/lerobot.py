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
from torch.utils.data import SequentialSampler, Subset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm


def create_lerobot_dataset(
    repo_id: str,
    root: str | None = None,
    episodes: list[int] | None = None,
    revision: str | None = None,
    video_backend: str | None = None,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        revision=revision,
        video_backend=video_backend,
    )


def iter_lerobot_episode_indices(dataset, max_episodes: int = 0) -> list[np.ndarray]:
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    unique_episode_indices = np.unique(episode_indices)
    if max_episodes > 0:
        unique_episode_indices = unique_episode_indices[:max_episodes]
    return [
        sorted_contiguous_lerobot_episode_indices(dataset, episode_indices, episode_id)
        for episode_id in unique_episode_indices
    ]


def sorted_contiguous_lerobot_episode_indices(dataset, episode_indices: np.ndarray, episode_id: int) -> np.ndarray:
    indices = np.nonzero(episode_indices == int(episode_id))[0]
    if indices.size == 0:
        return indices

    frame_key = "frame_index" if "frame_index" in dataset.hf_dataset.column_names else "index"
    frames = np.asarray(dataset.hf_dataset.select(indices.tolist())[frame_key], dtype=np.int64).reshape(-1)
    order = np.argsort(frames, kind="stable")
    indices = indices[order]
    frames = frames[order]

    expected = np.arange(frames[0], frames[0] + len(frames))
    if not np.array_equal(frames, expected):
        raise ValueError(
            f"Episode {episode_id} is not contiguous by {frame_key}: first={frames[:10]}, last={frames[-10:]}"
        )
    return indices


def concat_lerobot_batches(batches: list[dict]) -> dict:
    merged = {}
    for key in batches[0].keys():
        values = [batch[key] for batch in batches if key in batch]
        first = values[0]
        if torch.is_tensor(first):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first, np.ndarray):
            merged[key] = np.concatenate(values, axis=0)
        elif isinstance(first, list):
            merged[key] = [item for value in values for item in value]
        else:
            merged[key] = values
    return merged


def load_lerobot_episode(
    dataset,
    indices: np.ndarray,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> dict:
    dataloader = StatefulDataLoader(
        dataset=Subset(dataset, indices.tolist()),
        batch_size=len(indices),
        sampler=SequentialSampler(range(len(indices))),
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return concat_lerobot_batches(list(dataloader))


def iter_lerobot_episodes(
    repo_id: str,
    root: str | None = None,
    episodes: list[int] | None = None,
    revision: str | None = None,
    video_backend: str | None = None,
    max_episodes: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
    desc: str = "LeRobot episodes",
):
    dataset = create_lerobot_dataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        revision=revision,
        video_backend=video_backend,
    )
    for indices in tqdm(iter_lerobot_episode_indices(dataset, max_episodes=max_episodes), desc=desc):
        if len(indices) <= 1:
            continue
        yield load_lerobot_episode(
            dataset,
            indices,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
