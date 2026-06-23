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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset

from .config import LeRobotDataLoaderConfig


def build_lerobot_dataset(
    data_config: LeRobotDataLoaderConfig,
    *,
    repo_id: str | None = None,
    root: str | None = None,
    delta_timestamps: dict[str, list[float]] | None = None,
):
    return LeRobotDataset(
        repo_id=repo_id or data_config.repo_id,
        root=root if root is not None else data_config.root,
        revision=data_config.revision,
        video_backend=data_config.video_backend,
        delta_timestamps=delta_timestamps,
    )


def build_lerobot_sft_dataset(data_config: LeRobotDataLoaderConfig):
    action_delta_steps = int(data_config.action_delta_steps)
    delta_timestamps = None
    if action_delta_steps > 0:
        probe_dataset = build_lerobot_dataset(data_config)
        delta_timestamps = {"action": [t / probe_dataset.fps for t in range(action_delta_steps)]}
    return build_lerobot_dataset(data_config, delta_timestamps=delta_timestamps)


class RLPDTransitionDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        action_chunk_steps: int,
        root: str | None = None,
        revision: str | None = None,
        video_backend: str | None = None,
        episodes: list[int] | None = None,
        max_transitions: int = 0,
    ):
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            revision=revision,
            video_backend=video_backend,
        )
        self.records = list(
            iter_lerobot_transition_records(
                self.dataset,
                action_chunk_steps=action_chunk_steps,
                episodes=episodes,
                max_transitions=max_transitions,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        start = int(record["start"])
        next_index = int(record["next"])
        return {
            "start": start,
            "next": next_index,
            "terminal": bool(record["terminal"]),
            "t0_item": self.dataset[start],
            "t1_item": self.dataset[next_index],
        }


def iter_lerobot_transition_records(
    dataset,
    action_chunk_steps: int,
    episodes: list[int] | None = None,
    max_transitions: int = 0,
):
    selected_episodes = set(int(episode) for episode in episodes) if episodes is not None else None
    transition_window = action_chunk_steps * 2
    emitted = 0
    for episode in dataset.meta.episodes:
        episode_index = int(episode["episode_index"])
        if selected_episodes is not None and episode_index not in selected_episodes:
            continue

        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        num_transitions = end - start - transition_window + 1
        if num_transitions <= 0:
            continue

        for transition_offset in range(num_transitions):
            if max_transitions > 0 and emitted >= max_transitions:
                return
            transition_start = start + transition_offset
            yield {
                "start": transition_start,
                "next": transition_start + action_chunk_steps,
                "terminal": transition_start + transition_window == end,
            }
            emitted += 1
