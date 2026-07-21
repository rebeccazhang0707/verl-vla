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

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.utils import instantiate

from verl_vla.recorder.dataset import read_lerobot_episode_columns
from verl_vla.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized


@dataclass(frozen=True)
class ReplayConfig:
    root: str
    episode_indices: tuple[int, ...]
    extra_columns: tuple[str, ...]
    speed: float
    loop: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", str(self.root))
        object.__setattr__(self, "episode_indices", tuple(int(index) for index in self.episode_indices))
        object.__setattr__(self, "extra_columns", tuple(str(column) for column in self.extra_columns))
        object.__setattr__(self, "speed", float(self.speed))
        object.__setattr__(self, "loop", bool(self.loop))
        if not self.episode_indices:
            raise ValueError("episode_indices must contain at least one episode index.")
        if any(index < 0 for index in self.episode_indices):
            raise ValueError(f"episode_indices must be non-negative, got {self.episode_indices}.")
        if self.speed <= 0:
            raise ValueError(f"speed must be positive, got {self.speed}.")

    @classmethod
    def from_workflow_config(cls, config) -> ReplayConfig:
        return cls(
            root=config.root,
            episode_indices=config.episode_indices,
            extra_columns=config.extra_columns,
            speed=config.speed,
            loop=config.loop,
        )


@dataclass(frozen=True)
class ReplayEpisode:
    episode_index: int
    fps: float
    task: str
    actions: np.ndarray
    states: np.ndarray | None
    extra: dict[str, np.ndarray]


def run_replay(config) -> None:
    replay_config = ReplayConfig.from_workflow_config(config)
    episodes = [
        load_lerobot_episode(replay_config.root, index, extra_columns=replay_config.extra_columns)
        for index in replay_config.episode_indices
    ]
    episode_payloads = [_episode_payload(episode, replay_config.speed) for episode in episodes]

    ensure_ray_initialized(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        server_cfg = config.cluster.env.env_worker.teleop.server
        print(f"Replay viewer: https://localhost:{int(server_cfg.base_port)}")
        while True:
            for episode, payload in zip(episodes, episode_payloads, strict=True):
                print(
                    f"Replaying episode {episode.episode_index}: frames={len(episode.actions)} "
                    f"fps={episode.fps:g} speed={replay_config.speed:g}x task={episode.task!r}"
                )
                result = cluster.replay(payload)
                pprint(result)
            if not replay_config.loop:
                break
    finally:
        cluster.shutdown()


def _episode_payload(episode: ReplayEpisode, speed: float) -> dict:
    return {
        "episode_index": episode.episode_index,
        "actions": episode.actions,
        "states": episode.states,
        "fps": episode.fps,
        "speed": speed,
        "extra": episode.extra,
    }


def load_lerobot_episode(
    dataset_root: str | Path,
    episode_index: int,
    *,
    extra_columns: tuple[str, ...] = (),
) -> ReplayEpisode:
    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot info.json not found: {info_path}")
    with open(info_path) as file:
        info = json.load(file)

    total_episodes = int(info["total_episodes"])
    episode_index = int(episode_index)
    if episode_index < 0 or episode_index >= total_episodes:
        raise IndexError(f"Episode {episode_index} is outside dataset range [0, {total_episodes}).")

    required = ["action", "task_index"]
    optional = ["observation.state", *extra_columns]
    episode_columns = read_lerobot_episode_columns(
        root,
        episode_index,
        columns=required,
        optional_columns=optional,
    )
    task_indices = episode_columns["task_index"].astype(np.int64, copy=False)
    if np.any(task_indices != task_indices[0]):
        raise ValueError(f"Episode {episode_index} contains more than one task_index.")
    task = _load_task(root, int(task_indices[0]))
    actions = episode_columns["action"].astype(np.float32, copy=False)
    states = (
        episode_columns["observation.state"].astype(np.float32, copy=False)
        if "observation.state" in episode_columns
        else None
    )
    return ReplayEpisode(
        episode_index=episode_index,
        fps=float(info["fps"]),
        task=task,
        actions=actions,
        states=states,
        extra={column: episode_columns[column] for column in extra_columns if column in episode_columns},
    )


def _load_task(root: Path, task_index: int) -> str:
    task_table = pq.read_table(root / "meta" / "tasks.parquet")
    matches = task_table.filter(pa.compute.equal(task_table["task_index"], task_index))
    if matches.num_rows != 1:
        raise ValueError(f"Expected one task for task_index={task_index}, found {matches.num_rows}.")
    if "task" in matches.column_names:
        return str(matches["task"][0].as_py())
    if "__index_level_0__" in matches.column_names:
        return str(matches["__index_level_0__"][0].as_py())
    raise ValueError(f"Unsupported LeRobot task metadata columns: {matches.column_names}.")
