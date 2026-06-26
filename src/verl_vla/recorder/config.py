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

"""Configuration helpers for LeRobot dataset recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LeRobotRecorderConfig:
    enable: bool = True
    root: str = "/tmp/verl_vla_lerobot_records"
    repo_id: str = "local/verl_vla_libero"
    use_videos: bool = True
    image_writer_processes: int = 0
    image_writer_threads: int = 0
    batch_encoding_size: int = 1
    vcodec: str = "libsvtav1"
    video_files_size_in_mb: float = 1e-6
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoRecorderConfig:
    enable: bool = True
    root: str = "/tmp/verl_vla_videos"
    fps: int = 30
    font_size: int = 14


@dataclass(frozen=True)
class RecorderConfig:
    enable: bool = True
    async_enable: bool = False
    async_queue_size: int = 256
    recorders: tuple[str, ...] = ("lerobot", "video")
    lerobot: LeRobotRecorderConfig = field(default_factory=LeRobotRecorderConfig)
    video: VideoRecorderConfig = field(default_factory=VideoRecorderConfig)

    def __post_init__(self):
        if isinstance(self.recorders, str):
            recorders = (self.recorders,)
        else:
            recorders = tuple(str(recorder).strip().lower() for recorder in self.recorders if str(recorder).strip())
        object.__setattr__(self, "recorders", recorders)
        object.__setattr__(self.lerobot, "enable", "lerobot" in recorders)
        object.__setattr__(self.video, "enable", "video" in recorders)
