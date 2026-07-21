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

"""Piper-to-LeRobot frame conversion helpers."""

from __future__ import annotations

from verl_vla.recorder.strategies.arena import ArenaLeRobotStrategy


class PiperLeRobotStrategy(ArenaLeRobotStrategy):
    """LeRobot recording strategy for real Piper environments."""

    def __init__(
        self,
        *,
        camera_names: tuple[str, ...],
        image_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        fps: int,
        robot_type: str | None = "piper",
    ) -> None:
        super().__init__(
            camera_names=camera_names,
            image_shape=image_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            fps=fps,
            robot_type=robot_type,
        )
