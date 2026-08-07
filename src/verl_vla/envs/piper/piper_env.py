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

import logging
import threading
import time
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from typing_extensions import override

from verl_vla.envs.base import BaseEnv
from verl_vla.envs.piper.questarm_ros import QuestArmRosBackend

logger = logging.getLogger(__name__)


class _PiperCameraStream:
    def __init__(self, device: str, *, width: int, height: int, fps: int, fourcc: str):
        self.device = str(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fourcc = str(fourcc)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: np.ndarray | None = None
        self._capture: cv2.VideoCapture | None = None
        self._thread = threading.Thread(target=self._loop, name=f"piper-camera-{self.device}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def read_latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        if self._capture is not None:
            self._capture.release()
        self._thread.join(timeout=1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._capture is None or not self._capture.isOpened():
                self._capture = self._open_capture()
                if self._capture is None:
                    time.sleep(0.5)
                    continue

            ok, frame = self._capture.read()
            if ok and frame is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frame = image
            else:
                time.sleep(0.02)

    def _open_capture(self) -> cv2.VideoCapture | None:
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if capture.isOpened():
            return capture
        capture.release()
        logger.warning("Failed to open Piper camera: %s", self.device)
        return None


class _PiperCameraSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self._streams: list[_PiperCameraStream] = []
        self._last_images = self._blank_images()

    def open(self) -> None:
        self._streams = []
        for device in self.cfg.camera_devices:
            camera = _PiperCameraStream(
                str(device),
                width=int(self.cfg.image_width),
                height=int(self.cfg.image_height),
                fps=int(self.cfg.camera_fps),
                fourcc=str(self.cfg.camera_fourcc),
            )
            camera.start()
            self._streams.append(camera)
            time.sleep(0.25)
        self._wait_for_frames(timeout=6.0)

    def close(self) -> None:
        for camera in self._streams:
            camera.close()
        self._streams = []

    def read_images(self) -> dict[str, np.ndarray]:
        images = {}
        for name, camera in zip(self.cfg.camera_names, self._streams, strict=False):
            camera_name = str(name)
            image = camera.read_latest()
            images[camera_name] = image if image is not None else self._last_images[camera_name].copy()
        self._last_images = images
        return images

    def _wait_for_frames(self, timeout: float) -> None:
        deadline = time.time() + timeout
        pending = set(str(name) for name in self.cfg.camera_names)
        while pending and time.time() < deadline:
            for name, camera in zip(self.cfg.camera_names, self._streams, strict=False):
                camera_name = str(name)
                if camera_name in pending and camera.read_latest() is not None:
                    pending.discard(camera_name)
            time.sleep(0.05)
        for camera_name in sorted(pending):
            logger.warning("Piper camera %s did not produce an initial frame", camera_name)

    def _blank_images(self) -> dict[str, np.ndarray]:
        shape = (int(self.cfg.image_height), int(self.cfg.image_width), 3)
        return {str(name): np.zeros(shape, dtype=np.uint8) for name in self.cfg.camera_names}


class PiperEnv(BaseEnv):
    """Dual Piper X environment controlled exclusively through QuestArm ROS."""

    env_type = "piper"

    def __init__(
        self,
        cfg,
        rank: int,
        world_size: int,
        stage_id: int = 0,
        stage_num: int = 1,
        only_eval: bool = False,
    ) -> None:
        del stage_num, only_eval
        self.piper_cfg = cfg.simulator.piper
        if int(cfg.num_envs) != 1:
            raise ValueError(f"PiperEnv only supports num_envs=1, got {cfg.num_envs}")
        if int(world_size) != 1:
            raise ValueError(f"PiperEnv requires one EnvWorker to own the ROS/CAN lifecycle, got {world_size}")

        self.action_dim = int(self.piper_cfg.action_dim)
        self.state_dim = int(self.piper_cfg.state_dim)
        self.task_description = str(self.piper_cfg.task_description)
        self.task_descriptions = [self.task_description]
        self._backend = QuestArmRosBackend(self.piper_cfg)
        self._cameras = _PiperCameraSystem(self.piper_cfg)
        self._step_id = 0
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "observation.state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_dim,),
                    dtype=np.float32,
                )
            }
        )
        super().__init__(cfg, rank, world_size, stage_id=stage_id)

    @override
    def env_init(self) -> None:
        self._backend.start()
        self._cameras.open()

    @override
    def env_reset(self, *, env_ids, reset_eval: bool = False):
        del reset_eval
        self._validate_env_ids(env_ids)
        self._step_id = 0
        self._backend.reset()
        return self._step_result(
            reward=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            success=np.zeros(1, dtype=bool),
        )

    @override
    def env_step(self, action, *, env_ids):
        self._validate_env_ids(env_ids)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1, self.action_dim):
            raise ValueError(f"Piper action must have shape [1, {self.action_dim}], got {action.shape}")
        self._backend.apply_action(action[0])
        self._step_id += 1
        return self._step_result(
            reward=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            success=np.zeros(1, dtype=bool),
        )

    @override
    def env_close(self) -> None:
        self._backend.close()
        self._cameras.close()

    @override
    def get_teleop_strategy_kwargs(self, device_type: str) -> dict[str, Any]:
        return {"frame_sink": self._backend.accept_webxr_frame} if device_type == "xr_controller" else {}

    @override
    def get_recorder_strategy_kwargs(self) -> dict[str, Any]:
        return {
            "camera_names": tuple(str(name) for name in self.piper_cfg.camera_names),
            "image_shape": (
                int(self.piper_cfg.image_height),
                int(self.piper_cfg.image_width),
                3,
            ),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "fps": int(self.cfg.recorder.video.fps),
            "robot_type": "piper",
        }

    def _validate_env_ids(self, env_ids) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if len(env_ids) != 1 or int(env_ids[0]) != 0:
            raise ValueError(f"PiperEnv only supports env_id 0, got {env_ids.tolist()}")

    def _step_result(self, *, reward, terminated, truncated, success) -> dict[str, Any]:
        return {
            "observation": [self._observation()],
            "task": [self.task_description],
            "task_id": np.zeros(1, dtype=np.int64),
            "next.reward": reward,
            "next.terminated": terminated,
            "next.truncated": truncated,
            "next.success": success,
        }

    def _observation(self) -> dict[str, np.ndarray]:
        obs = {"observation.state": self._backend.read_state()}
        for name, image in self._cameras.read_images().items():
            obs[f"observation.images.{name}"] = image
        return obs
