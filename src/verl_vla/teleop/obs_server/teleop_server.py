# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import queue
import struct
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Callable

import cv2
import numpy as np
import torch

from verl_vla.teleop.config import TeleopServerConfig
from verl_vla.teleop.devices import DeviceBase
from verl_vla.teleop.obs_server.server import TeleopObsServer

logger = logging.getLogger(__name__)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_as_jsonable(v) for v in value]
    return value


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")

    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    success, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not success:
        raise RuntimeError("Failed to encode observation image as JPEG")
    return encoded.tobytes()


def _pack_frame(metadata: dict[str, Any], images: dict[str, bytes]) -> bytes:
    header = {
        **metadata,
        "images": [{"name": name, "length": len(jpeg)} for name, jpeg in images.items()],
    }
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return struct.pack("!I", len(header_bytes)) + header_bytes + b"".join(images.values())


def _put_latest(target: queue.Queue, item: Any) -> None:
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    try:
        target.put_nowait(item)
    except queue.Full:
        pass


class ObsStore:
    def __init__(self, env_id: int, port: int, jpeg_quality: int = 80):
        self.env_id = env_id
        self.port = port
        self.jpeg_quality = jpeg_quality
        self._lock = Lock()
        self._latest: bytes | None = None
        self._subscribers: set[queue.Queue[bytes]] = set()

    def update(
        self,
        *,
        step: int,
        images: dict[str, np.ndarray],
        state: Any | None = None,
        extra: dict[str, Any] | None = None,
        task_description: str | None = None,
    ) -> None:
        frame = _pack_frame(
            {
                "env_id": self.env_id,
                "port": self.port,
                "step": int(step),
                "timestamp": time.time(),
                "task_description": task_description,
                "state": state,
                "extra": extra or {},
            },
            {str(name): _encode_jpeg(image, self.jpeg_quality) for name, image in images.items()},
        )
        with self._lock:
            self._latest = frame
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            _put_latest(subscriber, frame)

    def subscribe(self) -> queue.Queue[bytes]:
        subscriber: queue.Queue[bytes] = queue.Queue(maxsize=1)
        with self._lock:
            self._subscribers.add(subscriber)
            latest = self._latest
        if latest is not None:
            _put_latest(subscriber, latest)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)


@dataclass(frozen=True)
class _ObservationFrame:
    step: int
    images: dict[str, np.ndarray]
    state: Any | None
    extra: dict[str, Any]
    task_description: str | None


class TeleopServer:
    def __init__(
        self,
        cfg: TeleopServerConfig,
        *,
        rank: int,
        stage_id: int,
        env_id: int,
        input_devices: dict[str, DeviceBase],
        latest_input_fn: Callable[[], dict[str, Any]],
    ):
        self.cfg = cfg
        self.rank = rank
        self.stage_id = stage_id
        self.env_id = env_id
        self.input_devices = input_devices
        self.latest_input_fn = latest_input_fn
        self._step = 0
        self._store: ObsStore | None = None
        self._server: TeleopObsServer | None = None
        self._pending_frames: queue.Queue[_ObservationFrame] = queue.Queue(maxsize=1)
        self._publisher_stop = Event()
        self._publisher_thread: Thread | None = None

    @classmethod
    def from_cfg(
        cls,
        cfg: TeleopServerConfig,
        *,
        rank: int,
        stage_id: int,
        env_id: int,
        input_devices: dict[str, DeviceBase],
        latest_input_fn: Callable[[], dict[str, Any]],
    ) -> "TeleopServer":
        server = cls(
            cfg,
            rank=rank,
            stage_id=stage_id,
            env_id=env_id,
            input_devices=input_devices,
            latest_input_fn=latest_input_fn,
        )
        server.start()
        return server

    def port(self) -> int:
        return (
            self.cfg.base_port + self.rank * self.cfg.rank_stride + self.stage_id * self.cfg.stage_stride + self.env_id
        )

    def start(self) -> None:
        if self._server is not None:
            return
        port = self.port()
        store = ObsStore(env_id=self.env_id, port=port, jpeg_quality=self.cfg.jpeg_quality)
        server = TeleopObsServer(
            store=store,
            host=self.cfg.host,
            port=port,
            input_devices=self.input_devices,
            log_level=self.cfg.log_level,
            latest_input_fn=self.latest_input_fn,
            ssl_certfile=self.cfg.ssl_certfile,
            ssl_keyfile=self.cfg.ssl_keyfile,
        )
        server.start()
        self._store = store
        self._server = server
        self._publisher_stop.clear()
        self._publisher_thread = Thread(
            target=self._publish_loop,
            name=f"teleop-publisher-{self.rank}-{self.stage_id}-{self.env_id}",
            daemon=True,
        )
        self._publisher_thread.start()
        print(
            f"[teleop] rank={self.rank} stage={self.stage_id} env={self.env_id} obs_url={server.url}",
            flush=True,
        )
        logger.info(
            "Teleop obs server started for rank=%s stage=%s env=%s: %s",
            self.rank,
            self.stage_id,
            self.env_id,
            server.url,
        )

    def publish_obs(
        self,
        *,
        images: dict[str, Any],
        state: Any | None = None,
        extra: dict[str, Any] | None = None,
        task_description: str | None = None,
    ) -> None:
        if self._store is None:
            return
        frame = _ObservationFrame(
            step=self._step,
            images={str(name): np.array(image, copy=True, order="C") for name, image in images.items()},
            state=_as_jsonable(state) if state is not None else None,
            extra=_as_jsonable(extra or {}),
            task_description=task_description,
        )
        self._step += 1
        _put_latest(self._pending_frames, frame)

    def reset(self) -> None:
        self._step = 0

    def _publish_loop(self) -> None:
        min_interval_s = 1.0 / self.cfg.max_fps
        next_publish_at = 0.0
        while not self._publisher_stop.is_set():
            try:
                frame = self._pending_frames.get(timeout=0.1)
            except queue.Empty:
                continue

            wait_s = next_publish_at - time.monotonic()
            if wait_s > 0 and self._publisher_stop.wait(wait_s):
                return

            try:
                while True:
                    frame = self._pending_frames.get_nowait()
            except queue.Empty:
                pass

            store = self._store
            if store is None:
                return
            publish_started_at = time.monotonic()
            try:
                store.update(
                    step=frame.step,
                    images=frame.images,
                    state=frame.state,
                    extra=frame.extra,
                    task_description=frame.task_description,
                )
            except Exception:
                logger.exception("Failed to encode teleop observation frame")
            next_publish_at = publish_started_at + min_interval_s

    def write_console(self, text: str) -> None:
        if self._server is not None:
            self._server.console.write_backend(text)

    def close(self) -> None:
        self._publisher_stop.set()
        if self._publisher_thread is not None:
            self._publisher_thread.join()
        self._publisher_thread = None
        if self._server is not None:
            self._server.stop()
        self._server = None
        self._store = None
