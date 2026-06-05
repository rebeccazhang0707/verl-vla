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

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from verl_vla.teleop.devices import DeviceBase, DeviceEvent, KeyboardDevice

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from verl_vla.teleop.obs_server.teleop_server import ObsStore


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VERL-VLA Teleop Obs</title>
  <style>
    body { margin: 0; background: #111; color: #f4f4f4; font-family: Arial, sans-serif; }
    header { display: flex; gap: 18px; align-items: center; padding: 10px 14px; background: #202020; }
    main { display: grid; grid-template-columns: 1fr 1fr 320px; gap: 10px; padding: 10px; }
    section { min-width: 0; }
    #image-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-content: start; }
    img { width: 100%; background: #050505; object-fit: contain; border: 1px solid #333; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font-size: 12px; }
    .label { color: #aaa; font-size: 12px; margin-bottom: 6px; }
    .panel { display: grid; gap: 12px; align-content: start; }
    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 4px 0;
      border-bottom: 1px solid #2a2a2a;
      font-size: 13px;
    }
    .status-row span:first-child { color: #aaa; }
    .status-row span:last-child { text-align: right; overflow-wrap: anywhere; }
    @media (max-width: 1300px) { #image-grid { grid-template-columns: 1fr; } }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>VERL-VLA Teleop Obs</strong>
    <span>env: <span id="env-id">-</span></span>
    <span>step: <span id="step">-</span></span>
    <span>fps: <span id="fps">-</span></span>
    <span>ws: <span id="ws-status">connecting</span></span>
    <span>input: <span id="input-status">connecting</span></span>
    <span>intervention: <span id="intervention-status">off</span></span>
  </header>
  <main>
    <section id="image-grid"></section>
    <section>
      <div class="label">State</div>
      <pre id="state">{}</pre>
    </section>
    <section class="panel">
      <div>
        <div class="label">Teleop</div>
        <div class="status-row"><span>device</span><span id="teleop-device">-</span></div>
        <div class="status-row"><span>strategy</span><span id="teleop-strategy">-</span></div>
        <div class="status-row"><span>pressed</span><span id="pressed-keys">-</span></div>
        <div class="status-row"><span>active</span><span id="teleop-active">false</span></div>
        <div class="status-row"><span>gripper</span><span id="teleop-gripper">neutral</span></div>
      </div>
      <div>
        <div class="label">Command</div>
        <pre id="teleop-command">[]</pre>
      </div>
      <div>
        <div class="label">Key Bindings</div>
        <pre id="key-bindings">{}</pre>
      </div>
    </section>
  </main>
  <script>
    const TELEOP_DEVICE_TYPES = __TELEOP_DEVICE_TYPES__;
    let frames = 0;
    let lastFpsAt = performance.now();
    let inputSocket = null;

    function renderObs(obs) {
      document.getElementById("env-id").textContent = obs.env_id ?? "-";
      document.getElementById("step").textContent = obs.step ?? "-";
      document.getElementById("state").textContent = JSON.stringify({
        task_description: obs.task_description,
        state: obs.state,
        extra: obs.extra,
        timestamp: obs.timestamp
      }, null, 2);
      const imageGrid = document.getElementById("image-grid");
      const images = obs.images || {};
      for (const [name, data] of Object.entries(images)) {
        let image = document.getElementById(`image-${name}`);
        if (!image) {
          const wrapper = document.createElement("div");
          const label = document.createElement("div");
          image = document.createElement("img");
          label.className = "label";
          label.textContent = name;
          image.id = `image-${name}`;
          wrapper.appendChild(label);
          wrapper.appendChild(image);
          imageGrid.appendChild(wrapper);
        }
        image.src = "data:image/jpeg;base64," + data;
      }
      frames += 1;
      const now = performance.now();
      if (now - lastFpsAt > 1000) {
        document.getElementById("fps").textContent = frames.toString();
        frames = 0;
        lastFpsAt = now;
      }
    }

    function connectObsStream() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws/obs`);
      const status = document.getElementById("ws-status");

      socket.onopen = () => {
        status.textContent = "connected";
      };
      socket.onmessage = (event) => {
        renderObs(JSON.parse(event.data));
      };
      socket.onerror = () => {
        status.textContent = "error";
      };
      socket.onclose = () => {
        status.textContent = "reconnecting";
        setTimeout(connectObsStream, 1000);
      };
    }

    function connectInputStream() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      inputSocket = new WebSocket(`${protocol}://${window.location.host}/ws/input`);
      const status = document.getElementById("input-status");

      inputSocket.onopen = () => {
        status.textContent = "connected";
        fetch("/api/input/latest", {cache: "no-store"})
          .then((response) => response.json())
          .then(renderInput)
          .catch(() => {});
      };
      inputSocket.onmessage = (event) => {
        renderInput(JSON.parse(event.data));
      };
      inputSocket.onerror = () => {
        status.textContent = "error";
      };
      inputSocket.onclose = () => {
        status.textContent = "reconnecting";
        setTimeout(connectInputStream, 1000);
      };
    }

    function sendKeyboardEvent(event) {
      if (!inputSocket || inputSocket.readyState !== WebSocket.OPEN) {
        return;
      }
      inputSocket.send(JSON.stringify({
        type: "keyboard_event",
        device: "keyboard",
        payload: {
          event_type: event.type,
          key: event.key,
          code: event.code,
          repeat: event.repeat,
          timestamp: Date.now() / 1000
        }
      }));
    }

    function renderInput(input) {
      const isActive = Boolean(input.active || input.is_intervening);
      document.getElementById("intervention-status").textContent = isActive ? "on" : "off";
      document.getElementById("teleop-device").textContent = input.device ?? "-";
      document.getElementById("teleop-strategy").textContent = input.strategy ?? "-";
      document.getElementById("pressed-keys").textContent = (input.pressed_keys || []).join(", ") || "-";
      document.getElementById("teleop-active").textContent = isActive ? "true" : "false";
      const gripperState = input.gripper_active
        ? (input.close_gripper ? "close" : "open")
        : "neutral";
      document.getElementById("teleop-gripper").textContent = gripperState;
      document.getElementById("teleop-command").textContent = JSON.stringify(input.command || [], null, 2);
      document.getElementById("key-bindings").textContent = JSON.stringify(input.key_bindings || {}, null, 2);
    }

    if (TELEOP_DEVICE_TYPES.includes("keyboard")) {
      window.addEventListener("keydown", sendKeyboardEvent);
      window.addEventListener("keyup", sendKeyboardEvent);
    }

    connectObsStream();
    connectInputStream();
  </script>
</body>
</html>
"""


def create_app(
    store: "ObsStore",
    input_devices: dict[str, DeviceBase],
    latest_input_fn=None,
) -> FastAPI:
    app = FastAPI(title=f"VERL-VLA Teleop Obs env {store.env_id}")

    @app.get("/", response_class=HTMLResponse)
    def index():
        device_types = [device.name for device in input_devices.values()]
        return INDEX_HTML.replace("__TELEOP_DEVICE_TYPES__", json.dumps(device_types))

    @app.get("/api/obs/latest")
    def latest_obs():
        return store.latest()

    @app.get("/api/health")
    def health():
        return {"status": "ok", "env_id": store.env_id, "port": store.port}

    @app.get("/api/input/latest")
    def latest_input():
        if latest_input_fn is not None:
            return latest_input_fn()
        return {device_type: device.snapshot() for device_type, device in input_devices.items()}

    @app.get("/api/input/drain")
    def drain_input():
        return {
            "latest": {device_type: device.snapshot() for device_type, device in input_devices.items()},
            "events": {device_type: device.drain_events() for device_type, device in input_devices.items()},
        }

    @app.post("/api/input/reset")
    def reset_input():
        for input_device in input_devices.values():
            input_device.reset()
        return {device_type: device.snapshot() for device_type, device in input_devices.items()}

    @app.websocket("/ws/obs")
    async def obs_stream(websocket: WebSocket):
        await websocket.accept()
        subscriber = store.subscribe()
        try:
            while True:
                payload = await asyncio.to_thread(subscriber.get)
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            store.unsubscribe(subscriber)

    @app.websocket("/ws/input")
    async def input_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")
                device_type = str(message.get("device") or "")
                if device_type not in input_devices:
                    continue
                if device_type == "keyboard" and message_type != "keyboard_event":
                    continue
                input_devices[device_type].handle_event(DeviceEvent.from_payload(message.get("payload", {})))
                if latest_input_fn is not None:
                    await websocket.send_json(latest_input_fn())
                else:
                    await websocket.send_json(input_devices[device_type].snapshot())
        except WebSocketDisconnect:
            pass

    return app


@dataclass
class TeleopObsServer:
    store: "ObsStore"
    host: str
    port: int
    input_devices: dict[str, DeviceBase] | None = None
    log_level: str = "warning"
    latest_input_fn: Callable[[], dict] | None = None

    def __post_init__(self):
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        if self.input_devices is None:
            self.input_devices = {"keyboard": KeyboardDevice()}
        self.input_devices = {device.name: device for device in self.input_devices.values()}

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        config = uvicorn.Config(
            create_app(
                self.store,
                self.input_devices,
                latest_input_fn=self.latest_input,
            ),
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name=f"teleop-obs-{self.port}", daemon=True)
        self._thread.start()
        logger.info("Started teleop obs server for env %s at %s", self.store.env_id, self.url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def latest_input(self) -> dict:
        if self.latest_input_fn is not None:
            return self.latest_input_fn()
        return {device_type: device.snapshot() for device_type, device in self.input_devices.items()}
