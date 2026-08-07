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

"""QuestArm ROS lifecycle and transport owned by :class:`PiperEnv`."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_HOST = "127.0.0.1"
_COMMAND_PORT = 19001
_FEEDBACK_PORT = 19002
_START_TIMEOUT_S = 30.0
_ARM_STATE_SIZE = 14
_JOINT_COUNT = 6


class QuestArmRosBackend:
    """Run the native QuestArm ROS chain and expose it as a Piper backend.

    ROS runs in its own configured Conda environment.  PiperEnv communicates
    with the ROS bridge over two local UDP sockets so the Ray worker does not
    need to import a Python-version-specific ``rclpy`` build.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._ros_conda_sh = self._resolve_conda_sh(str(cfg.ros_conda_sh))
        self._questarm_setup_path = self._resolve_questarm_setup(str(cfg.questarm_setup_path))
        self._command_socket: socket.socket | None = None
        self._feedback_socket: socket.socket | None = None
        self._latest_state = np.zeros(int(cfg.state_dim), dtype=np.float32)
        self._initial_joint_angles: np.ndarray | None = None
        self._process: subprocess.Popen[str] | None = None
        self._closed = False

    def start(self) -> None:
        try:
            self._validate_setup()
            self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._feedback_socket.bind((_HOST, _FEEDBACK_PORT))
            self._feedback_socket.setblocking(False)
            command = self._launch_command()
            self._process = subprocess.Popen(
                ["bash", "-c", command],
                start_new_session=True,
            )

            deadline = time.monotonic() + _START_TIMEOUT_S
            while not self._receive_feedback():
                if self._process.poll() is not None:
                    raise RuntimeError(f"QuestArm ROS launch exited with code {self._process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for QuestArm ROS feedback")
                time.sleep(0.1)
            self._initial_joint_angles = self._resolve_initial_joint_angles()
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        if self._initial_joint_angles is None:
            raise RuntimeError("QuestArm ROS runtime has not captured its initial joint pose")
        self._send(
            {
                "type": "reset",
                "joint_positions": self._initial_joint_angles.astype(float).tolist(),
                "duration_s": float(self.cfg.reset_duration_s),
            }
        )
        deadline = time.monotonic() + float(self.cfg.reset_timeout_s)
        tolerance = float(self.cfg.reset_joint_tolerance)
        while True:
            self._receive_feedback()
            current_joint_angles = self._joint_angles_from_state(self._latest_state)
            if np.allclose(current_joint_angles, self._initial_joint_angles, rtol=0.0, atol=tolerance):
                return
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"QuestArm ROS launch exited with code {self._process.returncode} during reset")
            if time.monotonic() >= deadline:
                joint_errors = np.abs(current_joint_angles - self._initial_joint_angles)
                logger.warning(
                    "Piper reset did not reach the initial joint pose within %.1fs; maximum joint error is %.4frad",
                    float(self.cfg.reset_timeout_s),
                    float(joint_errors.max()),
                )
                return
            time.sleep(0.05)

    def apply_action(self, action: np.ndarray) -> None:
        if np.any(action != 0.0):
            self._send({"type": "action", "action": action.astype(float).tolist()})

    def accept_webxr_frame(self, frame: dict[str, Any]) -> None:
        self._send({"type": "xr_frame", "frame": frame})

    def read_state(self) -> np.ndarray:
        self._receive_feedback()
        return self._latest_state.copy()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._send({"type": "deactivate"})
        except (OSError, RuntimeError):
            pass
        self._closed = True
        if self._feedback_socket is not None:
            self._feedback_socket.close()
            self._feedback_socket = None
        if self._command_socket is not None:
            self._command_socket.close()
            self._command_socket = None
        self._stop_process()

    def _send(self, payload: dict[str, Any]) -> None:
        if self._command_socket is None:
            raise RuntimeError("QuestArm ROS runtime is not started")
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._command_socket.sendto(data, (_HOST, _COMMAND_PORT))

    def _receive_feedback(self) -> bool:
        if self._feedback_socket is None:
            return False
        received = False
        while True:
            try:
                data, _ = self._feedback_socket.recvfrom(1 << 20)
            except BlockingIOError:
                return received
            except OSError:
                return received
            try:
                payload = json.loads(data.decode("utf-8"))
                state = np.asarray(payload["state"], dtype=np.float32)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if state.shape != (int(self.cfg.state_dim),):
                continue
            self._latest_state = state
            received = True

    @staticmethod
    def _joint_angles_from_state(state: np.ndarray) -> np.ndarray:
        return np.stack(
            [state[arm_index * _ARM_STATE_SIZE : arm_index * _ARM_STATE_SIZE + _JOINT_COUNT] for arm_index in range(2)]
        )

    def _resolve_initial_joint_angles(self) -> np.ndarray:
        configured_initial_pose = self.cfg.initial_joint_angles
        if configured_initial_pose is not None:
            return np.asarray(configured_initial_pose, dtype=np.float32)
        return self._joint_angles_from_state(self._latest_state).copy()

    def _launch_command(self) -> str:
        launch_args = {
            "--left-can-port": str(self.cfg.can_channels[0]),
            "--right-can-port": str(self.cfg.can_channels[1]),
            "--position-scale": str(float(self.cfg.position_scale)),
            "--rotation-scale": str(float(self.cfg.rotation_scale)),
            "--ik-position-weight": str(float(self.cfg.ik_position_weight)),
            "--ik-smooth-weight": str(float(self.cfg.ik_smooth_weight)),
            "--gripper-max-range": str(float(self.cfg.gripper_open_width)),
            "--gripper-min-range": str(float(self.cfg.gripper_close_width)),
            "--gripper-width-step": str(float(self.cfg.gripper_width_step)),
            "--gripper-force": str(float(self.cfg.gripper_force)),
        }
        launch_suffix = " ".join(f"{key} {shlex.quote(value)}" for key, value in launch_args.items())
        launcher = Path(__file__).with_name("questarm_launch.py")
        lines = [
            f"source {shlex.quote(str(self._ros_conda_sh))}",
            f"conda activate {shlex.quote(str(self.cfg.ros_conda_env))}",
            f"source {shlex.quote(str(self._questarm_setup_path))}",
            "parent_pid=$PPID",
            "launcher_pid=$$",
            (
                "setsid bash -c 'parent_pid=$1; launcher_pid=$2; process_group=$3; "
                'while kill -0 "$parent_pid" 2>/dev/null && kill -0 "$launcher_pid" 2>/dev/null; '
                "do sleep 0.5; done; "
                'if kill -0 "$launcher_pid" 2>/dev/null; then '
                'kill -INT -- "-$process_group" 2>/dev/null || true; sleep 5; '
                'kill -TERM -- "-$process_group" 2>/dev/null || true; sleep 2; '
                'kill -KILL -- "-$process_group" 2>/dev/null || true; fi\' '
                '_ "$parent_pid" "$launcher_pid" "$launcher_pid" >/dev/null 2>&1 &'
            ),
            f"exec python {shlex.quote(str(launcher))} {launch_suffix}",
        ]
        return "\n".join(lines)

    def _validate_setup(self) -> None:
        for name, path in (
            ("ros_conda_sh", self._ros_conda_sh),
            ("questarm_setup_path", self._questarm_setup_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Piper ROS backend {name} does not exist: {path}")

    @staticmethod
    def _resolve_conda_sh(configured_path: str) -> Path:
        if configured_path:
            return Path(configured_path).expanduser()
        candidates = (
            Path.home() / "miniconda3/etc/profile.d/conda.sh",
            Path.home() / "anaconda3/etc/profile.d/conda.sh",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    @staticmethod
    def _resolve_questarm_setup(configured_path: str) -> Path:
        if configured_path:
            return Path(configured_path).expanduser()
        data_root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        sibling_workspace = Path(__file__).resolve().parents[4].parent / "QuestArmTeleop"
        candidates = (
            data_root / "verl-vla/QuestArmTeleop/install/setup.bash",
            sibling_workspace / "install-ninja2/setup.bash",
            sibling_workspace / "install/setup.bash",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5.0)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            logger.warning("QuestArm ROS process required SIGKILL (pid=%s)", process.pid)
