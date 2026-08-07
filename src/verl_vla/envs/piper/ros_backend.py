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

"""Native ROS runtime owned by :class:`PiperEnv`."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from ctypes import CDLL
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

logger = logging.getLogger(__name__)
_START_TIMEOUT_S = 30.0
_CAMERA_TIMEOUT_S = 6.0
_PR_SET_PDEATHSIG = 1


def _stop_launch_with_parent() -> None:
    """Make Linux stop the ROS launch process if its Piper worker disappears."""
    CDLL(None).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)


def _mat_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()


@dataclass
class _HandState:
    feedback_pose: np.ndarray | None = None
    command_pose: np.ndarray | None = None
    joint_positions: np.ndarray | None = None
    gripper_width: float | None = None
    gripper_force: float = 0.0
    gripper_target: float | None = None


@dataclass
class _ResetTrajectory:
    start_joint_angles: np.ndarray
    target_joint_angles: np.ndarray
    start_time: float
    duration_s: float
    next_publish_time: float = 0.0


class _PiperNode(Node):
    def __init__(self, cfg: Any) -> None:
        super().__init__("verl_vla_piper")
        self.cfg = cfg
        self._lock = threading.RLock()
        self._arm_names = tuple(arm.name for arm in cfg.arms)
        self._hands = {name: _HandState() for name in self._arm_names}
        self._camera_shapes = {camera.name: (int(camera.height), int(camera.width), 3) for camera in cfg.cameras}
        self._images = {name: np.zeros(shape, dtype=np.uint8) for name, shape in self._camera_shapes.items()}
        self._camera_ready: set[str] = set()
        self._reset_trajectory: _ResetTrajectory | None = None

        self._delta_publishers = {
            hand: self.create_publisher(PoseStamped, f"/{hand}_delta_pose", 10) for hand in self._arm_names
        }
        self._joint_publishers = {
            hand: self.create_publisher(JointState, f"/{hand}_arm/control/joint_states", 10) for hand in self._arm_names
        }
        self._reset_publishers = {
            hand: self.create_publisher(JointState, f"/{hand}_arm/control/move_j", 1) for hand in self._arm_names
        }
        for hand in self._arm_names:
            self.create_subscription(
                PoseStamped,
                f"/{hand}_arm/feedback/tcp_pose",
                lambda msg, hand=hand: self._on_tcp_pose(hand, msg),
                1,
            )
            self.create_subscription(
                JointState,
                f"/{hand}_arm/feedback/joint_states",
                lambda msg, hand=hand: self._on_joint_state(hand, msg),
                1,
            )
        for camera in cfg.cameras:
            camera_name = camera.name
            self.create_subscription(
                Image,
                f"/{camera_name}/image_raw",
                lambda msg, camera_name=camera_name: self._on_image(camera_name, msg),
                1,
            )
        self.create_timer(1.0 / 30.0, self._advance_reset)

    def arms_ready(self) -> bool:
        with self._lock:
            return all(
                state.feedback_pose is not None
                and state.joint_positions is not None
                and state.gripper_width is not None
                for state in self._hands.values()
            )

    def pending_cameras(self) -> set[str]:
        with self._lock:
            return set(self._images) - self._camera_ready

    def read_state(self) -> np.ndarray:
        with self._lock:
            parts: list[float] = []
            for hand in self._arm_names:
                state = self._hands[hand]
                if state.joint_positions is None or state.feedback_pose is None or state.gripper_width is None:
                    raise RuntimeError("Piper ROS feedback is not ready")
                parts.extend(state.joint_positions.tolist())
                parts.extend(state.feedback_pose[:3, 3].tolist())
                parts.extend(Rotation.from_matrix(state.feedback_pose[:3, :3]).as_euler("xyz").tolist())
                parts.extend([state.gripper_width, state.gripper_force])
            return np.asarray(parts, dtype=np.float32)

    def read_joint_angles(self) -> np.ndarray:
        with self._lock:
            positions = [self._hands[hand].joint_positions for hand in self._arm_names]
            if any(value is None for value in positions):
                raise RuntimeError("Piper joint feedback is not ready")
            return np.stack(positions)

    def read_images(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {name: image.copy() for name, image in self._images.items()}

    def read_arm_rotations(self) -> dict[str, np.ndarray]:
        with self._lock:
            rotations = {}
            for hand, state in self._hands.items():
                if state.feedback_pose is None:
                    raise RuntimeError("Piper ROS feedback is not ready")
                rotations[hand] = state.feedback_pose[:3, :3].copy()
            return rotations

    def apply_action(self, action: np.ndarray) -> None:
        if self._reset_trajectory is not None:
            return
        commands = np.asarray(action, dtype=float).reshape(len(self._arm_names), 7)
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            for hand, command in zip(self._arm_names, commands, strict=True):
                state = self._hands[hand]
                ee_delta = command[:6]
                if state.feedback_pose is not None and np.any(ee_delta != 0.0):
                    target = state.command_pose.copy() if state.command_pose is not None else state.feedback_pose.copy()
                    target[:3, 3] += ee_delta[:3]
                    target[:3, :3] = Rotation.from_rotvec(ee_delta[3:]).as_matrix() @ target[:3, :3]
                    state.command_pose = target
                    self._publish_target(hand, *_mat_to_pose(target), stamp)
                if command[6] != 0.0 and state.gripper_width is not None:
                    current = state.gripper_target if state.gripper_target is not None else state.gripper_width
                    state.gripper_target = float(
                        np.clip(
                            current + command[6],
                            float(self.cfg.gripper_close_width),
                            float(self.cfg.gripper_open_width),
                        )
                    )
                    self._publish_gripper(hand, state.gripper_target, stamp)

    def start_reset(self, target_joint_angles: np.ndarray, duration_s: float) -> None:
        with self._lock:
            self.deactivate()
            start_joint_angles = self.read_joint_angles()
            if np.allclose(start_joint_angles, target_joint_angles, rtol=0.0, atol=1e-6):
                self._publish_joint_targets(target_joint_angles)
                return
            self._reset_trajectory = _ResetTrajectory(
                start_joint_angles=start_joint_angles,
                target_joint_angles=target_joint_angles,
                start_time=time.monotonic(),
                duration_s=duration_s,
            )

    def deactivate(self) -> None:
        with self._lock:
            self._reset_trajectory = None
            for state in self._hands.values():
                self._reset_hand_reference(state)

    def sync_command_poses(self) -> None:
        with self._lock:
            for state in self._hands.values():
                state.command_pose = None if state.feedback_pose is None else state.feedback_pose.copy()

    def _on_tcp_pose(self, hand: str, msg: PoseStamped) -> None:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        ).as_matrix()
        matrix[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        with self._lock:
            state = self._hands[hand]
            state.feedback_pose = matrix
            if state.command_pose is None:
                state.command_pose = matrix.copy()

    def _on_joint_state(self, hand: str, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position, strict=False))
        joints = [positions.get(f"joint{index}") for index in range(1, 7)]
        if any(value is None for value in joints):
            return
        with self._lock:
            state = self._hands[hand]
            state.joint_positions = np.asarray(joints, dtype=float)
            if "gripper" in positions:
                index = list(msg.name).index("gripper")
                state.gripper_width = float(positions["gripper"])
                state.gripper_force = float(msg.effort[index]) if len(msg.effort) > index else 0.0
                if state.gripper_target is None:
                    state.gripper_target = state.gripper_width

    def _on_image(self, camera_name: str, msg: Image) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().error(f"Camera {camera_name} published unsupported encoding {msg.encoding!r}", once=True)
            return
        row_bytes = int(msg.width) * 3
        expected_height, expected_width, _ = self._camera_shapes[camera_name]
        if int(msg.height) != expected_height or int(msg.width) != expected_width:
            self.get_logger().error(
                f"Camera {camera_name} image is {msg.width}x{msg.height}, expected {expected_width}x{expected_height}",
                once=True,
            )
            return
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.step))[:, :row_bytes]
        image = image.reshape(int(msg.height), int(msg.width), 3).copy()
        with self._lock:
            self._images[camera_name] = image
            self._camera_ready.add(camera_name)

    def _advance_reset(self) -> None:
        with self._lock:
            trajectory = self._reset_trajectory
            if trajectory is None:
                return
            now = time.monotonic()
            if now < trajectory.next_publish_time:
                return
            trajectory.next_publish_time = now + 1.0 / 30.0
            progress = min((now - trajectory.start_time) / trajectory.duration_s, 1.0)
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            positions = trajectory.start_joint_angles + smooth_progress * (
                trajectory.target_joint_angles - trajectory.start_joint_angles
            )
            self._publish_joint_targets(positions)
            if progress >= 1.0:
                self._reset_trajectory = None

    def _publish_joint_targets(self, joint_angles: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        for hand, positions in zip(self._arm_names, joint_angles, strict=True):
            target = JointState(header=Header(stamp=stamp))
            target.name = [f"joint{index}" for index in range(1, 7)]
            target.position = positions.tolist()
            self._reset_publishers[hand].publish(target)

    def _publish_gripper(self, hand: str, width: float, stamp: Any) -> None:
        target = JointState(header=Header(stamp=stamp))
        target.name = ["gripper"]
        target.position = [float(width)]
        target.effort = [float(self.cfg.gripper_force)]
        self._joint_publishers[hand].publish(target)

    def _publish_target(self, hand: str, xyz: np.ndarray, quat: np.ndarray, stamp: Any) -> None:
        target = PoseStamped(header=Header(stamp=stamp, frame_id="vr_device"))
        target.pose.position.x, target.pose.position.y, target.pose.position.z = [float(value) for value in xyz]
        target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = [
            float(value) for value in quat
        ]
        self._delta_publishers[hand].publish(target)

    @staticmethod
    def _reset_hand_reference(state: _HandState) -> None:
        if state.feedback_pose is not None:
            state.command_pose = state.feedback_pose.copy()
        if state.gripper_width is not None:
            state.gripper_target = state.gripper_width


class PiperRosBackend:
    """Own the ROS executor and upstream QuestArm processes in the Piper worker."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._node: _PiperNode | None = None
        self._executor: MultiThreadedExecutor | None = None
        self._executor_thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._initial_joint_angles: np.ndarray | None = None

    def start(self) -> None:
        try:
            rclpy.init()
            self._node = _PiperNode(self.cfg)
            self._executor = MultiThreadedExecutor(num_threads=2)
            self._executor.add_node(self._node)
            self._process = subprocess.Popen(
                self._launch_command(),
                start_new_session=True,
                text=True,
                preexec_fn=_stop_launch_with_parent,
            )
            self._executor_thread = threading.Thread(target=self._spin, name="piper-ros", daemon=True)
            self._executor_thread.start()
            self._wait_for_arms()
            current_pose = self._node.read_joint_angles()
            self._initial_joint_angles = np.stack(
                [
                    current if arm.initial_joint_angles is None else np.asarray(arm.initial_joint_angles, dtype=float)
                    for arm, current in zip(self.cfg.arms, current_pose, strict=True)
                ]
            )
            self._wait_for_cameras()
        except Exception:
            self.close()
            raise

    def reset(self) -> None:
        node = self._require_node()
        if self._initial_joint_angles is None:
            raise RuntimeError("Piper ROS runtime has not captured its initial joint pose")
        node.start_reset(self._initial_joint_angles, float(self.cfg.reset_duration_s))
        deadline = time.monotonic() + float(self.cfg.reset_timeout_s)
        tolerance = float(self.cfg.reset_joint_tolerance)
        while True:
            current = node.read_joint_angles()
            if np.allclose(current, self._initial_joint_angles, rtol=0.0, atol=tolerance):
                node.sync_command_poses()
                return
            self._check_process("during reset")
            if time.monotonic() >= deadline:
                error = np.abs(current - self._initial_joint_angles).max()
                logger.warning(
                    "Piper reset did not reach its target within %.1fs; maximum joint error is %.4frad",
                    float(self.cfg.reset_timeout_s),
                    float(error),
                )
                node.sync_command_poses()
                return
            time.sleep(0.05)

    def apply_action(self, action: np.ndarray) -> None:
        if np.any(action != 0.0):
            self._require_node().apply_action(action)

    def read_state(self) -> np.ndarray:
        return self._require_node().read_state()

    def read_images(self) -> dict[str, np.ndarray]:
        return self._require_node().read_images()

    def read_arm_rotations(self) -> dict[str, np.ndarray]:
        return self._require_node().read_arm_rotations()

    def close(self) -> None:
        if self._node is not None:
            self._node.deactivate()
        self._stop_process()
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2.0)
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=2.0)
        if self._node is not None:
            self._node.destroy_node()
        self._node = None
        self._executor = None
        self._executor_thread = None
        if rclpy.ok():
            rclpy.shutdown()

    def _wait_for_arms(self) -> None:
        node = self._require_node()
        deadline = time.monotonic() + _START_TIMEOUT_S
        while not node.arms_ready():
            self._check_process("during startup")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Piper ROS feedback")
            time.sleep(0.1)

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except ExternalShutdownException:
            pass
        except Exception:
            if rclpy.ok():
                raise

    def _wait_for_cameras(self) -> None:
        node = self._require_node()
        deadline = time.monotonic() + _CAMERA_TIMEOUT_S
        while node.pending_cameras() and time.monotonic() < deadline:
            self._check_process("during camera startup")
            time.sleep(0.05)
        for name in sorted(node.pending_cameras()):
            logger.warning("Piper camera %s did not produce an initial ROS image", name)

    def _launch_command(self) -> list[str]:
        command = [
            sys.executable,
            str(Path(__file__).with_name("ros_launch.py")),
            "--ik-position-weight",
            str(float(self.cfg.ik_position_weight)),
            "--ik-smooth-weight",
            str(float(self.cfg.ik_smooth_weight)),
        ]
        for arm in self.cfg.arms:
            command.extend(
                ["--arm", json.dumps({"name": arm.name, "can_channel": arm.can_channel, "model": arm.model})]
            )
        for camera in self.cfg.cameras:
            command.extend(
                [
                    "--camera",
                    json.dumps(
                        {
                            "name": camera.name,
                            "device": camera.device,
                            "width": camera.width,
                            "height": camera.height,
                            "pixel_format": camera.pixel_format,
                        }
                    ),
                ]
            )
        return command

    def _require_node(self) -> _PiperNode:
        if self._node is None:
            raise RuntimeError("Piper ROS runtime is not started")
        return self._node

    def _check_process(self, context: str) -> None:
        if self._process is not None and self._process.poll() is not None:
            raise RuntimeError(f"Piper ROS launch exited with code {self._process.returncode} {context}")

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        for sig, timeout in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0)):
            try:
                os.killpg(process.pid, sig)
                process.wait(timeout=timeout)
                return
            except ProcessLookupError:
                return
            except subprocess.TimeoutExpired:
                continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
