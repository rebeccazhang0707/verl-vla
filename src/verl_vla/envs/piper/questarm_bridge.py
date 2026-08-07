#!/usr/bin/env python3
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

"""Bridge PiperEnv commands to the upstream QuestArm ROS control chain.

This module is launched by :class:`QuestArmRosBackend` inside the configured
ROS Python environment. It deliberately is not imported by the verl-vla worker,
whose Python environment does not need to contain ``rclpy``.
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def xyzrpy_to_mat(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    matrix[:3, 3] = [x, y, z]
    return matrix


def pose_to_mat(pose: dict) -> np.ndarray | None:
    try:
        position = np.asarray(pose["position"], dtype=float)
        quaternion = np.asarray(pose["orientation"], dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,):
            return None
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 1e-9:
            return None
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
        matrix[:3, 3] = position
        return matrix
    except (KeyError, TypeError, ValueError):
        return None


def mat_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()


def calc_pose_incre(
    start_pose_matrix: np.ndarray,
    current_pose_matrix: np.ndarray,
    zero_matrix: np.ndarray,
    position_scale: float = 1.0,
    rotation_scale: float = 1.0,
):
    """Compose a scaled controller-relative transform onto the robot zero pose."""
    relative = np.linalg.inv(start_pose_matrix) @ current_pose_matrix
    relative[:3, 3] *= position_scale
    relative_rotation = Rotation.from_matrix(relative[:3, :3])
    relative[:3, :3] = Rotation.from_rotvec(relative_rotation.as_rotvec() * rotation_scale).as_matrix()
    return mat_to_pose(zero_matrix @ relative)


@dataclass
class HandState:
    feedback_pose: np.ndarray | None = None
    joint_positions: np.ndarray | None = None
    gripper_width: float | None = None
    gripper_force: float = 0.0
    gripper_target: float | None = None
    zero_matrix: np.ndarray | None = None
    start_pose_matrix: np.ndarray | None = None
    active: bool = False
    primary_down: bool = False
    secondary_down: bool = False


@dataclass
class ResetTrajectory:
    start_joint_angles: np.ndarray
    target_joint_angles: np.ndarray
    start_time: float
    duration_s: float
    next_publish_time: float


class QuestArmBridge(Node):
    def __init__(self) -> None:
        super().__init__("questarm_bridge")
        self.declare_parameter("gripper_max_range", 0.1)
        self.declare_parameter("gripper_min_range", 0.0)
        self.declare_parameter("gripper_width_step", 0.005)
        self.declare_parameter("gripper_force", 1.0)
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("rotation_scale", 1.0)
        self._adj_mat = np.array(
            [[0.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        self._r_adj = xyzrpy_to_mat(0.0, 0.0, 0.0, -math.pi, 0.0, -math.pi / 2.0)
        self._ros_to_arm_mat = xyzrpy_to_mat(0.0, 0.0, 0.0, -1.5708, 0.0, -1.5708)
        self._hands = {"left": HandState(), "right": HandState()}
        self._reset_trajectory: ResetTrajectory | None = None
        self._position_scale = float(self.get_parameter("position_scale").value)
        self._rotation_scale = float(self.get_parameter("rotation_scale").value)
        if self._position_scale <= 0.0 or self._rotation_scale <= 0.0:
            raise ValueError("position_scale and rotation_scale must be positive")

        self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._command_socket.bind(("127.0.0.1", 19001))
        self._command_socket.setblocking(False)
        self._feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._feedback_target = ("127.0.0.1", 19002)

        self._delta_publishers = {
            "left": self.create_publisher(PoseStamped, "/left_delta_pose", 10),
            "right": self.create_publisher(PoseStamped, "/right_delta_pose", 10),
        }
        self._joint_publishers = {
            "left": self.create_publisher(JointState, "/left_arm/control/joint_states", 10),
            "right": self.create_publisher(JointState, "/right_arm/control/joint_states", 10),
        }
        self._reset_publishers = {
            "left": self.create_publisher(JointState, "/left_arm/control/move_j", 1),
            "right": self.create_publisher(JointState, "/right_arm/control/move_j", 1),
        }
        self.create_subscription(
            PoseStamped,
            "/left_arm/feedback/tcp_pose",
            lambda msg: self._tcp_feedback_callback("left", msg),
            1,
        )
        self.create_subscription(
            PoseStamped,
            "/right_arm/feedback/tcp_pose",
            lambda msg: self._tcp_feedback_callback("right", msg),
            1,
        )
        self.create_subscription(
            JointState,
            "/left_arm/feedback/joint_states",
            lambda msg: self._joint_feedback_callback("left", msg),
            1,
        )
        self.create_subscription(
            JointState,
            "/right_arm/feedback/joint_states",
            lambda msg: self._joint_feedback_callback("right", msg),
            1,
        )
        self.create_timer(0.01, self._timer_callback)
        self.get_logger().info("PiperEnv bridge ready; native QuestArm IK owns robot output")

    def _receive_packet(self) -> dict | None:
        latest = None
        while True:
            try:
                data, _ = self._command_socket.recvfrom(1 << 20)
            except BlockingIOError:
                return latest
            except OSError:
                return latest
            try:
                packet = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(packet, dict):
                latest = packet

    def _tcp_feedback_callback(self, hand: str, msg: PoseStamped) -> None:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        ).as_matrix()
        matrix[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        state = self._hands[hand]
        state.feedback_pose = matrix
        if state.zero_matrix is None:
            state.zero_matrix = matrix.copy()

    def _joint_feedback_callback(self, hand: str, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position, strict=False))
        joints = [positions.get(f"joint{index}") for index in range(1, 7)]
        if any(value is None for value in joints):
            return
        state = self._hands[hand]
        state.joint_positions = np.asarray(joints, dtype=float)
        if "gripper" in positions:
            gripper_index = list(msg.name).index("gripper")
            state.gripper_width = float(positions["gripper"])
            state.gripper_force = float(msg.effort[gripper_index]) if len(msg.effort) > gripper_index else 0.0
            if state.gripper_target is None:
                state.gripper_target = state.gripper_width

    def _timer_callback(self) -> None:
        packet = self._receive_packet()
        if packet is not None:
            packet_type = packet.get("type")
            if packet_type == "xr_frame":
                frame = packet.get("frame")
                if isinstance(frame, dict):
                    self._handle_xr_frame(frame)
            elif packet_type == "action":
                self._handle_action(packet.get("action"))
            elif packet_type == "reset":
                self._handle_reset(packet.get("joint_positions"), packet.get("duration_s"))
            elif packet_type == "deactivate":
                self._reset_trajectory = None
                self._reset_reference()
        self._advance_reset()
        self._publish_feedback()

    def _handle_xr_frame(self, frame: dict) -> None:
        if self._reset_trajectory is not None:
            return
        controllers = frame.get("controllers", {})
        if not isinstance(controllers, dict):
            return
        stamp = self.get_clock().now().to_msg()
        for hand, state in self._hands.items():
            controller = controllers.get(hand)
            if not isinstance(controller, dict):
                state.active = False
                state.primary_down = False
                state.secondary_down = False
                continue
            pose = controller.get("grip_pose") or controller.get("target_ray_pose")
            pose_matrix = pose_to_mat(pose) if isinstance(pose, dict) else None
            if pose_matrix is None:
                continue
            handle_pose = self._adj_mat @ pose_matrix @ self._r_adj @ self._ros_to_arm_mat
            buttons = controller.get("buttons", {})
            buttons = buttons if isinstance(buttons, dict) else {}
            primary_down = self._button_pressed(buttons, "primary")
            secondary_down = self._button_pressed(buttons, "secondary")
            if primary_down and not state.primary_down and state.feedback_pose is not None:
                state.start_pose_matrix = handle_pose.copy()
                state.zero_matrix = state.feedback_pose.copy()
                state.active = True
            if secondary_down and not state.secondary_down:
                self._reset_hand_reference(state)
            state.primary_down = primary_down
            state.secondary_down = secondary_down
            trigger = self._button_value(buttons, "trigger")
            width = max(0.0, min(trigger, 1.0)) * float(self.get_parameter("gripper_max_range").value)
            self._publish_gripper(hand, width, stamp)
            if (
                state.active
                and state.feedback_pose is not None
                and state.start_pose_matrix is not None
                and state.zero_matrix is not None
            ):
                xyz, quat = calc_pose_incre(
                    state.start_pose_matrix,
                    handle_pose,
                    state.zero_matrix,
                    self._position_scale,
                    self._rotation_scale,
                )
                self._publish_target(hand, xyz, quat, stamp)

    def _handle_action(self, raw_action) -> None:
        if self._reset_trajectory is not None:
            return
        try:
            action = np.asarray(raw_action, dtype=float).reshape(2, 7)
        except (TypeError, ValueError):
            return
        stamp = self.get_clock().now().to_msg()
        for hand, command in zip(("left", "right"), action, strict=True):
            state = self._hands[hand]
            ee_delta = command[:6]
            if state.feedback_pose is not None and np.any(ee_delta != 0.0):
                target = state.feedback_pose.copy()
                target[:3, 3] += ee_delta[:3]
                target[:3, :3] = Rotation.from_rotvec(ee_delta[3:]).as_matrix() @ target[:3, :3]
                xyz, quat = mat_to_pose(target)
                self._publish_target(hand, xyz, quat, stamp)
            if command[6] != 0.0 and state.gripper_width is not None:
                current_target = state.gripper_target if state.gripper_target is not None else state.gripper_width
                next_target = current_target + np.sign(command[6]) * float(
                    self.get_parameter("gripper_width_step").value
                )
                state.gripper_target = float(
                    np.clip(
                        next_target,
                        float(self.get_parameter("gripper_min_range").value),
                        float(self.get_parameter("gripper_max_range").value),
                    )
                )
                self._publish_gripper(hand, state.gripper_target, stamp)

    def _reset_reference(self) -> None:
        for state in self._hands.values():
            self._reset_hand_reference(state)

    def _handle_reset(self, raw_joint_positions, raw_duration_s) -> None:
        try:
            joint_positions = np.asarray(raw_joint_positions, dtype=float)
            duration_s = float(raw_duration_s)
        except (TypeError, ValueError):
            return
        if (
            joint_positions.shape != (2, 6)
            or not np.all(np.isfinite(joint_positions))
            or not math.isfinite(duration_s)
            or duration_s <= 0.0
        ):
            return
        current_joint_angles = [self._hands[hand].joint_positions for hand in ("left", "right")]
        if any(positions is None for positions in current_joint_angles):
            return
        self._reset_reference()
        start_joint_angles = np.stack(current_joint_angles)
        if np.allclose(start_joint_angles, joint_positions, rtol=0.0, atol=1e-6):
            self._publish_joint_targets(joint_positions)
            self._reset_trajectory = None
            return
        self._reset_trajectory = ResetTrajectory(
            start_joint_angles=start_joint_angles,
            target_joint_angles=joint_positions,
            start_time=time.monotonic(),
            duration_s=duration_s,
            next_publish_time=0.0,
        )

    def _advance_reset(self) -> None:
        trajectory = self._reset_trajectory
        if trajectory is None:
            return
        now = time.monotonic()
        if now < trajectory.next_publish_time:
            return
        trajectory.next_publish_time = now + 1.0 / 30.0
        progress = min((now - trajectory.start_time) / trajectory.duration_s, 1.0)
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        joint_angles = trajectory.start_joint_angles + smooth_progress * (
            trajectory.target_joint_angles - trajectory.start_joint_angles
        )
        self._publish_joint_targets(joint_angles)
        if progress >= 1.0:
            self._reset_trajectory = None

    def _publish_joint_targets(self, joint_angles: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        for hand, positions in zip(("left", "right"), joint_angles, strict=True):
            target = JointState()
            target.header = Header(stamp=stamp)
            target.name = [f"joint{index}" for index in range(1, 7)]
            target.position = positions.tolist()
            self._reset_publishers[hand].publish(target)

    @staticmethod
    def _reset_hand_reference(state: HandState) -> None:
        state.active = False
        state.start_pose_matrix = None
        if state.feedback_pose is not None:
            state.zero_matrix = state.feedback_pose.copy()
        if state.gripper_width is not None:
            state.gripper_target = state.gripper_width

    def _publish_feedback(self) -> None:
        parts = []
        for hand in ("left", "right"):
            state = self._hands[hand]
            if state.joint_positions is None or state.feedback_pose is None or state.gripper_width is None:
                return
            tcp_xyz = state.feedback_pose[:3, 3]
            tcp_rpy = Rotation.from_matrix(state.feedback_pose[:3, :3]).as_euler("xyz")
            parts.extend(state.joint_positions.tolist())
            parts.extend(tcp_xyz.tolist())
            parts.extend(tcp_rpy.tolist())
            parts.extend([state.gripper_width, state.gripper_force])
        payload = json.dumps({"state": parts}, separators=(",", ":")).encode("utf-8")
        try:
            self._feedback_socket.sendto(payload, self._feedback_target)
        except OSError:
            pass

    def _publish_gripper(self, hand: str, width: float, stamp) -> None:
        gripper = JointState()
        gripper.header = Header(stamp=stamp)
        gripper.name = ["gripper"]
        gripper.position = [float(width)]
        gripper.effort = [float(self.get_parameter("gripper_force").value)]
        self._joint_publishers[hand].publish(gripper)

    def _publish_target(self, hand: str, xyz: np.ndarray, quat: np.ndarray, stamp) -> None:
        target = PoseStamped()
        target.header = Header(stamp=stamp, frame_id="vr_device")
        target.pose.position.x, target.pose.position.y, target.pose.position.z = [float(v) for v in xyz]
        target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = [
            float(v) for v in quat
        ]
        self._delta_publishers[hand].publish(target)

    @staticmethod
    def _button_value(buttons: dict, name: str) -> float:
        button = buttons.get(name, {})
        if not isinstance(button, dict):
            return 0.0
        return float(button.get("value", 1.0 if button.get("pressed", False) else 0.0))

    @classmethod
    def _button_pressed(cls, buttons: dict, name: str) -> bool:
        return cls._button_value(buttons, name) >= 0.5

    def destroy_node(self):
        self._command_socket.close()
        self._feedback_socket.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QuestArmBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
