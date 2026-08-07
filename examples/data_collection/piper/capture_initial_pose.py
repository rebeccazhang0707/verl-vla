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

"""Print the current dual-Piper joint pose as a Hydra YAML value."""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class InitialPoseCapture(Node):
    def __init__(self, arm_names: list[str]) -> None:
        super().__init__("piper_initial_pose_capture")
        self.joint_angles: dict[str, list[float]] = {}
        for hand in arm_names:
            self.create_subscription(
                JointState,
                f"/{hand}_arm/feedback/joint_states",
                lambda message, hand=hand: self._joint_callback(hand, message),
                1,
            )

    def _joint_callback(self, hand: str, message: JointState) -> None:
        positions = dict(zip(message.name, message.position, strict=False))
        joint_angles = [positions.get(f"joint{index}") for index in range(1, 7)]
        if any(angle is None for angle in joint_angles):
            return
        self.joint_angles[hand] = [float(angle) for angle in joint_angles]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=("left", "right"), default=["left", "right"])
    parser.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for configured arms")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError(f"timeout must be positive, got {args.timeout}")

    rclpy.init()
    node = InitialPoseCapture(args.arms)
    try:
        deadline = time.monotonic() + args.timeout
        while len(node.joint_angles) != len(args.arms) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        missing_hands = [hand for hand in args.arms if hand not in node.joint_angles]
        if missing_hands:
            raise TimeoutError(f"Timed out waiting for joint feedback from: {', '.join(missing_hands)}")

        for hand in args.arms:
            values = ", ".join(f"{angle:.8f}" for angle in node.joint_angles[hand])
            print(f"{hand}: [{values}]")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
