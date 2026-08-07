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

"""Launch the upstream QuestArm dual-Piper chain for :class:`PiperEnv`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-can-port", required=True)
    parser.add_argument("--right-can-port", required=True)
    parser.add_argument("--position-scale", required=True, type=float)
    parser.add_argument("--rotation-scale", required=True, type=float)
    parser.add_argument("--ik-position-weight", required=True, type=float)
    parser.add_argument("--ik-smooth-weight", required=True, type=float)
    parser.add_argument("--gripper-max-range", required=True, type=float)
    parser.add_argument("--gripper-min-range", required=True, type=float)
    parser.add_argument("--gripper-width-step", required=True, type=float)
    parser.add_argument("--gripper-force", required=True, type=float)
    return parser.parse_args()


def create_launch_description(args: argparse.Namespace) -> LaunchDescription:
    agx_ctrl = get_package_share_directory("agx_arm_ctrl")
    oculus_reader = get_package_share_directory("oculus_reader")
    ik_config = os.path.join(oculus_reader, "config", "arm_ik_pose_node.piper_x.yaml")
    with open(ik_config, encoding="utf-8") as stream:
        ik_parameters = yaml.safe_load(stream)["arm_ik_pose_node"]["ros__parameters"]
    ik_parameters["w_pos"] = args.ik_position_weight
    ik_parameters["w_smooth"] = args.ik_smooth_weight
    ik_parameters["locked_joints"] = [*ik_parameters["locked_joints"], "gripper"]
    driver_launch = os.path.join(agx_ctrl, "launch", "start_single_agx_arm.launch.py")

    def driver(can_port: str, namespace: str) -> IncludeLaunchDescription:
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(driver_launch),
            launch_arguments={
                "can_port": can_port,
                "namespace": namespace,
                "arm_type": "piper_x",
                "auto_enable": "true",
                "effector_type": "agx_gripper",
                "tcp_offset": "[0.0, 0.0, 0.13, 0.0, 0.0, 0.0]",
                "fast_mode": "true",
                "control_enabled": "true",
            }.items(),
        )

    def ik(name: str, pose_topic: str, feedback_topic: str, output_topic: str) -> Node:
        return Node(
            package="oculus_reader",
            executable="arm_ik_pose_node.py",
            name=name,
            output="screen",
            parameters=[
                {
                    **ik_parameters,
                    "pose_stamped_topic": pose_topic,
                    "feedback_joint_topic": feedback_topic,
                    "pin_joint_status_topic": output_topic,
                }
            ],
        )

    bridge_script = Path(__file__).with_name("questarm_bridge.py")
    bridge_parameters = {
        "position_scale": args.position_scale,
        "rotation_scale": args.rotation_scale,
        "gripper_max_range": args.gripper_max_range,
        "gripper_min_range": args.gripper_min_range,
        "gripper_width_step": args.gripper_width_step,
        "gripper_force": args.gripper_force,
    }
    bridge_command = [sys.executable, str(bridge_script), "--ros-args"]
    for name, value in bridge_parameters.items():
        bridge_command.extend(["-p", f"{name}:={value}"])

    return LaunchDescription(
        [
            driver(args.left_can_port, "left_arm"),
            driver(args.right_can_port, "right_arm"),
            ik(
                "left_arm_ik_pose_node",
                "/left_delta_pose",
                "/left_arm/feedback/joint_states",
                "/left_arm/control/joint_states",
            ),
            ik(
                "right_arm_ik_pose_node",
                "/right_delta_pose",
                "/right_arm/feedback/joint_states",
                "/right_arm/control/joint_states",
            ),
            ExecuteProcess(cmd=bridge_command, output="screen"),
        ]
    )


def main() -> int:
    service = LaunchService()
    service.include_launch_description(create_launch_description(parse_args()))
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())
