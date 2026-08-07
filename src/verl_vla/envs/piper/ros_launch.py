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

"""Launch the upstream ROS nodes used by :class:`PiperEnv`."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, type=json.loads)
    parser.add_argument("--ik-position-weight", required=True, type=float)
    parser.add_argument("--ik-smooth-weight", required=True, type=float)
    parser.add_argument("--camera", action="append", default=[], type=json.loads)
    return parser.parse_args()


def _create_launch_description(args: argparse.Namespace) -> LaunchDescription:
    agx_ctrl = get_package_share_directory("agx_arm_ctrl")
    oculus_reader = get_package_share_directory("oculus_reader")
    driver_launch = os.path.join(agx_ctrl, "launch", "start_single_agx_arm.launch.py")

    def driver(arm: dict) -> IncludeLaunchDescription:
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(driver_launch),
            launch_arguments={
                "can_port": arm["can_channel"],
                "namespace": f"{arm['name']}_arm",
                "arm_type": arm["model"],
                "auto_enable": "true",
                "effector_type": "agx_gripper",
                "tcp_offset": "[0.0, 0.0, 0.13, 0.0, 0.0, 0.0]",
                "fast_mode": "true",
                "control_enabled": "true",
            }.items(),
        )

    def ik(arm: dict) -> Node:
        model = arm["model"]
        config_model = "piper_x" if model == "piper_x" else "piper"
        ik_config = os.path.join(oculus_reader, "config", f"arm_ik_pose_node.{config_model}.yaml")
        with open(ik_config, encoding="utf-8") as stream:
            parameters = yaml.safe_load(stream)
        ik_parameters = next(iter(parameters.values()))["ros__parameters"]
        ik_parameters["urdf_relative_path"] = f"agx_arm_urdf/{model}/urdf/{model}_with_gripper_description.urdf"
        ik_parameters["w_pos"] = args.ik_position_weight
        ik_parameters["w_smooth"] = args.ik_smooth_weight
        ik_parameters["locked_joints"] = [*ik_parameters["locked_joints"], "gripper"]
        hand = arm["name"]
        return Node(
            package="oculus_reader",
            executable="arm_ik_pose_node.py",
            name=f"{hand}_arm_ik_pose_node",
            output="screen",
            parameters=[
                {
                    **ik_parameters,
                    "pose_stamped_topic": f"/{hand}_delta_pose",
                    "feedback_joint_topic": f"/{hand}_arm/feedback/joint_states",
                    "pin_joint_status_topic": f"/{hand}_arm/control/joint_states",
                }
            ],
        )

    cameras = []
    for camera in args.camera:
        name = camera["name"]
        cameras.append(
            Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                namespace=name,
                name="camera",
                output="screen",
                parameters=[
                    {
                        "video_device": camera["device"],
                        "pixel_format": camera["pixel_format"],
                        "output_encoding": "rgb8",
                        "image_size": [camera["width"], camera["height"]],
                        "camera_frame_id": name,
                    }
                ],
            )
        )

    return LaunchDescription(
        [
            *(driver(arm) for arm in args.arm),
            *(ik(arm) for arm in args.arm),
            *cameras,
        ]
    )


def _run_launch() -> int:
    service = LaunchService()
    service.include_launch_description(_create_launch_description(_parse_args()))
    return service.run()


def _process_group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_for_group(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        if not _process_group_exists(pid):
            return True
        time.sleep(0.1)
    return False


def _stop_child_group(pid: int) -> int:
    for sig, timeout_s in ((signal.SIGINT, 3.0), (signal.SIGTERM, 2.0)):
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return 0
        if _wait_for_group(pid, timeout_s):
            return 0
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _wait_for_group(pid, 1.0)
    return 0


def main() -> int:
    """Supervise the launch process so abrupt Ray shutdown cannot orphan ROS nodes."""
    child_pid = os.fork()
    if child_pid == 0:
        os.setsid()
        raise SystemExit(_run_launch())

    stop_requested = False

    def request_stop(signum, frame) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stop_requested:
        completed_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if completed_pid:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.1)
    return _stop_child_group(child_pid)


if __name__ == "__main__":
    raise SystemExit(main())
