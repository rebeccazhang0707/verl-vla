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
from dataclasses import dataclass
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from scipy.optimize import lsq_linear
from typing_extensions import override

from verl_vla.envs.base import BaseEnv

logger = logging.getLogger(__name__)


@dataclass
class _PiperArm:
    name: str
    robot: Any
    effector: Any
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    last_joint_angles: np.ndarray
    last_ee_pose: np.ndarray
    initial_joint_target: np.ndarray | None = None
    gripper_target_width: float = 0.0
    gripper_target_initialized: bool = False

    def disconnect(self) -> None:
        self.robot.disconnect()

    def read_joint_angles(self) -> np.ndarray:
        joint_msg = self.robot.get_joint_angles()
        if joint_msg is not None:
            self.last_joint_angles = np.asarray(joint_msg.msg, dtype=np.float32)
        return self.last_joint_angles.copy()

    def read_ee_pose(self) -> np.ndarray:
        pose_msg = self.robot.get_flange_pose()
        if pose_msg is not None:
            self.last_ee_pose = np.asarray(pose_msg.msg, dtype=np.float32)
        return self.last_ee_pose.copy()

    def read_gripper_state(self) -> np.ndarray:
        if self.effector is None:
            return np.zeros(2, dtype=np.float32)
        status = self.effector.get_gripper_status()
        if status is None:
            return np.zeros(2, dtype=np.float32)
        if not self.gripper_target_initialized:
            self.gripper_target_width = float(status.msg.value)
        return np.asarray([status.msg.value, status.msg.force], dtype=np.float32)

    def reset_gripper_target(self) -> None:
        self.gripper_target_initialized = False

    def move_gripper_by_delta(self, cfg, direction: float) -> None:
        if self.effector is None:
            logger.warning("Piper arm %s has no initialized gripper", self.name)
            return
        if not self.gripper_target_initialized:
            self.gripper_target_width = float(self.read_gripper_state()[0])
            self.gripper_target_initialized = True
        width = np.clip(
            self.gripper_target_width + np.sign(direction) * float(cfg.gripper_width_step),
            float(cfg.gripper_close_width),
            float(cfg.gripper_open_width),
        )
        self.gripper_target_width = float(width)
        self.effector.move_gripper_m(value=float(width), force=float(cfg.gripper_force))


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


class _PiperDifferentialIK:
    def __init__(self, cfg):
        self.cfg = cfg

    def joint_delta(
        self,
        *,
        robot: Any,
        joints: np.ndarray,
        ee_delta: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
    ) -> np.ndarray:
        ee_delta = ee_delta.astype(np.float32, copy=False)
        if not np.any(ee_delta != 0.0):
            return np.zeros(6, dtype=np.float32)

        lower_bound, upper_bound = self._joint_delta_bounds(
            joints=joints,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )
        if np.allclose(lower_bound, 0.0) and np.allclose(upper_bound, 0.0):
            return np.zeros(6, dtype=np.float32)

        result = lsq_linear(
            self._numerical_jacobian(robot, joints),
            ee_delta,
            bounds=(lower_bound, upper_bound),
            lsmr_tol="auto",
        )
        return result.x.astype(np.float32, copy=False)

    def _joint_delta_bounds(
        self,
        *,
        joints: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_limits = joint_lower.astype(np.float64, copy=False)
        upper_limits = joint_upper.astype(np.float64, copy=False)
        joints = joints.astype(np.float64, copy=False)
        step_limit = float(self.cfg.max_joint_delta_per_step)
        lower_delta = np.maximum(lower_limits - joints, -step_limit)
        upper_delta = np.minimum(upper_limits - joints, step_limit)
        below_limit = joints < lower_limits
        above_limit = joints > upper_limits
        lower_delta[below_limit] = 0.0
        upper_delta[below_limit] = step_limit
        lower_delta[above_limit] = -step_limit
        upper_delta[above_limit] = 0.0
        return lower_delta, upper_delta

    def _numerical_jacobian(self, robot: Any, joints: np.ndarray) -> np.ndarray:
        joints = joints.astype(np.float64, copy=True)
        base_pose = np.asarray(robot.fk(joints.astype(float).tolist()), dtype=np.float64)
        eps = float(self.cfg.ik_jacobian_eps)
        columns = []
        for joint_idx in range(6):
            perturbed = joints.copy()
            perturbed[joint_idx] += eps
            pose = np.asarray(robot.fk(perturbed.astype(float).tolist()), dtype=np.float64)
            columns.append(self._pose_error(pose, base_pose) / eps)
        return np.stack(columns, axis=1)

    @classmethod
    def _pose_error(cls, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        error = np.zeros(6, dtype=np.float64)
        error[:3] = target[:3] - current[:3]
        target_rotation = cls._pose_rotation(target)
        current_rotation = cls._pose_rotation(current)
        error[3:] = (target_rotation * current_rotation.inv()).as_rotvec()
        return error

    @staticmethod
    def _pose_rotation(pose: np.ndarray):
        from pyAgxArm.utiles.tf import rpy_to_rot
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(np.asarray(rpy_to_rot(*pose[3:]), dtype=np.float64))


class _PiperArmSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.arm_count = len(cfg.can_channels)
        self._ik = _PiperDifferentialIK(cfg)
        self._arms: list[_PiperArm] = []

    def connect(self) -> None:
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyAgxArm is required for PiperEnv. Install the piper dependency before running vvla-teleop."
            ) from exc

        arm_models = {
            "piper": ArmModel.PIPER,
            "piper_h": ArmModel.PIPER_H,
            "piper_l": ArmModel.PIPER_L,
            "piper_x": ArmModel.PIPER_X,
        }
        robot_model_names = self._robot_model_names()

        self.close()
        for arm_name, channel, robot_model_name in zip(
            self.cfg.arm_names,
            self.cfg.can_channels,
            robot_model_names,
            strict=False,
        ):
            arm = self._connect_arm(
                arm_name=str(arm_name),
                channel=str(channel),
                robot_model_name=str(robot_model_name),
                arm_models=arm_models,
                arm_factory=AgxArmFactory,
                config_factory=create_agx_arm_config,
            )
            self._arms.append(arm)
        self._set_initial_joint_targets()

    def _connect_arm(
        self,
        *,
        arm_name: str,
        channel: str,
        robot_model_name: str,
        arm_models: dict[str, Any],
        arm_factory: Any,
        config_factory: Any,
    ) -> _PiperArm:
        if robot_model_name not in arm_models:
            raise ValueError(
                f"Unsupported Piper robot_model {robot_model_name!r}; expected one of {sorted(arm_models)}"
            )

        robot_cfg = config_factory(
            robot=arm_models[robot_model_name],
            firmeware_version=str(self.cfg.firmware_version),
            interface=str(self.cfg.can_interface),
            channel=channel,
            bitrate=int(self.cfg.bitrate),
            log_level="WARNING",
        )
        joint_lowers, joint_uppers = self._joint_limits_from_config(robot_cfg)
        robot = arm_factory.create_arm(robot_cfg)
        try:
            robot.connect()
            joint_msg = self._wait_for(robot.get_joint_angles, f"{arm_name} joint angles")
            joint_angles = np.asarray(joint_msg.msg, dtype=np.float32)
            pose_msg = self._wait_for(robot.get_flange_pose, f"{arm_name} ee pose")
            ee_pose = np.asarray(pose_msg.msg, dtype=np.float32)
            robot.set_speed_percent(int(self.cfg.speed_percent))
            if not self._ensure_robot_enabled(robot):
                logger.warning("Piper arm %s did not report enabled during initialization", arm_name)
            return _PiperArm(
                name=arm_name,
                robot=robot,
                effector=self._init_gripper(robot, arm_name),
                joint_lower=joint_lowers,
                joint_upper=joint_uppers,
                last_joint_angles=joint_angles,
                last_ee_pose=ee_pose,
            )
        except Exception:
            robot.disconnect()
            raise

    def close(self) -> None:
        for arm in self._arms:
            arm.disconnect()
        self._arms = []

    def reset_to_initial_pose(self) -> None:
        if not self._arms:
            return
        reset_speed = int(self.cfg.reset_speed_percent)
        normal_speed = int(self.cfg.speed_percent)
        active_targets = []
        for arm in self._arms:
            current_joints = arm.read_joint_angles()
            target_joints = arm.initial_joint_target
            if target_joints is None:
                continue
            if np.allclose(current_joints, target_joints, atol=float(self.cfg.reset_joint_tolerance)):
                continue
            if not self._ensure_robot_enabled(arm.robot):
                logger.warning("Piper arm %s did not report enabled before reset move", arm.name)
                continue
            arm.robot.set_speed_percent(reset_speed)
            arm.robot.move_j(target_joints.astype(float).tolist())
            active_targets.append((arm, target_joints.copy()))
        try:
            for arm, target_joints in active_targets:
                if not self._wait_for_reset_target(arm, target_joints):
                    logger.warning("Piper arm %s did not reach reset target before timeout", arm.name)
        finally:
            for arm, _ in active_targets:
                arm.robot.set_speed_percent(normal_speed)

    def reset_gripper_targets(self) -> None:
        for arm in self._arms:
            arm.reset_gripper_target()

    def apply_action(self, action: np.ndarray) -> None:
        commands = action.reshape(self.arm_count, 7)
        for arm_idx, command in enumerate(commands):
            ee_delta = command[:6]
            if np.any(ee_delta != 0.0):
                self._move_arm_by_ee_delta(self._arms[arm_idx], ee_delta)
            gripper_command = float(command[6])
            if gripper_command != 0.0:
                self._arms[arm_idx].move_gripper_by_delta(self.cfg, gripper_command)

    def read_state(self) -> np.ndarray:
        parts = []
        for arm in self._arms:
            parts.append(arm.read_joint_angles())
            parts.append(arm.read_ee_pose())
            parts.append(arm.read_gripper_state())
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _robot_model_names(self) -> list[str]:
        robot_model_names = [str(model) for model in self.cfg.robot_models]
        if len(robot_model_names) != self.arm_count:
            raise ValueError(
                f"robot_models must contain {self.arm_count} per-arm model names, got {len(robot_model_names)}"
            )
        return robot_model_names

    def _move_arm_by_ee_delta(self, arm: _PiperArm, delta: np.ndarray) -> None:
        current_joints = arm.read_joint_angles()
        joint_delta = self._ik.joint_delta(
            robot=arm.robot,
            joints=current_joints,
            ee_delta=delta,
            joint_lower=arm.joint_lower,
            joint_upper=arm.joint_upper,
        )
        if not np.any(joint_delta != 0.0):
            return
        target_joints = current_joints + joint_delta
        if not self._ensure_robot_enabled(arm.robot):
            logger.warning("Piper arm %s did not report enabled before move command", arm.name)
            return
        arm.robot.move_j(target_joints.astype(float).tolist())

    def _wait_for_reset_target(self, arm: _PiperArm, target_joints: np.ndarray) -> bool:
        deadline = time.time() + float(self.cfg.reset_timeout_s)
        tolerance = float(self.cfg.reset_joint_tolerance)
        poll_interval = float(self.cfg.reset_poll_interval_s)
        while time.time() < deadline:
            current_joints = arm.read_joint_angles()
            status = arm.robot.get_arm_status()
            motion_done = status is not None and getattr(status.msg, "motion_status", None) == 0
            if np.allclose(current_joints, target_joints, atol=tolerance) and motion_done:
                return True
            time.sleep(poll_interval)
        return bool(np.allclose(arm.read_joint_angles(), target_joints, atol=tolerance))

    def _init_gripper(self, robot, arm_name: str):
        try:
            return robot.init_effector("agx_gripper")
        except Exception:
            logger.exception("Failed to initialize Piper gripper for arm %s", arm_name)
            return None

    @staticmethod
    def _joint_limits_from_config(robot_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
        limits = robot_cfg["joint_limits"]
        ordered_limits = [limits[f"joint{idx}"] for idx in range(1, 7)]
        return (
            np.asarray([limit[0] for limit in ordered_limits], dtype=np.float32),
            np.asarray([limit[1] for limit in ordered_limits], dtype=np.float32),
        )

    def _set_initial_joint_targets(self) -> None:
        targets = self._resolve_initial_joint_targets()
        for arm, target in zip(self._arms, targets, strict=True):
            arm.initial_joint_target = target.astype(np.float32, copy=True)

    def _resolve_initial_joint_targets(self) -> np.ndarray:
        startup_joint_angles = np.stack([arm.last_joint_angles for arm in self._arms]).astype(
            np.float32,
            copy=False,
        )
        configured_targets = self.cfg.initial_joint_angles
        if configured_targets is None:
            return self._clip_startup_joint_targets(startup_joint_angles)
        targets = np.asarray(configured_targets, dtype=np.float32)
        joint_lowers = np.stack([arm.joint_lower for arm in self._arms])
        joint_uppers = np.stack([arm.joint_upper for arm in self._arms])
        lower_ok = targets >= joint_lowers
        upper_ok = targets <= joint_uppers
        if not bool(np.all(lower_ok & upper_ok)):
            raise ValueError("initial_joint_angles contains a target outside the configured Piper joint limits")
        return targets

    def _clip_startup_joint_targets(self, startup_joint_angles: np.ndarray) -> np.ndarray:
        joint_lowers = np.stack([arm.joint_lower for arm in self._arms])
        joint_uppers = np.stack([arm.joint_upper for arm in self._arms])
        targets = np.clip(startup_joint_angles, joint_lowers, joint_uppers)
        for arm_idx, delta in enumerate(np.abs(targets - startup_joint_angles).max(axis=1)):
            if float(delta) > 0.0:
                logger.warning(
                    "Piper arm %s startup initial target was clipped by %.6frad to satisfy SDK joint limits",
                    self._arms[arm_idx].name,
                    float(delta),
                )
        return targets.astype(np.float32, copy=False)

    @staticmethod
    def _wait_for(getter, name: str, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            value = getter()
            if value is not None:
                return value
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for Piper {name}")

    @staticmethod
    def _ensure_robot_enabled(robot, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if robot.enable():
                return True
            time.sleep(0.05)
        return bool(robot.enable())


class PiperEnv(BaseEnv):
    """Real Piper environment for browser teleoperation."""

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

        self.action_dim = int(self.piper_cfg.action_dim)
        self.state_dim = int(self.piper_cfg.state_dim)
        self.arm_count = len(self.piper_cfg.can_channels)
        self.task_description = str(self.piper_cfg.task_description)
        self.task_descriptions = [self.task_description]
        self._arms = _PiperArmSystem(self.piper_cfg)
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
        self._arms.connect()
        self._cameras.open()

    @override
    def env_reset(self, *, env_ids, reset_eval: bool = False):
        del reset_eval
        self._validate_env_ids(env_ids)
        self._step_id = 0
        self._arms.reset_to_initial_pose()
        self._arms.reset_gripper_targets()
        return self._step_result(
            reward=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            success=np.zeros(1, dtype=bool),
        )

    @override
    def env_step(self, action, *, env_ids):
        self._validate_env_ids(env_ids)
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        if action.shape[1] != self.action_dim:
            raise ValueError(f"Piper action must have shape [1, {self.action_dim}], got {action.shape}")

        self._arms.apply_action(action[0])
        self._step_id += 1
        return self._step_result(
            reward=np.zeros(1, dtype=np.float32),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
            success=np.zeros(1, dtype=bool),
        )

    @override
    def env_close(self) -> None:
        self._arms.close()
        self._cameras.close()

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
        obs = {"observation.state": self._arms.read_state()}
        for name, image in self._cameras.read_images().items():
            obs[f"observation.images.{name}"] = image
        return obs
