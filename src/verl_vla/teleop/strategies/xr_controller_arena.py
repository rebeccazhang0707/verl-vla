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

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import override

from verl_vla.teleop.config import XRControllerTeleopConfig
from verl_vla.teleop.devices import DeviceBase, XRControllerDevice
from verl_vla.teleop.strategies.base import InterventionStrategyBase

logger = logging.getLogger(__name__)


class ArenaXRControllerStrategy(InterventionStrategyBase):
    """Quest controller intervention for Arena G1 50D joint actions.

    The browser sends WebXR controller poses. This strategy uses relative
    controller motion to update left/right wrist targets, runs Arena's existing
    G1 PINK IK controller, and returns the 50D ``g1_wbc_joint`` action:
    43 sim-order joint targets + 3D navigation + 1D base height + 3D torso rpy.
    """

    env_type = "arena"
    device_type = "xr_controller"

    _POSITION_AXES = (2, 0, 1)
    _POSITION_SIGNS = (-1.0, -1.0, 1.0)
    _ROTATION_AXES = (2, 0, 1)
    _ROTATION_SIGNS = (-1.0, -1.0, 1.0)

    _MOVE_SCALE = 0.5
    _TURN_SCALE = 1.0
    _BASE_HEIGHT_INITIAL = 0.75
    _BASE_HEIGHT_STEP = 0.01
    _BASE_HEIGHT_MIN = 0.35
    _BASE_HEIGHT_MAX = 1.2
    _THUMBSTICK_DEADZONE = 0.2
    _CONTROLLER_POSITION_DEADZONE = 0.002
    _CONTROLLER_ROTATION_DEADZONE = 0.01
    _INSTRUCTIONS: ClassVar[tuple[str, ...]] = (
        "Squeeze: enter VR intervention.",
        "Release squeeze: return to autonomous rollout.",
        "Move left controller: move G1 left wrist relatively.",
        "Move right controller: move G1 right wrist relatively.",
        "Rotate left controller: rotate G1 left wrist relatively.",
        "Rotate right controller: rotate G1 right wrist relatively.",
        "Left trigger: close left hand.",
        "Right trigger: close right hand.",
        "Left thumbstick up/down: move base forward/backward.",
        "Left thumbstick left/right: move base left/right.",
        "Right thumbstick left/right: yaw turn base.",
        "Right thumbstick up/down: raise/lower base height.",
    )

    def __init__(
        self,
        cfg: XRControllerTeleopConfig | None = None,
    ):
        xr_cfg = cfg or XRControllerTeleopConfig()
        super().__init__(xr_cfg)
        self.cfg = xr_cfg
        self._active = False
        self._adapter = self._create_adapter()
        self._controller_refs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._ee_refs: dict[str, np.ndarray] = {}
        self._base_height = self._BASE_HEIGHT_INITIAL
        self._last_command = self._default_command()

    @override
    def reset(self) -> None:
        self._active = False
        self._controller_refs.clear()
        self._ee_refs.clear()
        self._base_height = self._BASE_HEIGHT_INITIAL
        self._last_command = self._default_command()
        if self._adapter is not None:
            self._adapter.reset()

    @override
    def is_intervening(self, device: DeviceBase) -> bool:
        frame = cast(XRControllerDevice, device).latest_frame()
        active = self._is_frame_active(frame)
        if active and not self._active:
            self._controller_refs.clear()
            self._ee_refs.clear()
        elif not active and self._active:
            self._controller_refs.clear()
            self._ee_refs.clear()
        self._active = active
        return active

    @override
    def apply_action(self, action: Any, device: DeviceBase) -> Any:
        action_array = np.asarray(action)
        if not self.is_intervening(device):
            return action

        return self._apply_active_action(action_array, device)

    @override
    def get_action(self, device: DeviceBase) -> Any:
        if not self.is_intervening(device):
            self._last_command[43:46] = 0.0
            return self._last_command.copy()

        return self._apply_active_action(self._last_command, device)

    def _apply_active_action(self, action_array: np.ndarray, device: DeviceBase) -> np.ndarray:
        if self._adapter is None:
            return action_array

        frame = cast(XRControllerDevice, device).latest_frame()
        self._base_height = self._base_height_from_action(action_array)
        if not self._controller_refs:
            self._set_reference(frame, action_array)
        wrist_targets = self._wrist_targets(frame)
        hand_commands = self._hand_commands(frame)
        locomotion = self._locomotion(frame, action_array)
        torso_rpy = self._torso_rpy(action_array)
        command = self._adapter.build_action(
            base_action=action_array,
            wrist_targets=wrist_targets,
            hand_commands=hand_commands,
            locomotion=locomotion,
            torso_rpy=torso_rpy,
        )
        self._last_command = command.astype(np.float32, copy=False)
        return self._last_command.astype(action_array.dtype, copy=False)

    def _default_command(self) -> np.ndarray:
        command = np.zeros(50, dtype=np.float32)
        command[46] = self._BASE_HEIGHT_INITIAL
        return command

    @override
    def snapshot(self, device: DeviceBase) -> dict[str, Any]:
        return {
            "strategy": f"{self.env_type}:{self.device_type}",
            "is_intervening": self._active,
            "active": self._active,
            "base_height": float(self._base_height),
            "adapter_ready": self._adapter is not None,
            "instructions": list(self._INSTRUCTIONS),
            **cast(XRControllerDevice, device).snapshot(),
        }

    def _create_adapter(self) -> _ArenaG1EEToJointAdapter | None:
        try:
            return _ArenaG1EEToJointAdapter()
        except Exception:
            logger.exception("Failed to initialise Arena G1 EE-to-joint teleop adapter")
            return None

    def _set_reference(self, frame: dict[str, Any], action: np.ndarray) -> None:
        if self._adapter is None:
            return
        self._controller_refs.clear()
        self._ee_refs = self._adapter.current_wrist_poses(action)
        for hand in ("left", "right"):
            pose = self._controller_pose(frame, hand)
            if pose is not None and hand in self._ee_refs:
                self._controller_refs[hand] = pose

    def _is_frame_active(self, frame: dict[str, Any]) -> bool:
        for hand in ("left", "right"):
            controller = self._controller(frame, hand)
            if self._button_value(controller, str(self.cfg.intervention_button)) >= self.cfg.button_threshold:
                return True
        return False

    def _wrist_targets(self, frame: dict[str, Any]) -> dict[str, np.ndarray]:
        targets: dict[str, np.ndarray] = {}
        for hand in ("left", "right"):
            ref = self._controller_refs.get(hand)
            ee_ref = self._ee_refs.get(hand)
            pose = self._controller_pose(frame, hand)
            if ref is None or ee_ref is None or pose is None:
                continue

            ref_pos, ref_quat = ref
            pos, quat = pose
            raw_delta_pos = pos - ref_pos
            raw_delta_rot = Rotation.from_quat(quat) * Rotation.from_quat(ref_quat).inv()
            raw_delta_rotvec = raw_delta_rot.as_rotvec()
            if (
                np.linalg.norm(raw_delta_pos) <= self._CONTROLLER_POSITION_DEADZONE
                and np.linalg.norm(raw_delta_rotvec) <= self._CONTROLLER_ROTATION_DEADZONE
            ):
                continue

            delta_pos = self._remap_vector(raw_delta_pos, self._POSITION_AXES, self._POSITION_SIGNS)
            delta_pos *= float(self.cfg.pos_sensitivity)
            delta_rot = self._remap_rotation(raw_delta_rot, self._ROTATION_AXES, self._ROTATION_SIGNS)
            delta_rotvec = delta_rot.as_rotvec() * float(self.cfg.rot_sensitivity)
            delta_rot = Rotation.from_rotvec(delta_rotvec)

            target = ee_ref.copy()
            target[:3, 3] = ee_ref[:3, 3] + delta_pos
            target[:3, :3] = delta_rot.as_matrix() @ ee_ref[:3, :3]
            targets[hand] = target
        return targets

    def _hand_commands(self, frame: dict[str, Any]) -> tuple[tuple[float, float], tuple[bool, bool]]:
        left = self._button_value(self._controller(frame, "left"), str(self.cfg.gripper_button))
        right = self._button_value(self._controller(frame, "right"), str(self.cfg.gripper_button))
        left_active = left >= self.cfg.button_threshold
        right_active = right >= self.cfg.button_threshold
        return (
            (1.0 if left_active else 0.0, 1.0 if right_active else 0.0),
            (left_active, right_active),
        )

    def _locomotion(self, frame: dict[str, Any], action: np.ndarray) -> tuple[np.ndarray, float]:
        left_x, left_y = self._thumbstick(self._controller(frame, "left"))
        right_x, right_y = self._thumbstick(self._controller(frame, "right"))
        left_x = self._apply_deadzone(left_x)
        left_y = self._apply_deadzone(left_y)
        right_x = self._apply_deadzone(right_x)
        right_y = self._apply_deadzone(right_y)
        navigate = np.zeros(3, dtype=np.float32)
        if left_x != 0.0 or left_y != 0.0 or right_x != 0.0:
            navigate = np.asarray(
                [-left_y * self._MOVE_SCALE, -left_x * self._MOVE_SCALE, -right_x * self._TURN_SCALE],
                dtype=np.float32,
            )
        self._base_height = float(
            np.clip(
                self._base_height - right_y * self._BASE_HEIGHT_STEP,
                self._BASE_HEIGHT_MIN,
                self._BASE_HEIGHT_MAX,
            )
        )
        del action
        return navigate, self._base_height

    def _controller(self, frame: dict[str, Any], hand: str) -> dict[str, Any]:
        controllers = frame.get("controllers", {})
        controller = controllers.get(hand, {})
        return controller if isinstance(controller, dict) else {}

    def _controller_pose(self, frame: dict[str, Any], hand: str) -> tuple[np.ndarray, np.ndarray] | None:
        controller = self._controller(frame, hand)
        pose = controller.get("grip_pose") or controller.get("target_ray_pose") or {}
        if not isinstance(pose, dict):
            return None
        position = pose.get("position")
        orientation = pose.get("orientation")
        if position is None or orientation is None:
            return None
        pos = np.asarray(position, dtype=np.float32)
        quat = np.asarray(orientation, dtype=np.float32)
        if pos.shape != (3,) or quat.shape != (4,):
            return None
        return pos, quat

    def _button_value(self, controller: dict[str, Any], button_name: str) -> float:
        buttons = controller.get("buttons", {})
        button = buttons.get(button_name, {})
        if isinstance(button, dict):
            return float(button.get("value", 0.0))
        return 0.0

    def _thumbstick(self, controller: dict[str, Any]) -> tuple[float, float]:
        axes = np.asarray(controller.get("axes", []), dtype=np.float32)
        if axes.size < 2:
            return 0.0, 0.0
        return float(axes[-2]), float(axes[-1])

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self._THUMBSTICK_DEADZONE:
            return 0.0
        return value

    def _base_height_from_action(self, action: np.ndarray) -> float:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.size <= 46:
            return self._BASE_HEIGHT_INITIAL
        return float(flat[46])

    @staticmethod
    def _torso_rpy(action: np.ndarray) -> np.ndarray:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.size < 50:
            return np.zeros(3, dtype=np.float32)
        return flat[47:50].astype(np.float32, copy=True)

    @staticmethod
    def _remap_vector(vector: np.ndarray, axes: tuple[int, int, int], signs: tuple[float, float, float]) -> np.ndarray:
        return np.asarray([vector[axis] * sign for axis, sign in zip(axes, signs, strict=True)], dtype=np.float32)

    @staticmethod
    def _remap_rotation(rotation: Rotation, axes: tuple[int, int, int], signs: tuple[float, float, float]) -> Rotation:
        transform = np.zeros((3, 3), dtype=np.float32)
        for output_axis, (input_axis, sign) in enumerate(zip(axes, signs, strict=True)):
            transform[output_axis, input_axis] = sign
        return Rotation.from_matrix(transform @ rotation.as_matrix() @ transform.T)


class _ArenaG1EEToJointAdapter:
    _ACTION_DIM = 50
    _JOINT_DIM = 43
    _SIM_JOINT_NAMES: ClassVar[tuple[str, ...]] = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
        "left_hand_index_0_joint",
        "left_hand_middle_0_joint",
        "left_hand_thumb_0_joint",
        "right_hand_index_0_joint",
        "right_hand_middle_0_joint",
        "right_hand_thumb_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_1_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "right_hand_thumb_2_joint",
    )

    def __init__(self):
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.g1_wbc_upperbody_ik import (
            g1_wbc_upperbody_controller,
        )
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.run_policy import (
            convert_sim_joint_to_wbc_joint,
        )
        from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 import (
            instantiate_g1_robot_model,
        )

        self.robot_model = instantiate_g1_robot_model(waist_location="lower_body")
        self.wbc_order = self.robot_model.wbc_g1_joints_order
        self.sim_joint_names = list(self._SIM_JOINT_NAMES)
        self._convert_sim_joint_to_wbc_joint = convert_sim_joint_to_wbc_joint
        self.controller = g1_wbc_upperbody_controller.G1WBCUpperbodyController(
            robot_model=self.robot_model,
            body_active_joint_groups=["arms"],
        )

    def reset(self) -> None:
        if hasattr(self.controller, "in_warmup"):
            self.controller.in_warmup = True

    def current_wrist_poses(self, action: np.ndarray) -> dict[str, np.ndarray]:
        wbc_q = self._action_to_wbc_q(action)
        self.robot_model.cache_forward_kinematics(wbc_q)
        return {
            "left": self._frame_pose("left_wrist_yaw_link"),
            "right": self._frame_pose("right_wrist_yaw_link"),
        }

    def build_action(
        self,
        *,
        base_action: np.ndarray,
        wrist_targets: dict[str, np.ndarray],
        hand_commands: tuple[tuple[float, float], tuple[bool, bool]],
        locomotion: tuple[np.ndarray, float],
        torso_rpy: np.ndarray,
    ) -> np.ndarray:
        action = np.asarray(base_action, dtype=np.float32).copy()
        if action.shape[-1] != 50:
            action = np.zeros(50, dtype=np.float32)

        hand_states, hand_active = hand_commands
        if wrist_targets:
            self._seed_ik_from_action(action)
            target_poses = {f"{hand}_wrist_yaw_link": pose for hand, pose in wrist_targets.items()}
            target_wbc = self.controller.inverse_kinematics(
                target_poses,
                hand_states[0],
                hand_states[1],
            )
            action[: self._JOINT_DIM] = self._wbc_to_sim_order(np.asarray(target_wbc, dtype=np.float32))
            self._restore_non_teleop_joints(
                action,
                np.asarray(base_action, dtype=np.float32),
                hand_active,
                active_wrist_sides=set(wrist_targets),
            )
        self._apply_hand_commands(action, hand_commands)

        navigate, base_height = locomotion
        action[43:46] = navigate
        action[46] = np.float32(base_height)
        action[47:50] = np.asarray(torso_rpy, dtype=np.float32)
        return action

    def _action_to_wbc_q(self, action: np.ndarray) -> np.ndarray:
        sim_joint_pos = np.asarray(action, dtype=np.float32)[: self._JOINT_DIM][None]
        return self._convert_sim_joint_to_wbc_joint(sim_joint_pos, self.sim_joint_names, self.wbc_order)[0]

    def _wbc_to_sim_order(self, wbc_joint_pos: np.ndarray) -> np.ndarray:
        sim_joint_pos = np.zeros(self._JOINT_DIM, dtype=np.float32)
        for joint_name, wbc_index in self.wbc_order.items():
            if joint_name in self.sim_joint_names:
                sim_joint_pos[self.sim_joint_names.index(joint_name)] = wbc_joint_pos[wbc_index]
        return sim_joint_pos

    def _restore_non_teleop_joints(
        self,
        action: np.ndarray,
        base_action: np.ndarray,
        hand_active: tuple[bool, bool],
        *,
        active_wrist_sides: set[str],
    ) -> None:
        hand_active_by_side = dict(zip(("left", "right"), hand_active, strict=True))
        for index, joint_name in enumerate(self.sim_joint_names):
            side = joint_name.split("_", 1)[0]
            if self._is_arm_joint(joint_name) and side in active_wrist_sides:
                continue
            if joint_name.startswith(f"{side}_hand_") and hand_active_by_side.get(side, False):
                continue
            action[index] = base_action[index]

    def _apply_hand_commands(
        self,
        action: np.ndarray,
        hand_commands: tuple[tuple[float, float], tuple[bool, bool]],
    ) -> None:
        hand_states, _hand_active = hand_commands
        for side, hand_state in zip(("left", "right"), hand_states, strict=True):
            joint_values = self.controller.get_hand_joint_pos(hand_state)
            if side == "right":
                joint_values = -joint_values
            joint_names = (
                f"{side}_hand_thumb_0_joint",
                f"{side}_hand_thumb_1_joint",
                f"{side}_hand_thumb_2_joint",
                f"{side}_hand_index_0_joint",
                f"{side}_hand_index_1_joint",
                f"{side}_hand_middle_0_joint",
                f"{side}_hand_middle_1_joint",
            )
            for joint_name, joint_value in zip(joint_names, joint_values, strict=True):
                action[self.sim_joint_names.index(joint_name)] = joint_value

    @staticmethod
    def _is_arm_joint(joint_name: str) -> bool:
        return any(part in joint_name for part in ("shoulder", "elbow", "wrist"))

    def _seed_ik_from_action(self, action: np.ndarray) -> None:
        wbc_q = self._action_to_wbc_q(action)
        ik_solver = self.controller.body_ik_solver
        if self.controller.using_reduced_robot_model:
            ik_q = self.controller.body.full_to_reduced_configuration(wbc_q)
        else:
            ik_q = wbc_q
        ik_solver.configuration.q = self.controller.body.clip_configuration(ik_q)
        ik_solver.configuration.update()
        self.controller.body.cache_forward_kinematics(ik_solver.configuration.q)
        for task in ik_solver.tasks.values():
            task.set_target_from_configuration(ik_solver.configuration)

    def _frame_pose(self, frame_name: str) -> np.ndarray:
        pose = self.robot_model.frame_placement(frame_name)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = pose.translation
        matrix[:3, :3] = pose.rotation
        return matrix
