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

"""Isaac Lab Arena environment adapted to the shared BaseEnv interface."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from typing_extensions import override

from verl_vla.envs.arena.embodiment import make_arena_embodiment
from verl_vla.envs.arena.utils import (
    disable_lightwheel_ssl_verify,
    register_external_arena_env,
)
from verl_vla.envs.base import BaseEnv
from verl_vla.utils.envs.action import to_tensor

logger = logging.getLogger(__name__)


class IsaacLabArenaEnv(BaseEnv):
    """Arena vector environment with BaseEnv-owned chunking, recording and teleop.

    Embodiment-agnostic: every robot/control-mode-specific concern (CLI args, env
    cfg patching, policy->sim action conversion, state/image extraction, stable-hold
    joint indices) lives in an :class:`~verl_vla.envs.arena.embodiment.ArenaEmbodiment`
    adapter held as ``self.embodiment`` and selected by ``arena_state_mode`` (default
    ``g1_wbc_joint``).

    Auto-reset is layered: IsaacLab resets terminated/timed-out envs *intra-chunk*
    (``ManagerBasedRLEnv.step`` keeps the termination terms and resets done envs
    in-step), while ``env_step`` passes the sim's ``reward``/``terminated``/``truncated``
    straight through. Because the chunk-wise MDP keeps stepping the remaining chunk
    actions on those in-step-reset envs, ``BaseEnv._reset_done_envs`` re-resets the
    done envs at the *chunk boundary* via ``env_reset`` so the next chunk starts from a
    clean episode.
    """

    env_type = "arena"

    def __init__(
        self,
        cfg,
        rank: int,
        world_size: int,
        stage_id: int = 0,
        stage_num: int = 1,
        only_eval: bool = False,
    ):
        del stage_num, only_eval
        disable_lightwheel_ssl_verify()

        self.arena_cfg = OmegaConf.to_object(cfg.simulator.arena)
        self.seed = int(self.arena_cfg.seed) + int(rank)
        self.device = getattr(cfg, "device", None) or "cuda:0"
        self.enable_cameras = self.arena_cfg.enable_cameras
        self.camera_names = list(self.arena_cfg.camera_names)
        self.task_description = self.arena_cfg.task_description

        self.action_dim = int(self.arena_cfg.action_dim)
        self.state_dim = int(self.arena_cfg.state_dim or self.action_dim)
        self.env = None
        self.app = None

        # Embodiment adapter: owns joint maps, action conversion, state/image
        # extraction, CLI args, env-cfg patching, camera names and the stable-hold
        # indices. The wrapper delegates to it so it stays embodiment-agnostic.
        self.embodiment = make_arena_embodiment(self.arena_cfg, num_envs=int(cfg.num_envs))
        # Whether to step the raw policy action or route through the stable-hold /
        # teleop adapter. Embodiment-driven: G1 WBC -> False (unchanged smoke path),
        # GR1 joint / Franka LIBERO -> True (execute real policy actions).
        self.use_policy_action = bool(self.embodiment.use_policy_action)

        # Stable-hold buffer: hold the leading joint targets + base-height command.
        # The magic indices live on the embodiment (None => stable-hold disabled).
        self._stable_actions = np.zeros((int(cfg.num_envs), self.action_dim), dtype=np.float32)
        if self.embodiment.base_height_index is not None:
            self._stable_actions[:, self.embodiment.base_height_index] = self.embodiment.base_height_command

        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=self.enable_cameras).app
        super().__init__(cfg, rank, world_size, stage_id=stage_id)

    @override
    def env_init(self) -> None:
        self._init_env()

    def _build_args(self) -> argparse.Namespace:
        # Generic builder args; embodiment/task-specific knobs (object/kitchen_style
        # for G1/GR1, task_suite/task_id/... for LIBERO) are added by the adapter so
        # this method stays embodiment-agnostic.
        args = argparse.Namespace(
            num_envs=self.num_envs,
            env_spacing=self.arena_cfg.env_spacing,
            disable_fabric=self.arena_cfg.disable_fabric,
            device=self.device,
            seed=self.seed,
            solve_relations=self.arena_cfg.solve_relations,
            mimic=False,
            enable_pinocchio=self.arena_cfg.enable_pinocchio,
            placement_seed=self.arena_cfg.placement_seed,
            resolve_on_reset=self.arena_cfg.resolve_on_reset,
            presets=self.arena_cfg.presets,
            embodiment=self.arena_cfg.embodiment,
            enable_cameras=self.enable_cameras,
            teleop_device=None,
        )
        self.embodiment.add_cli_args(args, self.arena_cfg)
        return args

    def _init_env(self) -> None:
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena_environments.cli import ExampleEnvironments

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                logger.exception("Failed to close previous Arena env")
            import omni

            omni.usd.get_context().new_stage()

        disable_lightwheel_ssl_verify()

        args = self._build_args()

        # External (non-built-in) Arena env registration
        register_external_arena_env(self.arena_cfg.env_name, self.arena_cfg.external_env_class_path)
        if self.arena_cfg.env_name not in ExampleEnvironments:
            raise ValueError(
                f"Arena env '{self.arena_cfg.env_name}' not found. Available: {sorted(ExampleEnvironments.keys())}"
            )

        arena_env = ExampleEnvironments[self.arena_cfg.env_name]().get_env(args)
        task = getattr(arena_env, "task", None)
        if task is not None and hasattr(task, "get_task_description"):
            desc = task.get_task_description()
            if desc:
                self.task_description = desc

        env_builder = ArenaEnvBuilder(arena_env, args)
        _, env_cfg = env_builder.build_registered()
        # Embodiment-owned cfg patch: for G1/GR1 this turns composite-success into a
        # sparse RL reward (gated on rl_success_reward) -- WITHOUT touching the
        # termination terms, so IsaacLab keeps owning auto-reset (episode horizon stays
        # the Arena task's native episode_length_s).
        self.embodiment.patch_env_cfg(env_cfg, self.arena_cfg)
        self.env = env_builder.make_registered(env_cfg=env_cfg)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        base = getattr(self.env, "unwrapped", self.env)
        action_mgr = getattr(base, "action_manager", None)
        if action_mgr is not None:
            self.action_dim = int(action_mgr.total_action_dim)
        logger.info(
            "Arena environment initialised: state_mode=%s action_dim=%d state_dim=%d cameras=%s",
            self.embodiment.state_mode,
            self.action_dim,
            self.state_dim,
            self.camera_names,
        )

    @property
    def _raw_env(self):
        return getattr(self.env, "unwrapped", self.env)

    def _extract_success(self, sim_terminated: np.ndarray) -> np.ndarray:
        """Per-env success = the sim's dedicated ``success`` termination term this step.

        The sim's ``terminated`` conflates success with failure terminations (e.g.
        ``object_dropped``), so we read the ``success`` term specifically from the
        TerminationManager (it survives the in-step reset -- ``_term_dones`` is only
        overwritten on the next ``compute``). Tasks without a ``success`` term (should
        not happen for G1/GR1) fall back to the natural termination.
        """
        tm = getattr(self._raw_env, "termination_manager", None)
        if tm is not None and "success" in getattr(tm, "active_terms", []):
            return self._to_numpy(tm.get_term("success")).astype(bool)
        return np.asarray(sim_terminated, dtype=bool)

    def _reset_episode_state(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        env_ids = np.asarray(env_ids, dtype=np.int64)
        self._stable_actions[env_ids] = 0.0
        if self.embodiment.base_height_index is not None:
            self._stable_actions[env_ids, self.embodiment.base_height_index] = self.embodiment.base_height_command

    ### BaseEnv hooks ###

    @override
    def env_reset(
        self,
        *,
        env_ids,
        reset_eval: bool = False,
    ):
        del reset_eval
        env_ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        reset_env_ids = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)
        raw_obs, _info = self._raw_env.reset(env_ids=reset_env_ids)
        del _info
        self._reset_episode_state(env_ids)
        obs = self._make_obs(raw_obs, env_ids=env_ids)
        if not self.use_policy_action:
            self._update_stable_actions_from_obs(obs["observation"], env_ids)
        return obs

    @override
    def env_step(self, action, *, env_ids):
        env_ids = np.asarray(env_ids, dtype=np.int64)
        # Policy action -> sim action via the embodiment (identity 50->50 for G1 WBC,
        # 26->36 joint scatter for GR1, pose passthrough+reorder for Franka LIBERO).
        sim_action = self.embodiment.policy_to_sim_action(action, self.device)

        # IsaacLab owns auto-reset: ``ManagerBasedRLEnv.step`` computes reward +
        # terminated (success | failure) + truncated (time_out), resets the done envs
        # in-step, and returns the post-reset obs for them. We pass those signals
        # straight through; ``next.success`` is the sim's dedicated success term.
        raw_obs, reward, sim_terminated, sim_truncated, _info = self._raw_env.step(sim_action)
        del _info
        step_reward = self._to_numpy(reward).astype(np.float32)
        terminations = self._to_numpy(sim_terminated).astype(bool)
        timeouts = self._to_numpy(sim_truncated).astype(bool)
        successes = self._extract_success(terminations)

        obs = self._make_obs(raw_obs, env_ids=env_ids)
        return {
            "observation": obs["observation"],
            "task": obs["task"],
            "task_id": obs["task_id"],
            "next.reward": to_tensor(step_reward),
            "next.terminated": to_tensor(terminations),
            "next.truncated": to_tensor(timeouts),
            "next.success": to_tensor(successes),
        }

    # Stable-action adapter: temporarily replace policy actions with a held pose.

    @override
    def step_with_teleop_and_recording(self, action, chunk_intervened, merged_step_result, critic_value=None):
        if not self.use_policy_action:
            action = self._replace_with_stable_actions(action)
        return super().step_with_teleop_and_recording(
            action,
            chunk_intervened=chunk_intervened,
            merged_step_result=merged_step_result,
            critic_value=critic_value,
        )

    def _replace_with_stable_actions(self, action) -> np.ndarray:
        action = np.asarray(action).copy()
        n = min(self.num_envs, action.shape[0])
        action[:n] = self._stable_actions[:n]
        return action

    @override
    def apply_teleop_action(self, action):
        action = action if self.use_policy_action else self._replace_with_stable_actions(action)
        action, intervention_mask, manual_reward, restart_episode, stop_episode = super().apply_teleop_action(action)
        if not self.use_policy_action:
            hold_slice = self.embodiment.stable_hold_joint_slice
            base_height_index = self.embodiment.base_height_index
            if hold_slice is not None:
                self._stable_actions[intervention_mask, :hold_slice] = action[intervention_mask, :hold_slice]
            if base_height_index is not None:
                self._stable_actions[intervention_mask, base_height_index] = action[
                    intervention_mask, base_height_index
                ]
        return action, intervention_mask, manual_reward, restart_episode, stop_episode

    def _update_stable_actions_from_obs(self, observations: list[dict[str, Any]], env_ids: np.ndarray) -> None:
        hold_slice = self.embodiment.stable_hold_joint_slice
        if hold_slice is None:
            return
        for obs, env_id in zip(observations, np.asarray(env_ids, dtype=np.int64), strict=True):
            state = np.asarray(obs["observation.state"], dtype=np.float32)
            self._stable_actions[int(env_id), :hold_slice] = state[:hold_slice]

    # End stable-action adapter.

    @override
    def env_close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.app is not None:
            self.app.close()
            self.app = None

    @override
    def get_recorder_strategy_kwargs(self) -> dict[str, Any]:
        # The recorder logs the POLICY action passed to env.step (26-DOF for GR1),
        # NOT the scattered sim action (self.action_dim, overwritten to the 36-DOF
        # sim width in _init_env). Prefer the embodiment's policy action width and
        # fall back to the sim action_dim for identity embodiments (G1: unchanged).
        recorder_action_dim = self.embodiment.policy_action_dim or self.action_dim
        return {
            "camera_names": tuple(self.camera_names),
            "image_shape": self._image_shape(),
            "state_dim": self.state_dim,
            "action_dim": recorder_action_dim,
            "fps": int(self.cfg.recorder.video.fps),
            "robot_type": self.arena_cfg.embodiment,
        }

    ### Observation formatting ###

    def _make_obs(self, raw_obs, *, env_ids):
        observations = self._make_observations(raw_obs, env_ids=env_ids)
        tasks = [self.task_description] * len(observations)
        task_id = np.zeros(len(observations), dtype=np.int64)
        return {"observation": observations, "task": tasks, "task_id": task_id}

    def _make_observations(self, raw_obs, *, env_ids) -> list[dict[str, Any]]:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        # Delegate embodiment-specific extraction. G1 WBC reproduces the previous
        # wrapper behaviour bit for bit; GR1/Franka gather joints / read camera_obs.
        scene = getattr(self._raw_env, "scene", None)
        camera_images = self.embodiment.extract_images(raw_obs)
        state = self.embodiment.extract_state(raw_obs, scene)
        self.state_dim = int(state.shape[-1])

        observations = []
        for env_id in env_ids:
            item = {key: value[env_id] for key, value in camera_images.items()}
            item["observation.state"] = state[env_id].astype(np.float32)
            observations.append(item)
        return observations

    def _image_shape(self) -> tuple[int, int, int]:
        if self.enable_cameras:
            return self.arena_cfg.image_shape
        return (1, 1, 3)
