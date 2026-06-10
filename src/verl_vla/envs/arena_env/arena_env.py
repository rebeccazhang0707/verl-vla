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

"""IsaacLabArenaEnv: Isaac Lab Arena env wrapped as a verl-compatible gym env.

GR00T data channel (scheme Y — env owns packing + decoding, see
``docs/MIGRATION_gr00t_arena.md``):

  * **obs packing** — ``_wrap_obs`` runs ``GR00TN16Adapter.build_inputs`` and
    returns the model-ready eagle tensors (``images / lang_tokens / lang_masks /
    states``) as the ``images_and_states`` keys, so the pipeline maps them to
    ``obs.*`` → ``t0.obs.*`` exactly like the pi05 channel (no rollout-side
    packing; ``Gr00tN1d6ForSAC._obs_to_state_dict`` reads them unchanged).
  * **action decoding** — ``chunk_step`` decodes the **whole normalised chunk**
    (max_action_dim) into policy joints via
    ``GR00TN16Adapter.decode_actions_flat`` **once, against a single base state
    captured at chunk start** (the obs that fed this chunk's inference). This
    replicates the source rollout (``sac/naive_rollout_gr00t.py``), which decodes
    the full horizon against a single ``raw_state_groups`` (T=1, broadcast over
    the chunk). For relative-action checkpoints (``use_relative_action=true``)
    this yields ``base + δ[i]`` per step; decoding per-step against the *live*
    joint state would instead give ``base + δ[i-1] + δ[i]`` (offsets accumulate,
    joint targets diverge → policy learns on a corrupted action space). The
    per-step ``step`` then only scatters the already-decoded policy joints into
    the sim joints. The rollout only emits the normalised action, so replay/critic
    operate in the normalised space.

Joint-space conversion (derived from the embodiment config YAMLs) is owned by
the ``ArenaJointMapping`` / embodiment definition in ``embodiment.py`` and
reached via ``self.joint_map``:
  full robot state → policy state    (``joint_map.gather_state``)
  policy actions   → sim actions     (``joint_map.scatter_action``)
"""

import argparse
import logging
import os
from collections import OrderedDict
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from verl_vla.envs.arena_env.embodiment import ArenaJointMapping
from verl_vla.envs.arena_env.utils import (
    apply_rl_reward_and_disable_autoreset,
    build_env_cfg_without_recorder,
    disable_lightwheel_ssl_verify,
)
from verl_vla.models.gr00t.gr00t_policy import GR00TN16Adapter
from verl_vla.models.gr00t.utils import get_embodiment_spec, split_flat_state_to_groups
from verl_vla.utils.envs.action import to_tensor

logger = logging.getLogger(__name__)


class IsaacLabArenaEnv(gym.Env):
    """Isaac Lab Arena env wrapped as a verl-compatible gymnasium env.

    Mirrors the ``IsaacEnv`` interface so it can be used as a drop-in
    replacement via ``simulator_type: arena`` in the verl config.
    """

    def __init__(self, cfg, rank: int, world_size: int, stage_id: int = 0):
        # Arena fetches scene/object USDs from the lightwheel registry on env build; its
        # SDK doesn't disable TLS verification, so an expired server cert breaks the load.
        # Patch here (per worker construction) before anything can touch the registry.
        disable_lightwheel_ssl_verify()

        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.seed = cfg.seed + rank
        self.num_envs = cfg.num_envs
        # GR00T model-side embodiment (the policy's joint groups / action width). The
        # joint_map (gather_state / scatter_action) is derived from this spec so it
        # always tracks ``embodiment_tag`` instead of a hardcoded GR1 singleton.
        self.embodiment_tag = cfg.get("embodiment_tag", "gr1")
        self.embodiment_spec = get_embodiment_spec(self.embodiment_tag)
        self.joint_map = ArenaJointMapping.from_spec(self.embodiment_spec)
        self.action_dim = cfg.get("action_dim", self.embodiment_spec.action_dim)
        self.device = cfg.get("device", "cuda:0")
        self.camera_name = cfg.get("camera_name", "robot_pov_cam_rgb")
        self.arena_env_name = cfg.get("arena_env_name", "put_item_in_fridge_and_close_door")
        self.arena_object = cfg.get("arena_object", "ranch_dressing_hope_robolab")
        # GR00T outputs joint positions -> use the joint-control embodiment.
        # "gr1_pink" is Pink-IK (end-effector) control and is NOT compatible with joint targets.
        self.arena_embodiment = cfg.get("arena_embodiment", "gr1_joint")
        self.arena_object_set = cfg.get("object_set", None)
        self.kitchen_style = cfg.get("kitchen_style", 2)
        self.task_description = cfg.get(
            "task_description",
            "Place the sauce bottle on the top shelf of the fridge, and close the fridge door.",
        )

        # Asymmetric actor-critic: privileged critic-only obs. This wrapper is task-agnostic, so
        # the obs itself is DECLARED IN THE ARENA TASK ENV CFG as a dedicated `critic_privileged`
        # observation group (see gr1_put_and_close_door_environment.py: object pose in the fridge-
        # shelf frame + door joint). Here we only pass that group through to the critic; no scene
        # handles are resolved in the wrapper. The group name must match the task cfg.
        self.critic_privileged_obs = bool(cfg.get("critic_privileged_obs", False))
        self.PRIV_OBS_GROUP = str(cfg.get("priv_obs_group", "critic_privileged"))
        # Resolved from the task's obs group via the ObservationManager after the env is built
        self.PRIV_OBS_DIM = None
        self._priv_warned = False

        # GR00T N1.6 processor adapter: packs obs (build_inputs) and decodes the
        # normalised action chunk (decode_actions_flat). Built once per worker from
        # the checkpoint path; loads the HF processor (no Isaac dependency).
        self.gr00t_model_path = cfg.get("gr00t_model_path", None) or cfg.get("model_path", None)
        self.adapter = GR00TN16Adapter(self.gr00t_model_path, embodiment_tag=self.embodiment_tag)

        # Graded subtask reward (0/0.5/1.0) vs single composite +1; drives the
        # full-success threshold used for done / success_once (see _success_reward_thresh).
        self.subtask_reward = bool(cfg.get("subtask_reward", False))

        self._generator = np.random.default_rng(seed=self.seed)
        self.env = None
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = bool(cfg.get("use_rel_reward", False))
        # Latest raw policy-order joint state; the action decoder converts the model's
        # relative action to absolute joints against this live state.
        self._last_policy_state = np.zeros((self.num_envs, self.embodiment_spec.action_dim), dtype=np.float32)
        self._last_full_image = None

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.max_episode_steps = cfg.max_episode_steps
        self.video_cfg = cfg.video_cfg

        self.render_images = []
        self.video_cnt = 0

        # Render-performance: the policy only consumes the camera image at chunk
        # boundaries (the obs returned from chunk_step feeds the next inference);
        # intermediate per-step images are discarded. Setting render_interval to
        # span a whole chunk (decimation * chunk_size physics substeps) makes the
        # RTX camera render exactly once per chunk instead of every render_interval
        # physics steps (e.g. 32x fewer renders at decimation=4, chunk=16,
        # render_interval=2). Disable when recording smooth video (save_video).
        self.render_on_chunk_boundary = bool(cfg.get("render_on_chunk_boundary", True))
        self._chunk_render_interval_set = False

        # Isaac Sim must be launched before any isaaclab imports
        from isaaclab.app import AppLauncher

        launch_args = {"headless": True, "enable_cameras": True}
        app_launcher = AppLauncher(**launch_args)
        self.app = app_launcher.app

    # ------------------------------------------------------------------
    # Env initialisation
    # ------------------------------------------------------------------

    def _build_args(self) -> argparse.Namespace:
        """Build the argparse.Namespace that ArenaEnvBuilder expects."""
        args = argparse.Namespace(
            num_envs=self.num_envs,
            env_spacing=float(self.cfg.get("env_spacing", 30.0)),
            disable_fabric=bool(self.cfg.get("disable_fabric", False)),
            device=self.device,
            seed=self.seed,
            solve_relations=bool(self.cfg.get("solve_relations", True)),
            mimic=False,
            enable_pinocchio=bool(self.cfg.get("enable_pinocchio", True)),
            # Relation-placement args read by ArenaEnvBuilder._solve_relations()
            # and compose_manager_cfg(). Defaults mirror the Arena CLI parser.
            placement_seed=self.cfg.get("placement_seed", None),
            resolve_on_reset=self.cfg.get("resolve_on_reset", None),
            presets=self.cfg.get("presets", None),
            # environment-specific args consumed by ExampleEnvironment.get_env()
            object=self.arena_object,
            embodiment=self.arena_embodiment,
            enable_cameras=True,
            teleop_device=None,
            # put_item_in_fridge_and_close_door.get_env() reads these two
            # (GR1PutAndCloseDoorEnvironment.add_cli_args defaults: kitchen_style=2,
            #  object_set=None). Other Arena envs ignore them harmlessly.
            kitchen_style=int(self.kitchen_style),
            object_set=self.arena_object_set,
        )
        return args

    def _init_env(self) -> None:
        """Launch or re-launch the Arena gym environment."""
        # Arena 0.2.0 collects the example environments in a plain dict
        # (the old EnvironmentRegistry / ensure_environments_registered API was removed).
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena_environments.cli import ExampleEnvironments

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            import omni

            omni.usd.get_context().new_stage()

        # Re-assert the lightwheel TLS workaround (idempotent) right before the env build
        # actually hits the registry, in case __init__ ran in a different process.
        disable_lightwheel_ssl_verify()

        args = self._build_args()

        # Look up the example environment class by its ``name`` and instantiate it.
        if self.arena_env_name not in ExampleEnvironments:
            raise ValueError(
                f"Arena env '{self.arena_env_name}' not found. "
                f"Available: {sorted(ExampleEnvironments.keys())}"
            )

        example_env = ExampleEnvironments[self.arena_env_name]()
        arena_env = example_env.get_env(args)

        # Try to obtain task description from the task object
        if hasattr(arena_env, "task") and arena_env.task is not None:
            task = arena_env.task
            if hasattr(task, "get_task_description"):
                desc = task.get_task_description()
                if desc:
                    self.task_description = desc

        env_builder = ArenaEnvBuilder(arena_env, args)

        # Build the env cfg first so we can patch it before instantiation (disable demo
        # HDF5 recording to avoid h5py file-lock clashes across parallel workers).
        env_cfg = build_env_cfg_without_recorder(env_builder)
        # Turn composite-success into an RL reward + disable auto-reset (LIBERO-style).
        if bool(self.cfg.get("rl_success_reward", True)):
            apply_rl_reward_and_disable_autoreset(env_cfg, subtask_reward=self.subtask_reward)
        self.env = env_builder.make_registered(env_cfg=env_cfg)

        # Resolve the privileged-obs width from the task's declared obs group now that the
        # ObservationManager exists (keeps critic_input_dim in sync with the task cfg).
        if self.critic_privileged_obs:
            self._resolve_priv_obs_dim()

        if self.cfg.video_cfg.save_video:
            video_dir = os.path.join(self.cfg.video_cfg.video_base_dir, f"rank_{self.rank}")
            os.makedirs(video_dir, exist_ok=True)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        logger.info("Arena Isaac Sim environment initialised")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def _success_reward_thresh(self) -> float:
        """Reward above which the task counts as FULLY solved.

        With the graded subtask reward, a value in (0, 1) is a partial milestone, so
        full success requires the composite reward (>= 1.0). With the plain success
        reward (0/1) this reduces to ``> 0``. Used for both ``done`` (chunk_step) and
        ``success_once`` (_record_metrics) so the two stay consistent.
        """
        return 1.0 - 1e-6 if self.subtask_reward else 0.0

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.returns[mask] = 0.0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0.0
            self.success_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | (step_reward > self._success_reward_thresh)
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        if any(self.elapsed_steps > 0):
            episode_info["reward"] = episode_info["return"] / self.elapsed_steps
        else:
            episode_info["reward"] = 0
        infos["episode"] = to_tensor(episode_info)
        return infos

    # ------------------------------------------------------------------
    # Step / Reset
    # ------------------------------------------------------------------

    def reset(self, env_idx=None, options: Optional[dict] = None):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        raw_obs, infos = self.env.reset()
        obs = self._wrap_obs(raw_obs)
        self._reset_metrics(env_idx)
        return obs, infos

    def step(self, actions=None, critic_values=None):
        if actions is None:
            return (None, None, None, None, None)

        truncations = self.elapsed_steps >= self.max_episode_steps

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        # ``actions`` is the already-decoded 26-DOF policy joint action for this
        # step. Action decoding (normalised → absolute joints) is owned by
        # ``chunk_step`` (one fixed-base decode of the whole chunk); ``step`` only
        # expands 26 → 36 sim joints. See the module docstring for why decoding
        # per-step against the live state would corrupt relative-action targets.
        policy_action = actions  # (B, 26)
        sim_actions = self.joint_map.scatter_action(policy_action)  # (B, 36)

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, _, infos = self.env.step(sim_actions)

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward.cpu().numpy())
        infos = self._record_metrics(step_reward, infos)

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": self.success_once,
                "task": [self.task_description] * self.num_envs,
            }
            if critic_values is not None:
                plot_infos["critic_value"] = np.asarray(critic_values, dtype=np.float32)
            self.add_new_frames(obs, plot_infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def _set_chunk_render_interval(self, chunk_size: int) -> None:
        """Render the RTX camera only once per action chunk (one-shot setup).

        IsaacLab renders inside ``ManagerBasedRLEnv.step`` whenever
        ``_sim_step_counter % cfg.sim.render_interval == 0`` (physics-step units).
        A chunk spans ``decimation * chunk_size`` physics substeps, so setting
        ``render_interval`` to that value makes the camera refresh exactly on the
        last substep of each chunk -- i.e. right before chunk_step returns the obs
        that feeds the next policy inference. Intermediate images are unused, so
        this is lossless for the policy while cutting renders by a large factor.
        ``step()`` reads ``cfg.sim.render_interval`` live every substep, so doing
        this once (before the first chunk runs) is enough.
        """
        if self._chunk_render_interval_set or not self.render_on_chunk_boundary:
            return
        base = getattr(self.env, "unwrapped", self.env)
        decimation = int(base.cfg.decimation)
        new_interval = decimation * int(chunk_size)
        old_interval = int(base.cfg.sim.render_interval)
        base.cfg.sim.render_interval = new_interval
        self._chunk_render_interval_set = True
        logger.info(
            f"[render-perf] render_interval {old_interval} -> {new_interval} "
            f"(decimation={decimation} x chunk_size={chunk_size}); RTX camera now "
            f"renders once per action chunk."
        )
        if self.video_cfg.save_video:
            logger.warning(
                "[render-perf] save_video=True with render_on_chunk_boundary=True: "
                "recorded video will show one fresh frame per chunk (choppy). Set "
                "render_on_chunk_boundary=False for smooth video."
            )

    def chunk_step(self, chunk_actions, chunk_values=None):
        """Execute a chunk of actions, tracking first-done semantics.

        The whole **normalised** chunk is decoded into 26-DOF policy joints **once**
        here, against a single base state captured at chunk entry (``_last_policy_state``
        = the obs that fed this chunk's inference; ``env_loop`` runs exactly one
        inference per chunk). This mirrors the source rollout's single fixed-base
        ``decode_actions_flat`` over the full horizon and is required for
        relative-action checkpoints (see the module docstring): per-step decoding
        against the live state would accumulate offsets and diverge. ``_wrap_obs``
        keeps updating ``_last_policy_state`` so the *next* chunk decodes against its own
        chunk-start state.

        Args:
            chunk_actions: (num_envs, chunk_size, action_dim) — normalised actions.
        """
        chunk_size = chunk_actions.shape[1]
        self._set_chunk_render_interval(chunk_size)

        # Fix the base once and decode the whole chunk → (B, chunk_size, 26).
        decoded_chunk = self._decode_chunk_to_policy_actions(chunk_actions)

        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        ever_done = torch.zeros(self.num_envs, dtype=torch.bool)

        for i in range(chunk_size):
            actions = decoded_chunk[:, i]  # already-decoded 26-DOF policy joints
            step_values = None
            if chunk_values is not None:
                if chunk_values.ndim == 1:
                    step_values = chunk_values
                elif chunk_values.ndim == 2:
                    step_values = chunk_values[:, i]

            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, critic_values=step_values
            )

            reward_val = (
                step_reward.cpu()
                if isinstance(step_reward, torch.Tensor)
                else torch.as_tensor(step_reward)
            )
            # Full-success threshold (subtask mode: reward in (0,1) is partial, not done).
            ever_done = ever_done | (reward_val > self._success_reward_thresh).view(-1)
            raw_chunk_terminations.append(ever_done.clone())
            chunk_rewards.append(step_reward)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        return extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos

    # ------------------------------------------------------------------
    # Action decoding (normalised model action → policy-order joints)
    # ------------------------------------------------------------------

    def _build_raw_state_groups(self) -> "OrderedDict[str, np.ndarray]":
        """Per-modality raw (un-normalised) joint state {group: (B, 1, d)}.

        Built from the cached policy-order joint state ``_last_policy_state``.
        Called once per chunk (from ``chunk_step``), so this is the chunk-start
        (inference) state — the single ``base`` against which ``decode_action``
        converts the whole relative-action chunk to absolute joints. The group
        split (left_arm / right_arm / left_hand / right_hand) comes from the
        embodiment spec bound to ``self.joint_map``.
        """
        groups = split_flat_state_to_groups(self._last_policy_state, self.joint_map.spec.state_group_dims)
        return OrderedDict(
            (key, value.reshape(self.num_envs, 1, -1).astype(np.float32)) for key, value in groups.items()
        )

    def _decode_chunk_to_policy_actions(self, chunk_actions) -> torch.Tensor:
        """Decode a whole normalised chunk (B, chunk, max_action_dim) → (B, chunk, 26).

        The base ``raw_state_groups`` is built **once** from the chunk-start joint
        state (``_last_policy_state``) and broadcast across the whole horizon — exactly
        the source rollout's single ``decode_actions_flat(full_chunk, base_groups)``
        call (``raw_state_groups`` T=1). For relative-action checkpoints this gives
        ``base + δ[i]`` per step (NOT ``base + δ[i-1] + δ[i]`` as live-state per-step
        decoding would). Returns a torch tensor so ``joint_map.scatter_action``
        (new_zeros + index assign) works downstream.
        """
        actions_np = (
            chunk_actions.detach().cpu().numpy()
            if isinstance(chunk_actions, torch.Tensor)
            else np.asarray(chunk_actions)
        )
        base_groups = self._build_raw_state_groups()  # fixed base from chunk-start obs
        decoded = self.adapter.decode_actions_flat(actions_np, base_groups)  # (B, chunk, 26)
        decoded = np.asarray(decoded, dtype=np.float32)
        return torch.from_numpy(decoded)

    # ------------------------------------------------------------------
    # Observation wrapping
    # ------------------------------------------------------------------

    def _wrap_obs(self, raw_obs) -> dict:
        """Pack raw obs into the model-ready GR00T eagle tensors (scheme Y).

        The packed tensors are returned as the ``images_and_states`` keys so the
        pipeline maps them to ``obs.images / obs.lang_tokens / obs.lang_masks /
        obs.states`` (then ``t0.obs.*``) — the exact slots
        ``Gr00tN1d6ForSAC._obs_to_state_dict`` reads. ``full_image`` is kept at the
        top level for video only (``create_env_batch_dataproto`` ignores it).
        """
        full_image, policy_state = self._extract_image_and_state(raw_obs)
        # Cache the live raw joint state for the next step's action decode.
        self._last_policy_state = policy_state
        self._last_full_image = full_image
        task_descriptions = [self.task_description] * self.num_envs

        inputs, _ = self.adapter.build_inputs(full_image, policy_state, task_descriptions)
        # Eagle pixel_values is a per-sample list of (n_patches, C, H, W) tensors;
        # stack to (B, n_patches, C, H, W) so the replay buffer can store it.
        pixel_values = inputs["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        packed = {
            "images": pixel_values,
            "lang_tokens": inputs["input_ids"],
            "lang_masks": inputs["attention_mask"],
            "states": inputs["state"],
        }
        # Asymmetric-AC privileged critic obs (arena only): ride the images_and_states dict so
        # the generic pipeline maps it to obs.priv_obs → t0.obs.priv_obs and the critic forward
        # consumes it (Gr00tN1d6ForSAC._obs_to_state_dict). Critic-only; the actor never sees it.
        if self.critic_privileged_obs:
            packed["priv_obs"] = self._extract_priv_obs(raw_obs)  # (B, PRIV_OBS_DIM) float32
        obs = {
            "images_and_states": to_tensor(packed),
            "task_descriptions": task_descriptions,
            # video only; not consumed by create_env_batch_dataproto / replay.
            "full_image": full_image,
        }
        return obs

    # ------------------------------------------------------------------
    # Privileged critic observation (asymmetric actor-critic)
    # ------------------------------------------------------------------

    def _resolve_priv_obs_dim(self) -> None:
        """Resolve the privileged-obs width from the task-declared obs group.

        The ``critic_privileged`` group is defined in the Arena task env cfg, so its
        width is a property of the task, not of this wrapper. We query the live
        ``ObservationManager`` (``group_obs_dim``) instead of hardcoding the dim, so the
        critic input width always tracks the task cfg. Falls back to the
        ``env.train.priv_obs_dim`` override (and warns) if the group can't be resolved.
        """
        base = getattr(self.env, "unwrapped", self.env)
        obs_mgr = getattr(base, "observation_manager", None)
        group_dims = getattr(obs_mgr, "group_obs_dim", None) if obs_mgr is not None else None

        dim = None
        if isinstance(group_dims, dict) and self.PRIV_OBS_GROUP in group_dims:
            shape = group_dims[self.PRIV_OBS_GROUP]
            if shape is not None:
                dim = int(np.prod(np.asarray(shape)))

        if not dim or dim <= 0:
            dim = int(self.cfg.get("priv_obs_dim", 8))
            logger.warning(
                "[arena_env][priv] could not resolve obs group '%s' dim from "
                "ObservationManager (available=%s); falling back to %d",
                self.PRIV_OBS_GROUP,
                list(group_dims.keys()) if isinstance(group_dims, dict) else None,
                dim,
            )
        else:
            logger.info(
                "[arena_env][priv] resolved obs group '%s' dim = %d from ObservationManager",
                self.PRIV_OBS_GROUP,
                dim,
            )
        self.PRIV_OBS_DIM = dim

    def _extract_priv_obs(self, raw_obs) -> np.ndarray:
        """Read the critic-only privileged obs from the task's dedicated obs group.

        The obs is DECLARED in the Arena task env cfg as the ``critic_privileged``
        observation group (object position in the fridge-shelf frame + door joint), so here
        we just pull that group out of the raw observation dict and hand it to the critic.
        Missing/short groups are zero-filled to keep ``critic_input_dim`` fixed.
        """
        out = np.zeros((self.num_envs, self.PRIV_OBS_DIM), dtype=np.float32)
        group = raw_obs.get(self.PRIV_OBS_GROUP) if isinstance(raw_obs, dict) else None
        # concatenate_terms=True ⇒ a single (B, D) tensor; tolerate a {term: tensor} dict too.
        if isinstance(group, dict):
            parts = [v for v in group.values()]
            group = (
                torch.cat([p if isinstance(p, torch.Tensor) else torch.as_tensor(p) for p in parts], dim=-1)
                if parts
                else None
            )
        if group is None:
            if not self._priv_warned:
                self._priv_warned = True
                logger.warning(
                    "[arena_env][priv] obs group '%s' not found in env observation (keys=%s); "
                    "priv_obs zero-filled. Ensure the arena task cfg declares it.",
                    self.PRIV_OBS_GROUP,
                    list(raw_obs.keys()) if isinstance(raw_obs, dict) else type(raw_obs),
                )
            return out
        priv = (
            group.detach().float().cpu().numpy()
            if isinstance(group, torch.Tensor)
            else np.asarray(group, dtype=np.float32)
        )
        d = min(priv.shape[-1], self.PRIV_OBS_DIM)
        out[:, :d] = priv[:, :d]
        if not self._priv_warned:
            self._priv_warned = True
            door_joint = float(out[0, -1]) if out.shape[-1] > 0 else float("nan")
            logger.info(
                "[arena_env][priv] obs group '%s' wired: obs[0]=%s shape=%s relpos_norm=%.3f doorjoint=%.4f",
                self.PRIV_OBS_GROUP,
                np.array2string(out[0], precision=3, max_line_width=200),
                out.shape,
                float(np.linalg.norm(out[0, 0:3])),
                door_joint,
            )
        return out

    def _extract_image_and_state(self, obs) -> "tuple[np.ndarray, np.ndarray]":
        """Return ``(full_image (B,H,W,C) uint8, policy_state (B, policy_dim) float32)``."""
        # --- image ---
        camera_obs = obs.get("camera_obs", {})
        if self.camera_name in camera_obs:
            rgb = camera_obs[self.camera_name]
        else:
            # Fallback: try any available camera
            available = list(camera_obs.keys())
            if available:
                rgb = camera_obs[available[0]]
                logger.warning(
                    f"Camera '{self.camera_name}' not found; using '{available[0]}'"
                )
            else:
                raise KeyError(
                    f"Camera '{self.camera_name}' not found in camera_obs. "
                    f"Available: {list(camera_obs.keys())}"
                )

        if isinstance(rgb, torch.Tensor):
            full_image = rgb.cpu().numpy()  # (B, H, W, C) uint8
        else:
            full_image = rgb

        # --- state: gather the policy-order joints from the full robot_joint_pos ---
        robot_joint_pos = obs["policy"]["robot_joint_pos"]  # (B, state_full_dim+)
        if isinstance(robot_joint_pos, torch.Tensor):
            robot_joint_pos_np = robot_joint_pos.cpu().numpy()
        else:
            robot_joint_pos_np = np.asarray(robot_joint_pos)

        policy_state = self.joint_map.gather_state(robot_joint_pos_np)  # (B, policy_dim)
        return full_image, policy_state.astype(np.float32)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _calc_step_reward(self, reward):
        if self.use_rel_reward:
            diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return diff
        return reward

    def add_new_frames(self, obs, plot_infos):
        from verl_vla.utils.envs.action import put_info_on_image, tile_images

        images = []
        for env_id, img in enumerate(obs["full_image"]):
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            info_item = {k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()}
            img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=max(1, int(np.sqrt(self.num_envs))))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        from verl_vla.utils.envs.action import save_rollout_video

        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, video_sub_dir)
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
            # Real-time playback = 1 / (sim_dt * decimation) = 50 Hz for the Arena defaults
            # (dt=1/200, decimation=4); one frame is captured per env.step().
            fps=int(getattr(self.video_cfg, "fps", 50)),
        )
        self.video_cnt += 1
        self.render_images = []

    def close(self):
        if self.env is not None:
            self.env.close()
            self.app.close()

    def get_state(self):
        return None

    # ------------------------------------------------------------------
    # State-ID based reset (required by EnvManager interface)
    # ------------------------------------------------------------------

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        logger.info(
            f"IsaacLabArenaEnv reset_envs_to_state_ids: "
            f"state_ids={state_ids_list[:4]}... task_ids={task_ids_list[:4]}..."
        )
        # Re-initialise the Arena environment (handles task switching)
        self._init_env()
        raw_obs, infos = self.env.reset()
        self._reset_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        obs = self._wrap_obs(raw_obs)
        return obs, infos

    # ------------------------------------------------------------------
    # State ID helpers (used by EnvWorker.get_all_state_ids)
    # ------------------------------------------------------------------

    def get_all_state_ids(self) -> list[int]:
        """Return a list of dummy state IDs (Arena uses random resets)."""
        return list(range(self.num_envs))
