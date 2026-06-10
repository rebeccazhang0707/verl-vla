# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Helpers for IsaacLabArenaEnv.

Three independent concerns live here so the env wrapper / embodiment modules stay
focused on their own logic:
  * RL success reward terms + env-cfg patching (auto-reset / recorder).
  * lightwheel asset-registry TLS workaround.
  * embodiment ⇄ Arena-sim joint-space YAML discovery + parsing (consumed by
    ``embodiment.ArenaJointMapping``).
"""

import importlib.util
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from verl_vla.models.gr00t.utils import JointSpaceYamls

logger = logging.getLogger(__name__)


def arena_task_success_reward(env, success_func, success_params):
    """RL reward = Arena composite-task success (+1.0 the step the whole task is solved).

    Reuses the sequential-task success function so the *stateful* subtask state
    machine (``_current_subtask_idx`` / ``_subtask_success_state`` /
    ``env.extras['subtask_success_state']``) is advanced exactly once per step --
    it now runs inside the RewardManager instead of the (removed) TerminationManager.
    verl's ``chunk_step`` / ``_record_metrics`` derive ``done`` / ``success_once``
    from ``reward > 0``, so this term is what makes a success visible to training
    (the raw Arena task defines no reward term at all).
    """
    success = success_func(env, **success_params)
    return success.float()


def arena_subtask_graded_reward(env, success_func, success_params):
    """Graded RL reward for a SEQUENTIAL task = fraction of subtasks completed.

    Returns 0 / 0.5 / 1.0 (for a 2-subtask task) — the latched progress, so e.g. +0.5 once
    pick-and-place is done and +1.0 once the door is also closed. ``success_func`` (the
    sequential-task success fn) is still called first so it advances the subtask state machine
    and writes ``env.extras['subtask_success_state']`` (a per-env list of per-subtask latched
    bools); we read that to compute the graded progress. This gives the long-horizon task an
    early (PnP) learning signal instead of a single composite +1 only after BOTH subtasks.

    NB: with this reward, ``reward > 0`` no longer means full success, so ``chunk_step`` must
    derive ``ever_done`` from ``reward >= 1.0`` (composite) — see arena_env.chunk_step.
    """
    import torch

    composite = success_func(env, **success_params)  # advances state machine + writes extras
    state = getattr(env, "extras", {}).get("subtask_success_state", None)
    if not state:
        return composite.float()
    progress = torch.tensor(
        [(sum(1 for x in s if x) / max(len(s), 1)) for s in state],
        device=composite.device,
        dtype=torch.float32,
    )
    return progress


def build_env_cfg_without_recorder(env_builder):
    """Build the Arena env cfg and disable demo HDF5 recording before instantiation.

    IsaacLab's RecorderManager creates an HDF5 dataset file on init (dataset_export_mode
    defaults to EXPORT_ALL) at /tmp/isaaclab/logs/dataset_<sec>_rank<local_rank>. With
    several parallel env workers / pipeline stages all sharing that path (same second,
    local_rank unset -> 0) h5py fails with "unable to lock file (errno 11)". We don't
    need demo HDF5s during RL, so force EXPORT_NONE -- the metric recorder terms still
    run in-memory (success/reward detection unaffected); only the file write is skipped.

    Returns the built ``env_cfg`` (ready to be patched further / handed to
    ``env_builder.make_registered``).
    """
    _, env_cfg = env_builder.build_registered()
    try:
        from isaaclab.managers.recorder_manager import DatasetExportMode

        recorders_cfg = getattr(env_cfg, "recorders", None)
        if recorders_cfg is not None and hasattr(recorders_cfg, "dataset_export_mode"):
            recorders_cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not disable Arena recorder file export: {exc}")
    return env_cfg


def apply_rl_reward_and_disable_autoreset(env_cfg, subtask_reward: bool = False) -> None:
    """Turn the Arena composite-success TERMINATION into a sparse RL REWARD and
    disable IsaacLab auto-reset (verl owns episode resets + horizon).

    Mirrors the LIBERO RL setup (franka_libero_rl_env_cfg + isaac_env.py):
      * success DoneTerm -> RewTerm(weight = 1 / step_dt), so ``reward > 0``
        marks success. verl's chunk_step derives ``done`` from reward > 0; the
        raw Arena task otherwise has no reward term, so a success would be
        invisible to training.
      * every termination term -> None, so reset_buf stays False (no auto-reset
        mid-rollout, which would corrupt fixed-length trajectories).

    RL-only: patches the env cfg built inside verl; the shared Arena task /
    eval / mimic configs are untouched. Gate off with
    ``env.train.rl_success_reward=False``.

    Args:
        env_cfg: the Arena env cfg to patch in place.
        subtask_reward: if True, use the graded subtask reward (0/0.5/1.0 = fraction
            of subtasks done) for earlier long-horizon credit, vs the single
            composite +1. Gate: ``env.train.subtask_reward``.
    """
    import dataclasses

    from isaaclab.managers import RewardTermCfg
    from isaaclab.utils import configclass

    term_cfg = getattr(env_cfg, "terminations", None)
    succ_term = getattr(term_cfg, "success", None) if term_cfg is not None else None
    if succ_term is None:
        logger.warning(
            "[arena_env] terminations.success not found; skipping RL success-reward patch"
        )
        return

    # step_dt = sim.dt * decimation (Arena default 1/200 * 4 = 0.02s -> 50 Hz).
    # RewardManager scales every term by step_dt, so weight = 1/step_dt emits
    # exactly +1.0 per step the task is solved (matches LIBERO weight=20 @ 0.05s).
    sim_dt = float(getattr(getattr(env_cfg, "sim", None), "dt", 1.0 / 200.0))
    decimation = int(getattr(env_cfg, "decimation", 4))
    step_dt = sim_dt * decimation
    weight = 1.0 / step_dt

    @configclass
    class _ArenaRLRewardsCfg:
        task_success: RewardTermCfg = None

    # Sequential-task option: graded subtask reward vs the single composite +1.
    reward_func = arena_subtask_graded_reward if subtask_reward else arena_task_success_reward

    rewards = _ArenaRLRewardsCfg()
    rewards.task_success = RewardTermCfg(
        func=reward_func,
        weight=weight,
        params={"success_func": succ_term.func, "success_params": succ_term.params},
    )
    env_cfg.rewards = rewards

    # Disable auto-reset: null every termination term (composite success now
    # lives in the reward above; subtask terms like object_dropped and any
    # time_out are dropped so verl controls when envs reset).
    disabled = [f.name for f in dataclasses.fields(term_cfg)]
    for name in disabled:
        setattr(term_cfg, name, None)

    logger.info(
        "[arena_env] RL patch: success->RewTerm weight=%.3f (step_dt=%.4fs); "
        "terminations disabled (%s) -> no auto-reset",
        weight,
        step_dt,
        ", ".join(disabled) or "none",
    )


_LIGHTWHEEL_SSL_PATCHED = False


def disable_lightwheel_ssl_verify() -> None:
    """Skip TLS cert verification for lightwheel asset-registry calls only.

    Arena loads the kitchen/object USDs from the lightwheel registry
    (``LW_API_ENDPOINT``, default the dev host). Its SDK calls ``requests`` with no
    ``verify=`` option, so an expired/invalid server cert makes every env's scene load
    die with ``SSLCertVerificationError: certificate has expired`` before the local
    asset cache is even consulted. Patch ``requests.Session.request`` to pass
    ``verify=False`` for lightwheel hosts (other hosts keep normal verification).
    Idempotent; assets themselves are integrity-checked by the cache, not the TLS cert.
    """
    global _LIGHTWHEEL_SSL_PATCHED
    if _LIGHTWHEEL_SSL_PATCHED:
        return
    try:
        import requests
        import urllib3

        _orig_request = requests.Session.request

        def _request(self, method, url, *args, **kwargs):
            if "lightwheel" in str(url):
                kwargs.setdefault("verify", False)
            return _orig_request(self, method, url, *args, **kwargs)

        requests.Session.request = _request
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        _LIGHTWHEEL_SSL_PATCHED = True
        logger.warning("Disabled TLS verification for lightwheel asset-registry requests (expired cert workaround)")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not patch lightwheel SSL verification: {exc}")


# ---------------------------------------------------------------------------
# Embodiment ⇄ Isaac Lab Arena joint-space YAML helpers
#
# Each Arena embodiment ships joint-space YAMLs that are the single source of truth
# for the GR00T-policy ⇄ sim index tables (see ``embodiment.py`` for how they are
# consumed). The YAML *filenames* are embodiment metadata and live on the spec
# (``EmbodimentSpec.joint_space_yamls``); these helpers only locate the dir and parse
# whichever files the spec points at:
#   policy: group -> [joint_name, ...]
#   action: joint_name -> column index (full sim action)
#   state:  joint_name -> column index (full sim state)
# ---------------------------------------------------------------------------


def _env_var_for(embodiment_tag: str) -> str:
    """Override env var holding an embodiment's joint-space dir (e.g. ``gr1`` -> ``ARENA_GR1_JOINT_SPACE_DIR``)."""
    return f"ARENA_{embodiment_tag.upper()}_JOINT_SPACE_DIR"


def find_arena_joint_space_dir(embodiment_tag: str) -> Optional[Path]:
    """Locate an Arena embodiment's joint-space YAML dir (without importing Arena).

    Discovery order:
      1. ``$ARENA_<TAG>_JOINT_SPACE_DIR`` (tag upper-cased)
      2. the installed ``isaaclab_arena_gr00t`` package (embodiments/<tag>)
    Returns ``None`` if neither is found.
    """
    env = os.environ.get(_env_var_for(embodiment_tag))
    if env and Path(env).is_dir():
        return Path(env)

    # Installed package location (use find_spec to avoid import side effects).
    try:
        spec = importlib.util.find_spec("isaaclab_arena_gr00t")
        if spec is not None and spec.submodule_search_locations:
            cand = Path(list(spec.submodule_search_locations)[0]) / "embodiments" / embodiment_tag
            if cand.is_dir():
                return cand
    except (ImportError, ValueError):
        pass

    return None


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def build_index_maps_from_yaml(
    joint_dir: "str | Path",
    yamls: "JointSpaceYamls",
    expected_group_dims: "Optional[OrderedDict]" = None,
) -> "tuple[list[int], list[int], int, int]":
    """Derive (state_full_to_policy, policy_to_action, sim_action_dim, state_full_dim).

    Replicates the name-based lookup Arena uses in ``joints_conversion``: flatten the
    policy groups (in YAML order) into the policy-order joint names, then look each
    name up in the state/action ``name -> index`` dicts.

    ``yamls`` carries the embodiment's joint-space YAML filenames (from
    ``EmbodimentSpec.joint_space_yamls``). If ``expected_group_dims`` (an ordered
    ``group -> dim`` mapping, e.g. ``spec.state_group_dims``) is provided, the YAML
    grouping is cross-checked against it and a warning is logged on mismatch (kept
    embodiment-agnostic — no global spec).
    """
    joint_dir = Path(joint_dir)
    policy_groups = _load_yaml(joint_dir / yamls.policy)["joints"]   # group -> [name]
    action_yaml = _load_yaml(joint_dir / yamls.action)
    state_yaml = _load_yaml(joint_dir / yamls.state)
    action_cfg = action_yaml["joints"]                              # name -> idx
    state_cfg = state_yaml["joints"]                                # name -> idx

    # Flatten policy groups (in YAML order) into the policy-order joint names.
    flat_names = [name for names in policy_groups.values() for name in names]

    # Sanity: YAML policy grouping must match the caller's EmbodimentSpec.
    if expected_group_dims is not None:
        yaml_group_dims = OrderedDict((g, len(names)) for g, names in policy_groups.items())
        if list(yaml_group_dims.items()) != list(expected_group_dims.items()):
            logger.warning(
                "Arena policy YAML groups %s differ from expected spec groups %s",
                dict(yaml_group_dims), dict(expected_group_dims),
            )

    state_indices = [state_cfg[name] for name in flat_names]
    action_map = [action_cfg[name] for name in flat_names]
    sim_action_dim = int(action_yaml.get("total_joints", len(action_cfg)))
    state_full_dim = int(state_yaml.get("total_joints", len(state_cfg)))
    return state_indices, action_map, sim_action_dim, state_full_dim


def resolve_joint_maps(
    embodiment_tag: str,
    yamls: "JointSpaceYamls",
    joint_dir: "str | Path | None" = None,
    expected_group_dims: "Optional[OrderedDict]" = None,
) -> "tuple[list[int], list[int], int, int]":
    """Discover an embodiment's joint-space dir (if not given) and build its index maps.

    ``embodiment_tag`` (e.g. ``"gr1"``) drives directory discovery; ``yamls`` are the
    embodiment's joint-space YAML filenames (``EmbodimentSpec.joint_space_yamls``) —
    the single source of truth (no hardcoded fallback). The embodiment ships them under
    ``isaaclab_arena_gr00t/embodiments/<tag>`` so discovery should always succeed; a
    clear ``RuntimeError`` is raised otherwise.
    """
    if joint_dir is None:
        joint_dir = find_arena_joint_space_dir(embodiment_tag)
        if joint_dir is None:
            raise RuntimeError(
                f"Arena {embodiment_tag!r} joint-space YAMLs not found; cannot derive joint maps. "
                f"Set {_env_var_for(embodiment_tag)} to the embodiments/{embodiment_tag} dir "
                f"(expected files: {yamls.policy}, {yamls.state}, {yamls.action})."
            )
    maps = build_index_maps_from_yaml(joint_dir, yamls, expected_group_dims=expected_group_dims)
    logger.info("Loaded %r joint-space maps from %s", embodiment_tag, joint_dir)
    return maps
