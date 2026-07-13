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

"""Helpers for IsaacLabArenaEnv."""

import logging

logger = logging.getLogger(__name__)


def arena_success_reward(env):
    """Sparse RL reward = the Arena ``success`` termination (+1.0 the step the task is solved).

    The raw Arena G1/GR1 task defines no reward term at all, so a success would
    otherwise be invisible to training. We simply READ the ``success`` termination the
    ``TerminationManager`` already computed this step: in ``ManagerBasedRLEnv.step``
    terminations run *before* rewards, so ``term_dones`` is fresh and we do NOT
    recompute (nor double-advance a stateful success). Scaled by the reward weight
    (``1/step_dt``) so exactly ``+1.0`` is emitted per solved step.
    """
    return env.termination_manager.get_term("success").float()


def arena_subtask_graded_reward(env):
    """Graded RL reward for a SEQUENTIAL task = fraction of subtasks completed (0 / 0.5 / 1.0).

    Reads the latched per-subtask state the ``success`` termination wrote to
    ``env.extras['subtask_success_state']`` this step (terminations run before rewards,
    so the state is fresh and not advanced twice). Gives long-horizon tasks an early
    (e.g. pick-and-place) credit signal instead of a single composite +1 only after
    BOTH subtasks. Falls back to the composite ``success`` term when the task exposes
    no subtask state.
    """
    import torch

    state = getattr(env, "extras", {}).get("subtask_success_state", None)
    if not state:
        return env.termination_manager.get_term("success").float()
    return torch.tensor(
        [(sum(1 for x in s if x) / max(len(s), 1)) for s in state],
        device=env.device,
        dtype=torch.float32,
    )


def apply_arena_rl_reward(env_cfg, subtask_reward: bool = False) -> bool:
    """Turn the Arena ``success`` termination into a sparse RL REWARD (reward half only).

    The raw Arena G1/GR1 task defines **no reward term at all**, so a success would
    otherwise be invisible to training. This mirrors the LIBERO RL setup
    (franka_libero_rl_env_cfg + isaac_env.py): install a ``RewTerm(weight = 1/step_dt)``
    that reads the ``success`` termination so exactly ``+1.0`` is emitted per solved
    step (or graded 0/0.5/1.0 subtask progress when ``subtask_reward``).

    Crucially this ONLY adds the reward -- the termination terms are LEFT IN PLACE, so
    IsaacLab keeps owning per-step episode auto-reset (see ``docs/mdp_auto_reset.md``).
    On a success step the reward fires ``+1`` AND IsaacLab flags ``terminated`` and
    resets the env; the reward func just reads ``term_dones`` (terminations run before
    rewards in the sim step) so nothing is double-computed.

    RL-only: patches the env cfg built inside verl; the shared Arena task / eval /
    mimic configs are untouched. Gate off with ``rl_success_reward=False``.

    Args:
        env_cfg: the Arena env cfg to patch in place.
        subtask_reward: if True, use the graded subtask reward (0/0.5/1.0 = fraction
            of subtasks done) for earlier long-horizon credit, vs the single
            composite +1. Gate: ``subtask_reward``.

    Returns:
        True if the success reward was installed, False if no success termination
        term was found (nothing patched).
    """
    from isaaclab.managers import RewardTermCfg
    from isaaclab.utils import configclass

    term_cfg = getattr(env_cfg, "terminations", None)
    succ_term = getattr(term_cfg, "success", None) if term_cfg is not None else None
    if succ_term is None:
        logger.warning("[arena_env] terminations.success not found; skipping RL success-reward patch")
        return False

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

    # Sequential-task option: graded subtask reward vs the single composite +1. Both
    # read the success term / subtask state the sim already computed (no params needed).
    reward_func = arena_subtask_graded_reward if subtask_reward else arena_success_reward

    rewards = _ArenaRLRewardsCfg()
    rewards.task_success = RewardTermCfg(func=reward_func, weight=weight, params={})
    env_cfg.rewards = rewards

    logger.info(
        "[arena_env] RL reward patch: success->RewTerm weight=%.3f (step_dt=%.4fs, subtask_reward=%s); "
        "termination terms kept -> IsaacLab owns per-step auto-reset",
        weight,
        step_dt,
        subtask_reward,
    )
    return True


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


def register_external_arena_env(env_name: str, external_class_path: str) -> None:
    """Register a non-built-in Arena env by ``module_path:ClassName`` (idempotent).

    Mirrors the Arena policy_runner's ``--external_environment_class_path`` flag so
    external tasks (e.g. the migrated LIBERO env) can be selected via
    ``arena_state_mode``/``env_name`` without living in Arena's built-in
    ``ExampleEnvironments`` dict. No-op when the env is already registered.
    """
    if not external_class_path:
        return

    from isaaclab_arena_environments.cli import (
        ExampleEnvironments,
        parse_and_return_external_environment_from_string,
    )

    if env_name in ExampleEnvironments:
        return
    ExampleEnvironments.update(parse_and_return_external_environment_from_string(external_class_path))
