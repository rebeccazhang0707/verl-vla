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

"""``GR00TRolloutRob``: rollout worker for GR00T **N1.6** + Arena SAC training.

Scheme Y (env owns packing + decoding) makes this rollout a thin wrapper around
the model's ``sac_sample_actions`` — it mirrors the pi05 channel:

  * **obs are already packed by the env.** ``IsaacLabArenaEnv._wrap_obs`` runs the
    GR00T processor (``GR00TN16Adapter.build_inputs``) and returns the eagle
    tensors as the ``images_and_states`` keys, so the pipeline delivers them to
    this rollout as un-prefixed ``prompts.batch["images" / "lang_tokens" /
    "lang_masks" / "states"]`` (the exact slots ``_obs_to_state_dict`` reads).
    No rollout-side ``build_inputs`` is needed.
  * **the env decodes the action.** This rollout emits the **normalised** action
    chunk only; ``IsaacLabArenaEnv.step`` runs ``decode_actions_flat`` →
    ``scatter_action``. No rollout-side ``decode_actions_flat`` is needed.

Why a dedicated rollout (instead of reusing ``HFRollout``):

  * ``HFRollout.generate_sequences`` ends with ``ret = output.to_data_proto()``.
    The GR00T ``sac_sample_actions`` returns a **plain dict**
    (``{"action": ..., "log_probs": ...}``) with no ``.to_data_proto()`` — calling
    ``HFRollout`` on it would raise ``AttributeError``.

Output schema (everything here lands in the ACTION slot and is later collated
with an ``action.`` prefix by the env loop, so it must NOT carry obs/action
prefixes itself; cf. ``env_loop._collate_trajectories``):
  * ``action``       : (B, num_action_chunks, max_action_dim) — **normalised**
                       model action chunk; the env executes ``num_action_chunks``
                       steps, decoding each before scattering. Collated to
                       ``action.action`` → the replay / critic action space.
  * ``critic_value`` : (B,) min-over-heads Q (optional, gated by config).
  * ``log_probs``    : (B,) sampler log-prob.

``info.*`` keys (``task_ids`` / ``positive_sample_mask`` / ``rewards`` /
``dones`` / ``valids``) are produced by the **trainer**
(``sac_ray_trainer._prepare_actor_input``); this rollout does not emit them.
``obs.*`` is produced by the **env** (``create_env_batch_dataproto``); this
rollout does not emit it.
"""

import logging
from typing import Any, Optional

import torch
from verl import DataProto
from verl.utils.device import get_device_id

from .naive_rollout_rob import NaiveRolloutRob

logger = logging.getLogger(__name__)

__all__ = ["GR00TRolloutRob", "assert_action_horizon_invariant"]


def assert_action_horizon_invariant(
    num_action_chunks: int, critic_action_horizon: int, action_horizon: int
) -> None:
    """Fail-fast guard (#2) on the env / critic / model action-length invariant.

    ``critic_action_horizon <= num_action_chunks <= action_horizon`` must hold:

      * ``num_action_chunks < critic_action_horizon`` → the critic head slices the
        first ``critic_action_horizon`` steps of the stored chunk
        (``modeling_gr00t_sac.sac_forward_critic``); a shorter chunk is SILENTLY
        truncated / zero-padded, corrupting the Q estimate.
      * ``num_action_chunks > action_horizon`` → the model cannot emit enough steps
        to fill the chunk the env executes.

    Pure / dependency-free so it is unit-testable on a gr00t-free CPU host.
    """
    assert critic_action_horizon <= num_action_chunks <= action_horizon, (
        f"num_action_chunks={num_action_chunks} must satisfy "
        f"critic_action_horizon={critic_action_horizon} <= num_action_chunks <= "
        f"action_horizon={action_horizon}: a smaller num_action_chunks silently truncates "
        f"the critic action input; a larger one exceeds the model decode horizon."
    )


class GR00TRolloutRob(NaiveRolloutRob):
    """GR00T N1.6 + Arena SAC rollout.

    Constructed via the engine-worker call site (see
    ``VLAActorRolloutRefWorker.init_model``)::

        rollout_cls(
            config=rollout_config,
            model_config=model_config,
            device_mesh=rollout_device_mesh,
            engine=self.actor.engine if "actor" in self.role else None,
            tokenizer=self.tokenizer,
        )

    so the signature mirrors ``HFRollout`` (NOT the legacy bare-``module``
    ``PI0RolloutRob`` signature). ``self.module`` is taken from the shared actor
    ``engine`` (``engine.module``) when no explicit ``module`` is passed.
    """

    def __init__(
        self,
        config: Any = None,
        model_config: Any = None,
        device_mesh: Any = None,
        engine: Any = None,
        module: Optional[torch.nn.Module] = None,
        tokenizer: Any = None,
        **kwargs,
    ):
        # NOTE: intentionally do NOT call ``NaiveRolloutRob.__init__`` — it loads an
        # OpenVLA checkpoint from disk. The GR00T module comes from the shared actor
        # engine; we set the attributes the inherited methods (update_weights /
        # release / resume) rely on directly.
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.engine = engine
        self.module = module if module is not None else (engine.module if engine is not None else None)
        self.tokenizer = tokenizer if tokenizer is not None else self._cfg_get(model_config, "tokenizer", None)
        self.output_critic_value = bool(getattr(config, "output_critic_value", True)) if config is not None else True

        # Register the FSDP forward methods only when a module is actually present
        # (it may be ``None`` on a ref-only worker / during certain init paths).
        if self.module is not None:
            from torch.distributed.fsdp import register_fsdp_forward_method

            register_fsdp_forward_method(self.module, "sac_sample_actions")
            if self.output_critic_value:
                register_fsdp_forward_method(self.module, "sac_get_critic_value")

        # Embodiment metadata for the env-facing chunk length (scheme Y: obs packing
        # + action decoding live in the env, so no ``GR00TN16Adapter`` is built here).
        # Imported lazily so the module stays importable on a gr00t-free host.
        from verl_vla.models.gr00t.utils import GR00TDim, get_embodiment_spec

        embodiment_tag = self._cfg_get(model_config, "embodiment_tag", "gr1")
        embodiment_spec = get_embodiment_spec(embodiment_tag)
        self.action_dim = int(self._cfg_get(model_config, "action_dim", embodiment_spec.action_dim))
        # Env executes this many steps per rollout interaction; must be <= the
        # model action horizon.
        self.num_action_chunks = int(self._cfg_get(model_config, "num_action_chunks", GR00TDim.ACTION_HORIZON))

        # Fail-fast guard (#2): the env executes ``num_action_chunks`` steps and
        # stores exactly that chunk as the replay/critic action; the critic head
        # internally slices the first ``critic_action_horizon`` steps
        # (modeling_gr00t_sac.sac_forward_critic). If num_action_chunks <
        # critic_action_horizon the critic action input is SILENTLY truncated /
        # zero-padded (corrupts the Q estimate); if num_action_chunks >
        # action_horizon the model cannot emit enough steps. All three values are
        # available here at worker init: num_action_chunks from model_config,
        # critic_action_horizon / action_horizon from model.override_config (set by
        # the run script; action_horizon falls back to the checkpoint default).
        override = self._cfg_get(model_config, "override_config", None) or {}
        action_horizon = int(self._cfg_get(override, "action_horizon", GR00TDim.ACTION_HORIZON))
        critic_action_horizon = int(self._cfg_get(override, "critic_action_horizon", action_horizon))
        assert_action_horizon_invariant(self.num_action_chunks, critic_action_horizon, action_horizon)

    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
        """Read ``key`` from a dataclass-like / dict-like config, else ``default``."""
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        device_id = get_device_id()
        prompts = prompts.to(device_id)

        # obs is already packed by the env (scheme Y): ``prompts.batch`` carries the
        # un-prefixed eagle tensors (images / lang_tokens / lang_masks / states) that
        # ``Gr00tN1d6ForSAC._obs_to_state_dict`` reads. Pass ``prompts`` straight to
        # the model — no rollout-side ``build_inputs``.
        out = self.module.sac_sample_actions(prompts)
        full_action_norm = out["action"].detach().float()         # (B, horizon, max_action_dim)
        log_probs = out["log_probs"].detach().float().reshape(-1)  # (B,)

        horizon = full_action_norm.shape[1]
        assert self.num_action_chunks <= horizon, (
            f"num_action_chunks={self.num_action_chunks} exceeds model action horizon={horizon}"
        )
        # Env-facing NORMALISED action chunk; the env decodes each step
        # (decode_actions_flat → scatter_action). Collated to ``action.action`` →
        # the replay / critic action space (normalised). Kept to the chunk the env
        # executes per interaction (max_interactions = max_episode_steps // chunk).
        action_chunk = full_action_norm[:, : self.num_action_chunks, :]

        tensor_batch = {
            "action": action_chunk,
            "log_probs": log_probs,
        }

        # Critic value (optional). ``sac_get_critic_value`` re-runs the backbone +
        # min-over-heads critic on the same (obs, normalised action). Computed on the
        # SAME chunk that is stored as ``action.action`` so replay / critic stay
        # consistent (the critic head internally uses the first
        # ``critic_action_horizon`` steps).
        if self.output_critic_value:
            critic_value = (
                self.module.sac_get_critic_value(prompts, {"action": action_chunk}).detach().float().reshape(-1)
            )
            tensor_batch["critic_value"] = critic_value

        return DataProto.from_dict(tensors=tensor_batch)
