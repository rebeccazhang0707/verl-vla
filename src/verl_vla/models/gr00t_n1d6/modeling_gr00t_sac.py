# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""``Gr00tN1d6ForSAC``: GR00T N1.6 (``Gr00tN1d6``) **subclass** with SAC heads.

Adapted from the legacy ``verl.experimental.vla.models.gr00t.modeling_gr00t_sac``
to the verl-vla ``SupportSACTraining`` contract (``models/base.py``). The core flow
matching denoiser / critic logic is unchanged; the outer SAC methods now take a
``DataProto`` obs (+ tokenizer) and route through the ``Gr00tInput`` / ``Gr00tOutput``
policy adapters, mirroring ``pi0_torch``.

Design (mirrors the pi0_torch SAC integration so it plugs into the shared verl
loader/FSDP/checkpoint plumbing unchanged):

  - This is a *subclass* of gr00t's ``Gr00tN1d6`` (not an external wrapper), so
    ``AutoModel.from_pretrained`` returns it directly once registered
    (see ``register_gr00t_model`` / ``register_gr00t_sac``).
  - Critic heads are built in ``__init__`` (gated by ``config.sac_enable``) so FSDP
    wraps them. ``sac_init()`` only registers the FSDP forward methods (with a lazy
    head-build fallback for the no-FSDP smoke test).
  - ``sac_forward_actor`` runs a **grad-enabled** flow-matching denoiser so ``-Q``
    can backprop into the actor (cf. pi0_torch ``_sample_actions_flow_sde``).
  - ``_denoise`` supports the **Flow-SDE** sampler (arXiv:2510.25889); with
    ``noise_scale == 0`` it reduces bit-identically to the deterministic Euler ODE.

Importing this module requires gr00t (training Docker image only); callers that may
run without gr00t (e.g. ``build_vla_model``) must guard the import.
"""

import contextlib
import copy
import logging
import math
import os
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing this module requires gr00t (training Docker image only). Callers that
# may run without gr00t (e.g. build_vla_model) must guard the import.
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from transformers.feature_extraction_utils import BatchFeature
from verl import DataProto

from ..base import ModelOutput, SupportSACTraining, SupportSFTTraining
from .compat import apply_gr00t_compat_patches
from .configuration_gr00t import cfg_get as _cfg
from .gr00t_policy import GR00TN16Adapter
from .policy import get_gr00t_policy_classes
from .utils import GR1, GR00TDim

logger = logging.getLogger(__name__)


class CriticMLP(nn.Module):
    """Two-layer MLP: input_dim -> 512 -> 256 -> 1.

    When ``use_layernorm`` is set, a ``LayerNorm`` is inserted after each hidden
    ``Linear`` and before the activation (DroQ/REDQ-style "LayerNorm critic"), a
    standard SAC stabiliser. Disabled by default so existing checkpoints stay
    bit-identical.
    """

    def __init__(self, input_dim: int, use_layernorm: bool = False):
        super().__init__()
        if use_layernorm:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def beta_schedule(step: int, beta0: float, beta_min: float, T: int) -> float:
    """Cosine anneal of the Flow-SDE exploration scale (mirrors pi0_torch)."""
    progress = min(step / T, 1.0)
    return beta_min + (beta0 - beta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


class Gr00tN1d6ForSAC(Gr00tN1d6, SupportSACTraining, SupportSFTTraining):
    """``Gr00tN1d6`` augmented with SAC critic heads.

    Loaded by ``AutoModel.from_pretrained`` once ``register_gr00t_sac()`` has
    overridden gr00t's own registration for the ``Gr00tN1d6`` config class.

    Also implements ``SupportSFTTraining`` so the shared TD3+BC / SFT plumbing can
    call the unified ``sft_loss(obs, tokenizer, actions, valids, ...)`` (the training
    worker's TD3-BC term uses it); see ``sft_loss`` / ``_bc_mse`` (true flow-matching
    velocity BC, GR00T t: noise→action).
    """

    def __init__(self, config):
        super().__init__(config)
        # Wire the SupportSFTTraining surface (sets self.config + self.sft_metrics)
        # without a second Gr00tN1d6.__init__ (mirrors pi0's cooperative init).
        SupportSFTTraining.__init__(self, config)

        # --- SAC routing / policy adapter config ---
        self.policy_type = str(_cfg(config, "policy_type", "arena"))
        self.embodiment_tag = str(_cfg(config, "embodiment_tag", GR1.name))
        # Lazily built (needs the gr00t processor + checkpoint); see _get_adapter.
        self._adapter: Optional[GR00TN16Adapter] = None
        self._adapter_model_path = getattr(config, "_name_or_path", None) or _cfg(config, "adapter_model_path", None)

        # --- dims from model config (authoritative); override_config can set the
        #     SAC-only ones that the gr00t config does not carry. ---
        self.action_horizon = int(_cfg(config, "action_horizon", GR00TDim.ACTION_HORIZON))
        self.max_action_dim = int(_cfg(config, "max_action_dim", GR00TDim.MAX_ACTION_DIM))
        self.max_state_dim = int(_cfg(config, "max_state_dim", GR00TDim.MAX_STATE_DIM))
        self.backbone_feature_dim = int(_cfg(config, "backbone_embedding_dim", 2048))
        self.num_inference_timesteps = int(_cfg(config, "num_inference_timesteps", 4))
        self.num_timestep_buckets = int(_cfg(config, "num_timestep_buckets", 1000))
        self.add_pos_embed = bool(_cfg(config, "add_pos_embed", False))
        self.use_alternate_vl_dit = bool(_cfg(config, "use_alternate_vl_dit", False))
        self.state_horizon = int(_cfg(config, "sac_state_horizon", GR00TDim.STATE_HORIZON))

        # env action chunk length executed per interaction. Defaults to 16
        # (GR00TDim.ACTION_HORIZON) rather than the checkpoint action_horizon (=50),
        # so we execute a short chunk of the long decoded horizon; clamped to the
        # decoded horizon so it can never over-index the action chunk.
        self.num_action_chunks = min(
            int(_cfg(config, "num_action_chunks", GR00TDim.ACTION_HORIZON)), self.action_horizon
        )

        # real (unpadded) GR1 action width used for the SAC critic and the env;
        # NOT present in the gr00t config -> comes from override_config / default.
        self.action_dim = int(_cfg(config, "action_dim", GR1.action_dim))
        self.embodiment_id = int(_cfg(config, "embodiment_id", GR1.embodiment_id))
        self.num_critic_heads = int(_cfg(config, "critic_head_num", 10))

        # how much of the action chunk the critic sees. Default to the *executed*
        # chunk length (``num_action_chunks``), not the full decoded horizon (often
        # 50): the env only steps the first N actions, so scoring the unused tail
        # wastes capacity and mismatches TD targets. Override via yaml when needed.
        self.critic_action_dim = int(_cfg(config, "critic_action_dim", self.action_dim))
        self.critic_action_horizon = int(
            _cfg(config, "critic_action_horizon", self.num_action_chunks)
        )

        # --- Critic representation options ---
        # critic_type (yaml) selects the pooling; keep critic_pooling as an escape hatch.
        critic_type = str(_cfg(config, "critic_type", "cross_attn")).lower()
        critic_pooling_default = {"cross_attn": "attn", "mean_pool": "mean"}.get(critic_type, "attn")
        self.critic_pooling = str(_cfg(config, "critic_pooling", critic_pooling_default)).lower()
        self.critic_use_encoded_state = bool(_cfg(config, "critic_use_encoded_state", False))
        self.critic_prefix_attn_heads = int(_cfg(config, "critic_prefix_attn_heads", 8))
        # state_encoder output width (used only when reusing the encoded state)
        self._state_feature_dim = int(getattr(self.action_head, "input_embedding_dim", self.max_state_dim))

        # Asymmetric actor-critic privileged obs (Phase 2; default off -> bit-identical critic).
        self.critic_privileged_obs = bool(_cfg(config, "critic_privileged_obs", False))
        self.critic_privileged_obs_dim = (
            int(_cfg(config, "critic_privileged_obs_dim", 0)) if self.critic_privileged_obs else 0
        )
        if self.critic_privileged_obs and self.critic_privileged_obs_dim > 0:
            self.register_buffer(
                "priv_obs_running_mean", torch.zeros(self.critic_privileged_obs_dim, dtype=torch.float32)
            )
            self.register_buffer(
                "priv_obs_running_var", torch.ones(self.critic_privileged_obs_dim, dtype=torch.float32)
            )
            self.register_buffer("priv_obs_running_count", torch.zeros((), dtype=torch.float64))

        # --- Critic dimensionality-reduction options (default OFF = legacy behaviour) ---
        self.critic_pool_proj_dim = int(_cfg(config, "critic_pool_proj_dim", 0))
        _real_state = _cfg(config, "critic_state_real_dim", None)
        self.critic_state_real_dim = int(_real_state) if _real_state is not None else None
        self.critic_mask_frozen_action = bool(_cfg(config, "critic_mask_frozen_action", False))
        self.critic_layernorm = bool(_cfg(config, "critic_layernorm", False))

        # critic input = pooled backbone + flat state + flat (sliced) action [+ privileged obs]
        base_state_width = self._state_feature_dim if self.critic_use_encoded_state else self.max_state_dim
        if (not self.critic_use_encoded_state) and self.critic_state_real_dim is not None:
            base_state_width = self.critic_state_real_dim
        self._critic_state_width = base_state_width
        self._critic_pooled_dim = (
            self.critic_pool_proj_dim if self.critic_pool_proj_dim > 0 else self.backbone_feature_dim
        )
        self.critic_input_dim = (
            self._critic_pooled_dim
            + self.state_horizon * self._critic_state_width
            + self.critic_action_horizon * self.critic_action_dim
            + self.critic_privileged_obs_dim
        )

        # --- Flow-SDE (stochastic flow sampler) config; see _denoise. ---
        self.flow_sde_enable = bool(_cfg(config, "flow_sde_enable", False))
        self.flow_sde_noise_level = float(_cfg(config, "flow_sde_noise_level", 0.065))
        self.flow_sde_rollout_noise_scale = float(_cfg(config, "flow_sde_rollout_noise_scale", 1.0))
        self.flow_sde_train_noise_scale = float(_cfg(config, "flow_sde_train_noise_scale", 1.0))
        per_dim = _cfg(config, "flow_sde_noise_level_per_dim", None)
        if per_dim is not None:
            vec = torch.full((self.max_action_dim,), float(self.flow_sde_noise_level), dtype=torch.float32)
            for i, v in enumerate(list(per_dim)[: self.max_action_dim]):
                vec[i] = float(v)
            self.register_buffer("flow_sde_noise_level_vec", vec.view(1, 1, -1), persistent=False)
        else:
            self.flow_sde_noise_level_vec = None
        self.flow_sde_initial_beta = float(_cfg(config, "flow_sde_initial_beta", 1.0))
        self.flow_sde_beta_min = float(_cfg(config, "flow_sde_beta_min", 0.02))
        self.flow_sde_beta_schedule_T = int(_cfg(config, "flow_sde_beta_schedule_T", 4000))
        self.register_buffer("flow_sde_step", torch.zeros((), dtype=torch.long))

        # --- Plan A: state-dependent trainable log-std HEAD (default off) ---
        self.flow_sde_std_head = bool(_cfg(config, "flow_sde_std_head", False))
        if self.flow_sde_std_head:
            std_in_dim = int(getattr(self.action_head, "hidden_size", self.backbone_feature_dim))
            std_hidden = int(_cfg(config, "flow_sde_std_head_hidden", 256))
            self.flow_sde_std_net = nn.Sequential(
                nn.Linear(std_in_dim, std_hidden),
                nn.ReLU(),
                nn.Linear(std_hidden, self.max_action_dim),
            )
            nn.init.zeros_(self.flow_sde_std_net[-1].weight)
            nn.init.constant_(self.flow_sde_std_net[-1].bias, math.log(max(self.flow_sde_noise_level, 1e-6)))
            self._flow_sde_log_std_min = math.log(1e-3)
            self._flow_sde_log_std_max = math.log(0.5)
            self._flow_sde_std_in_dim = std_in_dim

        # --- Trainable action dims (online RL action mask). None/unset -> train ALL dims. ---
        train_dims = _cfg(config, "sac_action_train_dims", None)
        mask = torch.zeros(self.max_action_dim, dtype=torch.bool)
        if train_dims is None:
            mask[:] = True
        else:
            for rng in train_dims:
                s, e = int(rng[0]), int(rng[1])
                mask[s:e] = True
        self.register_buffer("sac_action_train_mask", mask, persistent=False)
        self.sac_action_train_all = bool(mask.all().item())

        self.critic_heads: Optional[nn.ModuleList] = None
        self.target_critic_heads: Optional[nn.ModuleList] = None
        self.critic_pool_proj: Optional[nn.Module] = None
        self.target_pool_proj: Optional[nn.Module] = None

        # Build heads here (pre-FSDP) when SAC is enabled, so FSDP wraps them.
        if bool(getattr(config, "sac_enable", False)):
            self._build_sac_heads()

        # Freeze the Eagle vision tower for SAC (mirrors pi0.5). Toggle off with
        # freeze_vision_tower=False.
        self.freeze_vision_tower_enabled = bool(_cfg(config, "freeze_vision_tower", True))
        if self.freeze_vision_tower_enabled:
            self.freeze_vision_tower()

        # Freeze the action-head I/O adapters for same-task RL post-training. Default off.
        self.freeze_action_io_enabled = bool(_cfg(config, "freeze_action_io", False))
        if self.freeze_action_io_enabled:
            self.freeze_action_io()

    # ------------------------------------------------------------------
    # Policy adapters (Gr00tInput / Gr00tOutput) + processor adapter
    # ------------------------------------------------------------------

    def _get_policy_classes(self):
        return get_gr00t_policy_classes(self.policy_type)

    def _get_adapter(self) -> GR00TN16Adapter:
        """Lazily build the processor-delegating GR00TN16Adapter.

        Uses the checkpoint path recorded on the config (``_name_or_path`` set by
        ``from_pretrained``) so ``build_inputs`` / ``decode_action`` match the loaded
        weights. Kept lazy so the module imports (and the model constructs) without
        the processor / checkpoint present.
        """
        if self._adapter is None:
            if not self._adapter_model_path:
                raise RuntimeError(
                    "Gr00tN1d6ForSAC: cannot build GR00TN16Adapter without a checkpoint path; "
                    "config._name_or_path is unset (set adapter_model_path in override_config)."
                )
            self._adapter = GR00TN16Adapter(self._adapter_model_path, embodiment_tag=self.embodiment_tag)
        return self._adapter

    def _prepare_inputs(self, obs: DataProto, tokenizer=None) -> tuple[dict[str, torch.Tensor], "Any"]:
        """Build the eagle ``s`` dict + raw state groups from raw obs.

        SINGLE source of truth for GR00T model inputs: both ``sac_sample_actions``
        (rollout) and ``sac_forward_state_features`` (train) go through here so the
        processor-built inputs are identical, avoiding rollout/train drift.
        """
        del tokenizer  # the gr00t processor owns tokenization
        adapter = self._get_adapter()
        input_cls, _ = self._get_policy_classes()
        model_input = input_cls.from_env_obs(obs)

        inputs, raw_state_groups = adapter.build_inputs(
            model_input.images,
            model_input.state,
            model_input.task,
        )

        pixel_values = inputs["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values, dim=0)
        n_views = len(adapter.video_keys)
        batch_size = next(iter(model_input.images.values())).shape[0]
        pixel_values = pixel_values.reshape(batch_size, n_views, *pixel_values.shape[1:])
        s = {
            "images": pixel_values,
            "lang_tokens": inputs["input_ids"],
            "lang_masks": inputs["attention_mask"],
            "states": inputs["state"],
        }
        return s, raw_state_groups

    # ------------------------------------------------------------------
    # SAC head construction
    # ------------------------------------------------------------------

    def _build_sac_heads(self):
        self.critic_heads = nn.ModuleList(
            [
                CriticMLP(self.critic_input_dim, use_layernorm=self.critic_layernorm)
                for _ in range(self.num_critic_heads)
            ]
        )
        self.target_critic_heads = copy.deepcopy(self.critic_heads)
        for p in self.target_critic_heads.parameters():
            p.requires_grad_(False)

        if self.critic_pool_proj_dim > 0:
            self.critic_pool_proj = nn.Linear(self.backbone_feature_dim, self.critic_pool_proj_dim)
            self.target_pool_proj = copy.deepcopy(self.critic_pool_proj)
            for p in self.target_pool_proj.parameters():
                p.requires_grad_(False)
        else:
            self.critic_pool_proj = None
            self.target_pool_proj = None

        if self.critic_pooling == "attn":
            d = self.backbone_feature_dim
            # Learnable cross-attn query token held as nn.Embedding (NOT a bare Parameter)
            # so from_pretrained's _fast_init reaches + initialises it (avoids NaN query).
            self.critic_state_token = nn.Embedding(1, d)
            self.target_state_token = nn.Embedding(1, d)
            nn.init.normal_(self.critic_state_token.weight, mean=0.0, std=0.02)
            self.target_state_token.load_state_dict(self.critic_state_token.state_dict())
            for p in self.target_state_token.parameters():
                p.requires_grad_(False)
            self.critic_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=self.critic_prefix_attn_heads, batch_first=True
            )
            self.target_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=self.critic_prefix_attn_heads, batch_first=True
            )
            self.target_prefix_cross_attn.load_state_dict(self.critic_prefix_cross_attn.state_dict())
            for p in self.target_prefix_cross_attn.parameters():
                p.requires_grad_(False)
        else:
            self.critic_state_token = None
            self.target_state_token = None
            self.critic_prefix_cross_attn = None
            self.target_prefix_cross_attn = None

        logger.info(
            "Gr00tN1d6ForSAC: %d critic heads, input_dim=%d (pooled=%d[proj=%s], "
            "state=%dx%d[real=%s], action=%dx%d[mask_frozen=%s], priv=%d, layernorm=%s, pooling=%s)",
            self.num_critic_heads,
            self.critic_input_dim,
            self._critic_pooled_dim,
            self.critic_pool_proj_dim if self.critic_pool_proj_dim > 0 else "off",
            self.state_horizon,
            self._critic_state_width,
            self.critic_state_real_dim if self.critic_state_real_dim is not None else "off",
            self.critic_action_horizon,
            self.critic_action_dim,
            bool(self.critic_mask_frozen_action and not self.sac_action_train_all),
            self.critic_privileged_obs_dim,
            self.critic_layernorm,
            self.critic_pooling,
        )

    def freeze_vision_tower(self) -> None:
        """Freeze the Eagle vision tower (+ vision->LLM connector) for SAC, a la pi0.5."""
        eagle = getattr(self.backbone, "eagle_model", None)
        vision_model = getattr(eagle, "vision_model", None) if eagle is not None else None
        if vision_model is None:
            logger.warning("[gr00t-sac] backbone.eagle_model.vision_model not found; skipping freeze_vision_tower")
            return
        vision_model.requires_grad_(False)
        vision_model.eval()
        mlp1 = getattr(eagle, "mlp1", None)
        if mlp1 is not None:
            mlp1.requires_grad_(False)
            mlp1.eval()
        logger.info(
            "[gr00t-sac] vision tower frozen (eagle_model.vision_model%s)",
            " + mlp1 connector" if mlp1 is not None else "",
        )

    def freeze_action_io(self) -> None:
        """Freeze the action-head I/O adapters for same-task RL post-training."""
        ah = getattr(self, "action_head", None)
        if ah is None:
            logger.warning("[gr00t-sac] action_head not found; skipping freeze_action_io")
            return
        frozen = []
        for name in ("state_encoder", "action_encoder", "action_decoder"):
            mod = getattr(ah, name, None)
            if mod is None:
                logger.warning("[gr00t-sac] action_head.%s not found; skipping", name)
                continue
            mod.requires_grad_(False)
            mod.eval()
            frozen.append(name)
        logger.info("[gr00t-sac] action I/O frozen (action_head.%s)", ", action_head.".join(frozen))

    def _init_weights(self, module):
        """Initialize newly-added SAC critic-head params (missing from the checkpoint)."""
        sup = getattr(super(), "_init_weights", None)
        if callable(sup):
            sup(module)
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=5**0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            if getattr(module, "in_proj_weight", None) is not None:
                torch.nn.init.xavier_uniform_(module.in_proj_weight)
            if getattr(module, "in_proj_bias", None) is not None:
                torch.nn.init.zeros_(module.in_proj_bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # SupportSACTraining interface
    # ------------------------------------------------------------------

    def sac_init(self):
        # Production path: heads already built in __init__ (sac_enable=True) and
        # FSDP-wrapped. Lazy fallback covers the no-FSDP smoke test.
        if self.critic_heads is None:
            self._build_sac_heads()

        if self.freeze_vision_tower_enabled:
            self.freeze_vision_tower()

        from torch.distributed.fsdp import register_fsdp_forward_method

        # NOTE: sac_get_critic_value is registered on the rollout side (see
        # HFRollout.__init__), so it is intentionally NOT registered here.
        for method in (
            "sac_sample_actions",
            "sac_forward_state_features",
            "sac_forward_actor",
            "sac_forward_critic",
            "sac_update_target_network",
            "sft_loss",
        ):
            register_fsdp_forward_method(self, method)

    def sft_init(self):
        """SupportSFTTraining entry (pure-SFT / recap path).

        Mirrors ``sac_init`` for the FSDP forward-method registration but only needs
        ``sft_loss`` (+ the vision-tower freeze), matching pi0's ``sft_init``.
        """
        self.sft_metrics = {}
        if self.freeze_vision_tower_enabled:
            self.freeze_vision_tower()
        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(self, "sft_loss")

    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        assert self.critic_heads is not None, "Call sac_init() first"
        params = list(self.critic_heads.parameters())
        if self.critic_pool_proj is not None:
            params += list(self.critic_pool_proj.parameters())
        if self.critic_pooling == "attn":
            params += list(self.critic_prefix_cross_attn.parameters())
            params += list(self.critic_state_token.parameters())
        return params

    def sac_get_named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        actor_params = []
        for name, p in self.action_head.named_parameters():
            if p.requires_grad:
                actor_params.append((f"action_head.{name}", p))
        for name, p in self.backbone.named_parameters():
            if p.requires_grad:
                actor_params.append((f"backbone.{name}", p))
        return actor_params

    # ------------------------------------------------------------------
    # Rollout entry: obs -> Gr00tOutput
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer: Optional[torch.nn.Module] = None,
        eval: bool = False,
    ) -> ModelOutput:
        """Rollout-time action sampling: build inputs -> Flow-SDE sample -> decode.

        Reuses ``_prepare_inputs`` (same as ``sac_forward_state_features``) so rollout
        and train see identical processor inputs. Returns a ``Gr00tOutput`` whose
        ``action`` is the env-facing decoded chunk and whose ``full_action`` is the
        normalised model action stored for the replay/critic.
        """
        _, output_cls = self._get_policy_classes()
        s, raw_state_groups = self._prepare_inputs(obs, tokenizer)
        # Call the impl directly (NOT the registered sac_forward_state_features) so root
        # params stay all-gathered through _denoise -- sac_sample_actions is itself a
        # registered FSDP forward entry (see _state_features_impl docstring).
        state_features = self._state_features_impl(s)

        override = obs.meta_info.get("rollout_noise_scale", None) if obs.meta_info else None
        if eval:
            noise_scale = 0.0
        elif override is not None:
            noise_scale = float(override)
        else:
            noise_scale = self.flow_sde_rollout_noise_scale if self.flow_sde_enable else 0.0
        return_log_prob = self.flow_sde_enable and noise_scale > 0.0

        full_action, log_probs = self._denoise(
            state_features, noise_scale=noise_scale, requires_grad=False, return_log_prob=return_log_prob
        )
        full_action_norm = full_action.detach().float()

        # Decode normalised actions -> absolute policy-order joints (B, horizon, action_dim).
        decoded_flat = self._get_adapter().decode_actions_flat(full_action_norm.cpu().numpy(), raw_state_groups)
        decoded = torch.as_tensor(decoded_flat, dtype=torch.float32, device=full_action_norm.device)

        return output_cls.from_model_output(
            {
                "full_action": full_action_norm,
                "decoded_action": decoded,
                "log_probs": log_probs,
                "num_action_chunks": self.num_action_chunks,
            }
        )

    @torch.no_grad()
    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions: ModelOutput,
        tokenizer: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        state_features = self.sac_forward_state_features(obs, tokenizer)
        # The critic scores the NORMALISED action; rollouts always populate it.
        q_values = self.sac_forward_critic(
            {"full_action": actions.full_action},
            state_features,
            use_target_network=False,
            method="min",
            requires_grad=False,
        )
        return q_values.detach().float().reshape(-1)

    # ------------------------------------------------------------------
    # State features (backbone) -- reused by rollout + train
    # ------------------------------------------------------------------

    def sac_forward_state_features(
        self, obs: DataProto, tokenizer: Optional[torch.nn.Module] = None
    ) -> dict[str, torch.Tensor]:
        """Registered FSDP forward entry: raw obs -> backbone/state features.

        Builds the eagle inputs via ``_prepare_inputs`` (the SAME path as
        ``sac_sample_actions``) then runs the backbone + state encoder. Returns a
        flat dict of tensors so ``split_nested_dicts_or_tuples`` can split the
        concatenated (s0, s1) batch.
        """
        s, _ = self._prepare_inputs(obs, tokenizer)
        return self._state_features_impl(s)

    def _state_features_impl(self, s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the Eagle backbone on the stored inputs and encode the state."""
        ah = self.action_head
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        imgs = s["images"]
        if isinstance(imgs, torch.Tensor):
            imgs = imgs.flatten(0, 1)
            pixel_values = [imgs[i].to(device, dtype=dtype) for i in range(imgs.shape[0])]
        else:
            pixel_values = [t.to(device, dtype=dtype) for t in imgs]

        backbone_inputs = BatchFeature(
            data={
                "input_ids": s["lang_tokens"].to(device).long(),
                "attention_mask": s["lang_masks"].to(device),
                "pixel_values": pixel_values,
            }
        )
        backbone_outputs = self.backbone(backbone_inputs)

        raw_features = backbone_outputs["backbone_features"]  # (B, S, D)
        attn_mask = backbone_outputs["backbone_attention_mask"]  # (B, S) bool
        image_mask = backbone_outputs.get("image_mask", None)  # (B, S) bool | None

        vl_embeds = ah.vlln(raw_features)  # match action-head vlln

        # Masked mean-pool for the critic. Use torch.where (NOT vl_embeds * mask)
        # because padding positions can carry NaN/inf (0 * NaN = NaN would poison the pool).
        mask_b = attn_mask.unsqueeze(-1).bool()
        vl_safe = torch.where(mask_b, vl_embeds, torch.zeros_like(vl_embeds))
        denom = mask_b.sum(dim=1).clamp(min=1).to(vl_embeds.dtype)
        pooled = vl_safe.sum(dim=1) / denom

        state = s["states"].to(device, dtype=dtype)  # (B, T, max_state_dim)
        B = state.shape[0]
        embodiment_id = torch.full((B,), self.embodiment_id, dtype=torch.long, device=device)
        state_features = ah.state_encoder(state, embodiment_id)  # (B, T, input_emb_dim)

        out = {
            "pooled": pooled,
            "backbone_features": vl_embeds,
            "backbone_attention_mask": attn_mask,
            "state_features": state_features,
            "state": state,
            "embodiment_id": embodiment_id,
        }
        # Keep image_mask only when present so the returned dict is a flat tensor tree
        # (split_nested_dicts_or_tuples rejects None); consumers use .get("image_mask").
        if image_mask is not None:
            out["image_mask"] = image_mask
        # Asymmetric-AC privileged critic obs (Phase 2): carried through when present.
        if self.critic_privileged_obs and "priv_obs" in s:
            out["priv_obs"] = s["priv_obs"].to(device, dtype=dtype)
        return out

    # ------------------------------------------------------------------
    # Grad-enabled flow-matching denoiser (the actor sampler)
    # ------------------------------------------------------------------

    def _gaussian_log_prob(self, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Diagonal-Gaussian log-prob, mean-reduced over (horizon, action_dim)."""
        std_safe = std.clamp_min(1e-6)
        log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
        if self.sac_action_train_all:
            return log_prob.mean(dim=(-1, -2))
        m = self.sac_action_train_mask.view(1, 1, -1)
        n_train = int(self.sac_action_train_mask.sum().item())
        log_prob = log_prob.masked_fill(~m, 0.0).sum(dim=(-1, -2))
        return log_prob / max(self.action_horizon * n_train, 1)

    def flow_sde_beta(self) -> torch.Tensor:
        beta = beta_schedule(
            int(self.flow_sde_step.item()),
            beta0=self.flow_sde_initial_beta,
            beta_min=self.flow_sde_beta_min,
            T=self.flow_sde_beta_schedule_T,
        )
        return torch.tensor(beta, device=self.flow_sde_step.device, dtype=torch.float32)

    def _flow_sde_noise_level(self, dtype=None, device=None, features=None):
        """Effective per-step SDE noise magnitude sigma_0 used by ``_denoise``."""
        if getattr(self, "flow_sde_std_head", False) and features is not None:
            feat = features[:, -self.action_horizon :]
            assert feat.shape[-1] == self._flow_sde_std_in_dim, (
                f"flow_sde_std_head in_dim {self._flow_sde_std_in_dim} != DiT feature dim "
                f"{feat.shape[-1]}; std_in_dim should be action_head.hidden_size"
            )
            sigma0 = self.flow_sde_std_net(feat).clamp(self._flow_sde_log_std_min, self._flow_sde_log_std_max).exp()
            if dtype is not None or device is not None:
                sigma0 = sigma0.to(dtype=dtype, device=device)
            return sigma0
        vec = getattr(self, "flow_sde_noise_level_vec", None)
        if vec is not None:
            if dtype is not None or device is not None:
                return vec.to(dtype=dtype, device=device)
            return vec
        return self.flow_sde_noise_level

    def _denoise(
        self,
        sf: dict[str, torch.Tensor],
        *,
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample a (B, action_horizon, max_action_dim) action via the flow sampler.

        When only a subset of dims is trainable (``sac_action_train_dims``) it runs
        two decoupled passes from the SAME initial noise: an explored pass whose
        trainable dims carry the gradient + log-prob, and a deterministic base pass
        whose frozen dims are read verbatim.
        """
        vl_embeds = sf["backbone_features"]
        B = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype
        x0 = torch.randn((B, self.action_horizon, self.max_action_dim), dtype=dtype, device=device)

        if self.sac_action_train_all:
            return self._run_flow(
                sf, x0, noise_scale=noise_scale, requires_grad=requires_grad, return_log_prob=return_log_prob
            )

        x_train, log_probs = self._run_flow(
            sf, x0, noise_scale=noise_scale, requires_grad=requires_grad, return_log_prob=return_log_prob
        )
        with torch.no_grad():
            x_base, _ = self._run_flow(sf, x0, noise_scale=0.0, requires_grad=False, return_log_prob=False)
        mask = self.sac_action_train_mask.view(1, 1, -1)
        x = torch.where(mask, x_train, x_base)
        return x, log_probs

    def _run_flow(
        self,
        sf: dict[str, torch.Tensor],
        x: torch.Tensor,
        *,
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One flow-matching trajectory (NO per-dim masking) from initial noise ``x``.

        Replicates ``Gr00tN1d6ActionHead.get_action_with_features``, grad-enabled and
        optionally stochastic (Flow-SDE, arXiv:2510.25889). GR00T integrates
        noise (t=0) -> action (t=1) with ``x = x + dt*v``; the SDE posterior is applied
        under ``s = 1 - t``. With ``noise_scale == 0`` it reduces bit-identically to the
        deterministic Euler step.
        """
        ah = self.action_head
        vl_embeds = sf["backbone_features"]
        attn_mask = sf["backbone_attention_mask"]
        image_mask = sf.get("image_mask")
        state_features = sf["state_features"]
        emb_id = sf["embodiment_id"]

        B = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype = vl_embeds.dtype

        num_steps = self.num_inference_timesteps
        use_sde = self.flow_sde_enable and noise_scale > 0.0
        beta = self.flow_sde_beta().to(device=device, dtype=dtype) if use_sde else None
        step_log_probs: list[torch.Tensor] = []
        ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()

        with ctx:
            for i in range(num_steps):
                t_cur = i / float(num_steps)
                t_next = (i + 1) / float(num_steps)
                delta = t_next - t_cur

                t_disc = int(t_cur * self.num_timestep_buckets)
                timesteps = torch.full((B,), t_disc, device=device)

                action_features = ah.action_encoder(x, timesteps, emb_id)
                if self.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    action_features = action_features + ah.position_embedding(pos_ids).unsqueeze(0)

                sa_embs = torch.cat((state_features, action_features), dim=1)

                if self.use_alternate_vl_dit:
                    model_output = ah.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps,
                        image_mask=image_mask,
                        backbone_attention_mask=attn_mask,
                    )
                else:
                    model_output = ah.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps,
                    )
                pred = ah.action_decoder(model_output, emb_id)
                v = pred[:, -self.action_horizon :]

                if use_sde:
                    sigma0 = self._flow_sde_noise_level(dtype=dtype, device=device, features=model_output)
                    if self.flow_sde_std_head:
                        x_mean = x + delta * v
                        sigma_t = sigma0 * noise_scale  # (B, action_horizon, D)
                    else:
                        data_pred = x + v * (1.0 - t_cur)
                        noise_pred = x - v * t_cur
                        s_cur = min(max(1.0 - t_cur, 1e-4), 1.0 - 1e-4)
                        s_next = 1.0 - t_next
                        sigma = beta * (sigma0 * noise_scale * math.sqrt(s_cur / (1.0 - s_cur)))
                        data_w = 1.0 - s_next  # == t_next
                        noise_w = s_next - (sigma**2 * delta) / (2.0 * s_cur)
                        x_mean = data_pred * data_w + noise_pred * noise_w
                        sigma_t = math.sqrt(delta) * sigma
                        sigma_t = torch.as_tensor(sigma_t, device=device, dtype=dtype).reshape(1, 1, -1)
                    eps = torch.randn_like(x)
                    x = x_mean + sigma_t * eps
                    if return_log_prob:
                        step_log_probs.append(self._gaussian_log_prob(x, x_mean, sigma_t))
                else:
                    x = x + delta * v

        log_probs = torch.stack(step_log_probs, dim=1).sum(dim=1) if step_log_probs else None
        return x, log_probs

    def sac_forward_actor(
        self,
        state_features: dict[str, torch.Tensor],
        task_ids: Optional[torch.Tensor] = None,
        is_first_micro_batch: bool = False,
        noise_scale: Optional[float] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, float]]:
        """Grad-enabled action sampling for the actor update.

        Returns the NORMALISED model action (B, action_horizon, max_action_dim) -- the
        differentiable space the critic operates in (the training worker keys it as
        ``{"action": actions}``). ``task_ids`` is accepted for interface parity but
        unused by the cross-attn / mean-pool critic (GR00T noise is not task-routed).
        """
        del task_ids
        resolved_noise_scale = (
            (self.flow_sde_train_noise_scale if self.flow_sde_enable else 0.0)
            if noise_scale is None
            else float(noise_scale)
        )
        actions, log_probs = self._denoise(
            state_features,
            noise_scale=resolved_noise_scale,
            requires_grad=True,
            return_log_prob=self.flow_sde_enable and resolved_noise_scale > 0.0,
        )
        if is_first_micro_batch and self.flow_sde_enable:
            self.flow_sde_step.add_(1)
        metrics: dict[str, float] = {}
        if self.flow_sde_enable:
            metrics = {
                "flow_sde_beta": float(self.flow_sde_beta().item()),
                "flow_sde_step": float(self.flow_sde_step.item()),
                "flow_sde_noise_scale": float(resolved_noise_scale),
            }
        return actions, log_probs, metrics

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    def _cross_attention_pool(
        self, vl_embeds: torch.Tensor, attn_mask: torch.Tensor, use_target_network: bool
    ) -> torch.Tensor:
        """Learnable cross-attention pool: a state-token query attends over the VL tokens."""
        cross_attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
        state_token = (self.target_state_token if use_target_network else self.critic_state_token).weight  # (1, d)
        B = vl_embeds.shape[0]
        mask_b = attn_mask.unsqueeze(-1).to(torch.bool)
        vl_safe = torch.where(mask_b, vl_embeds, torch.zeros_like(vl_embeds))
        query = state_token.view(1, 1, -1).expand(B, -1, -1)
        key_padding_mask = ~attn_mask.to(dtype=torch.bool)  # True => ignore (padding)
        pooled, _ = cross_attn(
            query=query,
            key=vl_safe,
            value=vl_safe,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return pooled.squeeze(1)  # (B, D)

    def _normalize_priv_obs(self, priv: torch.Tensor, update: bool) -> torch.Tensor:
        """Whiten the privileged obs with online (Welford) running mean/std."""
        eps = 1e-5
        if update and torch.is_grad_enabled():
            with torch.no_grad():
                finite = torch.isfinite(priv).all(dim=-1)
                x = priv[finite].to(torch.float64)
                n = x.shape[0]
                if n > 0:
                    batch_mean = x.mean(dim=0)
                    batch_var = x.var(dim=0, unbiased=False)
                    count = self.priv_obs_running_count
                    new_count = count + n
                    delta = batch_mean - self.priv_obs_running_mean.to(torch.float64)
                    new_mean = self.priv_obs_running_mean.to(torch.float64) + delta * (n / new_count)
                    m_a = self.priv_obs_running_var.to(torch.float64) * count
                    m_b = batch_var * n
                    m2 = m_a + m_b + (delta**2) * count * n / new_count
                    new_var = m2 / new_count
                    self.priv_obs_running_mean.copy_(new_mean.to(self.priv_obs_running_mean.dtype))
                    self.priv_obs_running_var.copy_(new_var.to(self.priv_obs_running_var.dtype))
                    self.priv_obs_running_count.copy_(new_count)
        mean = self.priv_obs_running_mean.to(priv.dtype)
        std = (self.priv_obs_running_var.to(priv.dtype) + eps).sqrt()
        return (priv - mean) / std

    @staticmethod
    def _action_from_dict(a: dict[str, torch.Tensor]) -> torch.Tensor:
        """The critic action space is the NORMALISED ``full_action``.

        Replay transitions carry ``full_action`` (the differentiable model space);
        the actor update passes a single ``{"action": ...}`` that is already
        normalised. We always take ``full_action`` when present and otherwise the
        (already normalised) ``action`` -- the DECODED env action never reaches the
        critic because replay transitions always persist ``full_action``.
        """
        return a.get("full_action", a["action"])

    def _critic_input(
        self, a: dict[str, torch.Tensor], sf: dict[str, torch.Tensor], use_target_network: bool = False
    ) -> torch.Tensor:
        if self.critic_pooling == "attn":
            pooled = self._cross_attention_pool(
                sf["backbone_features"], sf["backbone_attention_mask"], use_target_network
            )  # (B, D)
        else:
            pooled = sf["pooled"]  # (B, D)
        if self.critic_pool_proj is not None:
            proj = self.target_pool_proj if use_target_network else self.critic_pool_proj
            pooled = proj(pooled)  # (B, critic_pool_proj_dim)
        B = pooled.shape[0]
        state_src = sf["state_features"] if self.critic_use_encoded_state else sf["state"]
        if (not self.critic_use_encoded_state) and self.critic_state_real_dim is not None:
            state_src = state_src[..., : self.critic_state_real_dim]
        state_flat = state_src.reshape(B, -1)  # (B, T*state_width)
        full_action = self._action_from_dict(a).to(pooled.dtype)  # (B, horizon, max_action_dim)
        act = full_action[:, : self.critic_action_horizon, : self.critic_action_dim]
        if self.critic_mask_frozen_action and not self.sac_action_train_all:
            m = self.sac_action_train_mask[: self.critic_action_dim].view(1, 1, -1)
            act = torch.where(m, act, torch.zeros_like(act))
        act = act.reshape(B, -1)
        parts = [pooled, state_flat, act]
        if self.critic_privileged_obs:
            priv = sf.get("priv_obs")
            if priv is not None:
                priv_flat = priv.reshape(B, -1).to(pooled.dtype)
                priv_flat = self._normalize_priv_obs(priv_flat, update=not use_target_network)
            else:
                priv_flat = pooled.new_zeros(B, self.critic_privileged_obs_dim)
            parts.insert(2, priv_flat)  # [pooled, state_flat, priv_obs, act]
        return torch.cat(parts, dim=-1)

    def sac_forward_critic(
        self,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        task_ids: Optional[torch.Tensor] = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Compute Q-values for (state, action) pairs.

        ``task_ids`` is accepted for interface parity but ignored (the cross-attn /
        mean-pool critic is not task-routed).
        """
        del task_ids
        assert self.critic_heads is not None, "Call sac_init() first"
        heads = self.target_critic_heads if use_target_network else self.critic_heads
        # Toggle ONLY the critic params -- never wrap in no_grad, so gradients still
        # flow to the action input (needed for the actor update).
        for p in heads.parameters():
            p.requires_grad_(requires_grad)
        if self.critic_pool_proj is not None:
            proj = self.target_pool_proj if use_target_network else self.critic_pool_proj
            for p in proj.parameters():
                p.requires_grad_(requires_grad)
        if self.critic_pooling == "attn":
            attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
            for p in attn.parameters():
                p.requires_grad_(requires_grad)
            tok = self.target_state_token if use_target_network else self.critic_state_token
            tok.requires_grad_(requires_grad)

        critic_input = self._critic_input(a, state_features, use_target_network)
        q_vals = torch.cat([h(critic_input) for h in heads], dim=-1)  # (B, num_heads)

        if method == "min":
            return q_vals.min(dim=-1).values
        return q_vals

    # ------------------------------------------------------------------
    # BC loss + target update
    # ------------------------------------------------------------------

    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: Optional[torch.nn.Module] = None,
        actions: Optional[dict[str, torch.Tensor]] = None,
        valids: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unified SFT / BC loss (``SupportSFTTraining`` contract).

        Builds the SAME processor inputs + state features as ``sac_sample_actions``
        (single source of truth via ``_prepare_inputs`` -> ``_state_features_impl``),
        then applies true flow-matching velocity BC (``_bc_mse``), matching GR00T's
        ``Gr00tN1d6ActionHead.forward`` (t: noise→action). ``target_values`` is
        accepted for interface parity but unused.
        """
        del target_values
        if actions is None or valids is None:
            raise ValueError("Gr00tN1d6ForSAC.sft_loss requires both `actions` and `valids`.")
        s, raw_state_groups = self._prepare_inputs(obs, tokenizer)
        state_features = self._state_features_impl(s)
        return self._bc_mse(
            state_features,
            actions,
            valids,
            action_mask=action_mask,
            raw_state_groups=raw_state_groups,
        )

    def _demo_action_normalized(
        self,
        actions: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        raw_state_groups: Optional[Any] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve demo actions into NORMALISED model space + a dense action mask.

        Preference order (aligned with pi0 / critic):
          1. ``full_action`` — already normalised ``(B, H, max_action_dim)`` from
             replay / rollout; mask covers real ``action_dim`` over the provided
             horizon (clamped to ``action_horizon``).
          2. env-space ``action`` — normalised via the checkpoint processor
             (``encode_actions_flat``), which needs ``raw_state_groups`` for
             relative-action embodiments. Raises if state groups are missing.
        """
        if "full_action" in actions:
            demo = actions["full_action"].to(device=device, dtype=dtype)
            B, H, D = demo.shape
            if H > self.action_horizon:
                demo = demo[:, : self.action_horizon]
                H = self.action_horizon
            elif H < self.action_horizon:
                demo = F.pad(demo, (0, 0, 0, self.action_horizon - H), value=0.0)
            if D < self.max_action_dim:
                demo = F.pad(demo, (0, self.max_action_dim - D), value=0.0)
            elif D > self.max_action_dim:
                demo = demo[..., : self.max_action_dim]
            mask = torch.zeros(
                (B, self.action_horizon, self.max_action_dim), device=device, dtype=dtype
            )
            real_h = min(H, self.action_horizon)
            mask[:, :real_h, : self.action_dim] = 1.0
            return demo, mask

        if "action" not in actions:
            raise KeyError(
                "Gr00tN1d6ForSAC BC requires `full_action` (normalised) or env-space "
                "`action` in the actions dict."
            )
        if raw_state_groups is None:
            raise ValueError(
                "Env-space `action` demos require raw_state_groups from `_prepare_inputs` "
                "so relative actions can be normalised via the GR00T processor. Pass "
                "`full_action` (replay) or ensure sft_loss builds state groups."
            )
        env_act = actions["action"]
        if env_act.ndim != 3:
            raise ValueError(f"actions['action'] must be (B, horizon, dim), got {tuple(env_act.shape)}")
        # Truncate / pad horizon to action_horizon before encode (processor pads too).
        H = env_act.shape[1]
        if H > self.action_horizon:
            env_act = env_act[:, : self.action_horizon]
        elif H < self.action_horizon:
            env_act = F.pad(env_act, (0, 0, 0, self.action_horizon - H), value=0.0)
        norm_np, mask_np = self._get_adapter().encode_actions_flat(
            env_act.detach().float().cpu().numpy(),
            raw_state_groups,
            max_action_dim=self.max_action_dim,
            max_action_horizon=self.action_horizon,
        )
        demo = torch.as_tensor(norm_np, device=device, dtype=dtype)
        mask = torch.as_tensor(mask_np, device=device, dtype=dtype)
        return demo, mask

    def _predict_flow_velocity(
        self,
        sf: dict[str, torch.Tensor],
        x_t: torch.Tensor,
        t_discretized: torch.Tensor,
    ) -> torch.Tensor:
        """Single DiT velocity prediction at discrete timestep ``t_discretized``.

        Mirrors one step of ``_run_flow`` / ``Gr00tN1d6ActionHead.forward`` without
        the Euler / SDE integration — used by flow-matching BC.
        """
        ah = self.action_head
        vl_embeds = sf["backbone_features"]
        attn_mask = sf["backbone_attention_mask"]
        image_mask = sf.get("image_mask")
        state_features = sf["state_features"]
        emb_id = sf["embodiment_id"]
        device = vl_embeds.device

        action_features = ah.action_encoder(x_t, t_discretized, emb_id)
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + ah.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        if self.use_alternate_vl_dit:
            model_output = ah.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_discretized,
                image_mask=image_mask,
                backbone_attention_mask=attn_mask,
            )
        else:
            model_output = ah.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_discretized,
            )
        pred = ah.action_decoder(model_output, emb_id)
        return pred[:, -self.action_horizon :]

    def _bc_mse(
        self,
        state_features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        raw_state_groups: Optional[Any] = None,
    ) -> torch.Tensor:
        """Flow-matching velocity BC (GR00T convention: t noise→action).

        Matches ``Gr00tN1d6ActionHead.forward``::

            x_t = (1 - t) * noise + t * actions
            u_t = actions - noise
            loss = MSE(pred_velocity, u_t)  (masked)

        Demo actions are taken from normalised ``full_action`` when present, else
        env-space ``action`` is processor-normalised (pi0-style). Optional
        ``action_mask`` may be ``(B, H)`` (horizon) or ``(B, H, D)``; combined with
        the dense pad mask from ``_demo_action_normalized``.
        """
        vl = state_features["backbone_features"]
        device, dtype = vl.device, vl.dtype
        demo, dense_mask = self._demo_action_normalized(
            actions, device=device, dtype=dtype, raw_state_groups=raw_state_groups
        )
        B = demo.shape[0]

        noise = torch.randn(demo.shape, device=device, dtype=dtype)
        # Prefer the action head's beta time sampler when available (bit-aligned with
        # GR00T SFT); fall back to Uniform(0,1) for unit tests / stripped heads.
        ah = self.action_head
        if hasattr(ah, "sample_time"):
            t = ah.sample_time(B, device=device, dtype=dtype)
        else:
            t = torch.rand(B, device=device, dtype=dtype)
        t_exp = t[:, None, None]
        x_t = (1.0 - t_exp) * noise + t_exp * demo
        u_t = demo - noise
        t_disc = (t * self.num_timestep_buckets).long()

        pred_v = self._predict_flow_velocity(state_features, x_t, t_disc)
        per_elem = F.mse_loss(pred_v, u_t, reduction="none")  # (B, H, D)

        mask = dense_mask
        if action_mask is not None:
            am = action_mask.to(device=device, dtype=dtype)
            if am.ndim == 2:
                # (B, H) -> broadcast over action dim
                if am.shape[1] != self.action_horizon:
                    if am.shape[1] < self.action_horizon:
                        am = F.pad(am, (0, self.action_horizon - am.shape[1]), value=0.0)
                    else:
                        am = am[:, : self.action_horizon]
                am = am.unsqueeze(-1).expand_as(mask)
            elif am.ndim == 3:
                if am.shape[1] != self.action_horizon or am.shape[2] != self.max_action_dim:
                    # Align to (B, action_horizon, max_action_dim)
                    h = min(am.shape[1], self.action_horizon)
                    d = min(am.shape[2], self.max_action_dim)
                    aligned = torch.zeros_like(mask)
                    aligned[:, :h, :d] = am[:, :h, :d]
                    am = aligned
            else:
                raise ValueError(f"action_mask must be (B,H) or (B,H,D), got {tuple(am.shape)}")
            mask = mask * am

        masked = per_elem * mask
        # Per-sample mean over valid elems, then mean over valid batch rows (like pi0).
        denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
        sample_loss = masked.sum(dim=(1, 2)) / denom
        valid_f = valids.to(device=device, dtype=sample_loss.dtype)
        return (sample_loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    def sac_update_target_network(self, tau: float):
        assert self.critic_heads is not None, "Call sac_init() first"
        for p_online, p_target in zip(
            self.critic_heads.parameters(), self.target_critic_heads.parameters(), strict=True
        ):
            p_target.data.lerp_(p_online.data, tau)
        if self.critic_pool_proj is not None:
            for p_online, p_target in zip(
                self.critic_pool_proj.parameters(), self.target_pool_proj.parameters(), strict=True
            ):
                p_target.data.lerp_(p_online.data, tau)
        if self.critic_pooling == "attn":
            for p_online, p_target in zip(
                self.critic_prefix_cross_attn.parameters(), self.target_prefix_cross_attn.parameters(), strict=True
            ):
                p_target.data.lerp_(p_online.data, tau)
            for p_online, p_target in zip(
                self.critic_state_token.parameters(), self.target_state_token.parameters(), strict=True
            ):
                p_target.data.lerp_(p_online.data, tau)


def load_gr00t_n1d6_for_sac(path, *, config, torch_dtype):
    """Load ``Gr00tN1d6ForSAC`` via native ``from_pretrained`` (no AutoClass registry).

    Applies the same Eagle/tokenizer compat patches and meta-init Beta workaround used
    by the SFT trainable loader. Called from ``build_vla_model`` when
    ``override_config.sac_enable`` or ``policy_type=arena``.
    """
    from transformers.modeling_utils import no_init_weights

    from .trainable_model import _BETA_PATCH_LOCK, _CpuBeta

    apply_gr00t_compat_patches()

    import gr00t.model.gr00t_n1d6.gr00t_n1d6 as upstream_model

    with _BETA_PATCH_LOCK, no_init_weights():
        original_beta = upstream_model.Beta
        upstream_model.Beta = _CpuBeta
        try:
            return Gr00tN1d6ForSAC.from_pretrained(path, config=config, torch_dtype=torch_dtype)
        finally:
            upstream_model.Beta = original_beta


def register_gr00t_sac() -> None:
    """Deprecated: prefer ``load_gr00t_n1d6_for_sac`` / ``build_vla_model``.

    Kept for ad-hoc notebooks that still call AutoModel. Do not use in verl-vla
    workers — native architecture loading no longer goes through AutoClass.
    """
    from transformers import AutoModel

    apply_gr00t_compat_patches()
    config_cls = Gr00tN1d6.config_class
    try:
        AutoModel.register(config_cls, Gr00tN1d6ForSAC, exist_ok=True)
    except TypeError:
        AutoModel._model_mapping._extra_content.pop(config_cls, None)
        AutoModel.register(config_cls, Gr00tN1d6ForSAC)
    logger.warning("register_gr00t_sac is deprecated; use build_vla_model / load_gr00t_n1d6_for_sac")
