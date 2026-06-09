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

"""``Gr00tN1d6ForSAC``: GR00T N1.6 (``Gr00tN1d6``) **subclass** with SAC heads.

Ported into ``verl_vla`` and upgraded to the package's **current** SAC model
interface (the one ``SACTrainingWorker`` / ``SupportSACTraining`` expect), so it
plugs into the shared loader / FSDP / checkpoint plumbing exactly like
``PI0ForActionPrediction``:

  - ``sac_forward_state_features(obs: DataProto, tokenizer)`` — preprocessing now
    lives on the **model** side of the interface. The GR00T eagle tensors are
    still packed by the rollout (``images / lang_tokens / lang_masks / states``)
    and arrive in ``obs.batch``; ``tokenizer`` is accepted for signature parity
    but **ignored** (GR00T uses its own processor on the rollout side).
  - ``sac_forward_actor(sf, task_ids=None, ...)`` /
    ``sac_forward_critic(a, sf, task_ids=None, *, ...)`` accept ``task_ids`` for
    signature parity with the multi-task pi0 interface but **ignore** it — Arena
    is single-task. The critic keeps the inline :class:`CriticMLP` ensemble (it
    is *not* wired into pi0's pluggable ``critic_api``).
  - Replay/worker now use a single ``action`` key, so the critic input and the
    BC demo read ``a["action"]`` (was ``a["full_action"]``).
  - ``sac_sample_actions`` / ``sac_get_critic_value`` mirror the pi0 entry-point
    names for the Phase 3 rollout. See their docstrings for the GR00T return
    contract (raw normalised tensors / dict, *not* a ``Pi0Output`` wrapper).

Design notes carried over from the source implementation:
  - This is a *subclass* of gr00t's ``Gr00tN1d6`` (not an external wrapper), so
    ``AutoModel.from_pretrained`` returns it directly once registered
    (see ``register_gr00t_sac``). ``_build_model_optimizer`` then FSDP-wraps +
    checkpoints it with no special casing.
  - Critic heads are built in ``__init__`` (gated by ``config.sac_enable``) so
    FSDP wraps them. ``sac_init()`` only registers the FSDP forward methods
    (with a lazy head-build fallback for the no-FSDP smoke test).
  - ``sac_forward_actor`` runs a **grad-enabled** flow-matching denoiser that
    replicates ``Gr00tN1d6ActionHead.get_action_with_features`` (which is
    ``@torch.no_grad`` and unusable for the actor), so ``-Q`` can backprop into
    the actor.
  - ``_denoise`` supports the **Flow-SDE** sampler (arXiv:2510.25889): when
    ``flow_sde_enable`` it injects per-step Gaussian noise and returns a tractable
    ``log_prob``, enabling *maximum-entropy* SAC. GR00T integrates noise→action
    (t: 0→1), the opposite of pi0_torch, so the SDE posterior is applied under
    ``s = 1 - t``; with ``noise_scale == 0`` it reduces bit-identically to the
    deterministic Euler ODE so the eval path is unchanged.
  - ``sac_forward_critic`` toggles only the critic-head ``requires_grad`` (never
    wraps the forward in ``torch.no_grad``), so gradients always reach the action
    input — matching the pi0_torch reference.

Dims are read from the model config (this export: action_horizon=50,
max_state_dim=128, max_action_dim=128, backbone_embedding_dim=2048,
use_alternate_vl_dit=True, add_pos_embed=True, num_inference_timesteps=4).
``action_dim`` / ``embodiment_id`` are NOT in the gr00t config — supply them via
``override_config`` in the yaml (defaults: 26 / 20 for GR1).

"""

from __future__ import annotations

import contextlib
import copy
import logging
import math
import os
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing this module requires gr00t (training Docker image only). Callers that
# may run without gr00t (e.g. register_vla_models) must guard the import.
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from transformers.feature_extraction_utils import BatchFeature

from verl_vla.models.base import SupportSACTraining, SupportSFTTraining
from verl_vla.models.gr00t.utils import GR1, GR00TDim

if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)


class CriticMLP(nn.Module):
    """Two-layer MLP: input_dim → 512 → 256 → 1."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _cfg(config, name, default):
    """Read a (possibly int) attribute off the HF config with a fallback."""
    val = getattr(config, name, None)
    return default if val is None else val


def beta_schedule(step: int, beta0: float, beta_min: float, T: int) -> float:
    """Cosine anneal of the Flow-SDE exploration scale (mirrors pi0_torch)."""
    progress = min(step / T, 1.0)
    return beta_min + (beta0 - beta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


class Gr00tN1d6ForSAC(Gr00tN1d6, SupportSACTraining, SupportSFTTraining):
    """``Gr00tN1d6`` augmented with SAC critic heads.

    Loaded by ``AutoModel.from_pretrained`` once ``register_gr00t_sac()`` has
    overridden gr00t's own registration for the ``Gr00tN1d6`` config class.
    """

    def __init__(self, config):
        super().__init__(config)

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

        # real (unpadded) GR1 action width used for the SAC critic and the env;
        # NOT present in the gr00t config → comes from override_config / default.
        self.action_dim = int(_cfg(config, "action_dim", GR1.action_dim))
        self.embodiment_id = int(_cfg(config, "embodiment_id", GR1.embodiment_id))
        self.num_critic_heads = int(_cfg(config, "critic_head_num", 10))

        # how much of the action chunk the critic sees
        self.critic_action_dim = int(_cfg(config, "critic_action_dim", self.action_dim))
        self.critic_action_horizon = int(_cfg(config, "critic_action_horizon", self.action_horizon))

        # --- Critic representation options (default = legacy: masked mean-pool + raw state) ---
        # critic_pooling:
        #   "mean" – masked mean-pool over the VL token sequence (param-free, legacy default).
        #   "attn" – learnable cross-attention pool: a ``critic_state_token`` query attends over
        #            the VL tokens (à la pi0.5). Output dim stays backbone_feature_dim so
        #            critic_input_dim is UNCHANGED; gives the critic capacity to focus on
        #            task-relevant tokens instead of averaging them away.
        # critic_use_encoded_state:
        #   False – raw flat state (max_state_dim per step; mostly zero-padding for GR1), legacy.
        #   True  – the actor's state_encoder output (input_embedding_dim per step; dense,
        #           embodiment-aware). Already computed in the same forward (free). This CHANGES
        #           critic_input_dim, so a critic trained with one setting cannot load a
        #           checkpoint trained with the other.
        self.critic_pooling = str(_cfg(config, "critic_pooling", "mean")).lower()
        self.critic_use_encoded_state = bool(_cfg(config, "critic_use_encoded_state", False))
        self.critic_prefix_attn_heads = int(_cfg(config, "critic_prefix_attn_heads", 8))
        # state_encoder output width (used only when reusing the encoded state)
        self._state_feature_dim = int(getattr(self.action_head, "input_embedding_dim", self.max_state_dim))

        # Asymmetric actor-critic: privileged critic obs (object pose + fridge door joint) fed
        # ONLY to the critic. Appends `critic_privileged_obs_dim` inputs to the critic MLP; the
        # actor never sees it. Changes critic_input_dim ⇒ critic must train fresh (resume_mode
        # is disabled for arena). Default off ⇒ bit-identical critic for pi0.5/libero/isaac.
        self.critic_privileged_obs = bool(_cfg(config, "critic_privileged_obs", False))
        self.critic_privileged_obs_dim = (
            int(_cfg(config, "critic_privileged_obs_dim", 0)) if self.critic_privileged_obs else 0
        )

        # critic input = pooled backbone + flat state + flat (sliced) action [+ privileged obs]
        critic_state_width = self._state_feature_dim if self.critic_use_encoded_state else self.max_state_dim
        self.critic_input_dim = (
            self.backbone_feature_dim
            + self.state_horizon * critic_state_width
            + self.critic_action_horizon * self.critic_action_dim
            + self.critic_privileged_obs_dim
        )

        # --- Flow-SDE (stochastic flow sampler) config; see _denoise. When
        #     disabled the sampler reduces *bit-identically* to the deterministic
        #     Euler ODE (x = x + dt*v) and log_probs are None (DDPG-style -Q). ---
        self.flow_sde_enable = bool(_cfg(config, "flow_sde_enable", False))
        self.flow_sde_noise_level = float(_cfg(config, "flow_sde_noise_level", 0.065))
        self.flow_sde_rollout_noise_scale = float(_cfg(config, "flow_sde_rollout_noise_scale", 1.0))
        self.flow_sde_train_noise_scale = float(_cfg(config, "flow_sde_train_noise_scale", 1.0))
        self.flow_sde_initial_beta = float(_cfg(config, "flow_sde_initial_beta", 1.0))
        self.flow_sde_beta_min = float(_cfg(config, "flow_sde_beta_min", 0.02))
        self.flow_sde_beta_schedule_T = int(_cfg(config, "flow_sde_beta_schedule_T", 4000))
        self.register_buffer("flow_sde_step", torch.zeros((), dtype=torch.long))

        # --- Trainable action dims (online RL action mask) ---
        # ``sac_action_train_dims``: list of [start, end) half-open ranges in the flat
        # action space (GR1 26-DOF order: left_arm 0:7, right_arm 7:14, left_hand 14:20,
        # right_hand 20:26). RL exploration + the actor gradient + log-prob are restricted
        # to these dims; all other dims follow the model's DETERMINISTIC (no-noise) output,
        # detached (frozen base GR00T behaviour). None/unset ⇒ train ALL dims (default,
        # bit-identical to before). Example right_arm+right_hand: [[7,14],[20,26]].
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

        # Build heads here (pre-FSDP) when SAC is enabled, so FSDP wraps them and
        # the checkpoint manager saves them — mirrors PI0ForActionPrediction.
        if bool(getattr(config, "sac_enable", False)):
            self._build_sac_heads()

        # Freeze the Eagle vision tower for SAC (mirrors pi0.5). The GR00T fine-tune path
        # leaves vision trainable (backbone tune_visual defaults True), but for online RL we
        # keep it frozen so only the LLM-conditioned action head + critic adapt. Done here
        # (pre-FSDP) so the actor optimizer / EMA — which filter by requires_grad — never see
        # vision params. Toggle off with freeze_vision_tower=False (legacy: vision follows the
        # backbone config's own tune_visual).
        self.freeze_vision_tower_enabled = bool(_cfg(config, "freeze_vision_tower", False))
        if self.freeze_vision_tower_enabled:
            self.freeze_vision_tower()

    # ------------------------------------------------------------------
    # SAC head construction
    # ------------------------------------------------------------------

    def _build_sac_heads(self):
        self.critic_heads = nn.ModuleList(
            [CriticMLP(self.critic_input_dim) for _ in range(self.num_critic_heads)]
        )
        self.target_critic_heads = copy.deepcopy(self.critic_heads)
        for p in self.target_critic_heads.parameters():
            p.requires_grad_(False)

        # Optional learnable cross-attention pooling (mirrors pi0.5). A single learnable
        # ``critic_state_token`` query attends over the VL token sequence; the target copy is
        # frozen and Polyak-tracked alongside the critic heads. Default ("mean") keeps the
        # param-free masked mean-pool from sac_forward_state_features → zero behaviour change.
        if self.critic_pooling == "attn":
            d = self.backbone_feature_dim
            # Learnable cross-attn query token. Held as nn.Embedding (NOT a bare nn.Parameter):
            # under from_pretrained's _fast_init, _init_weights only reaches params owned by a
            # SUB-module (dotted state-dict key like `critic_state_token.weight`). A bare root
            # Parameter (no-dot key) is never mapped to a module → stays uninitialized → NaN
            # query → NaN critic. The Embedding's .weight is (1, d) — used as the query token.
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
            "Gr00tN1d6ForSAC: %d critic heads, input_dim=%d "
            "(backbone=%d, state=%d, action=%dx%d), horizon=%d, max_action_dim=%d",
            self.num_critic_heads, self.critic_input_dim, self.backbone_feature_dim,
            self.state_horizon * self.max_state_dim, self.critic_action_horizon,
            self.critic_action_dim, self.action_horizon, self.max_action_dim,
        )

    def freeze_vision_tower(self) -> None:
        """Freeze the Eagle vision tower (+ vision→LLM connector) for SAC, à la pi0.5.

        Targets ``backbone.eagle_model.vision_model`` (SiglipVisionModel) and the ``mlp1``
        connector — the same modules Eagle's own ``set_trainable_parameters(tune_visual=False)``
        freezes. Sets ``requires_grad=False`` (so the actor optimizer / EMA, which filter by
        requires_grad, skip them) and puts them in eval mode. Idempotent and defensive: warns
        and no-ops if the path is absent (e.g. a CPU mock backbone without eagle_model).
        """
        eagle = getattr(self.backbone, "eagle_model", None)
        vision_model = getattr(eagle, "vision_model", None) if eagle is not None else None
        if vision_model is None:
            logger.warning(
                "[gr00t-sac] backbone.eagle_model.vision_model not found; skipping freeze_vision_tower"
            )
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

    def _init_weights(self, module):
        """Initialize newly-added SAC critic-head Linears.

        ``from_pretrained`` (with ``_fast_init``) builds the model on the meta
        device and only calls ``_init_weights`` on params absent from the
        checkpoint — i.e. the critic heads. Without this, those ``nn.Linear``
        params are materialized as uninitialized (NaN) memory → NaN Q-values.
        Base GR00T modules are still handled by ``super()._init_weights`` and/or
        overwritten by the loaded checkpoint weights.
        """
        sup = getattr(super(), "_init_weights", None)
        if callable(sup):
            sup(module)
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            # out_proj (nn.Linear) is covered above; the packed in-proj params are
            # bare Parameters, so init them here too (avoids NaN under meta-device fast-init).
            if getattr(module, "in_proj_weight", None) is not None:
                torch.nn.init.xavier_uniform_(module.in_proj_weight)
            if getattr(module, "in_proj_bias", None) is not None:
                torch.nn.init.zeros_(module.in_proj_bias)
        # The cross-attn pooling query tokens (`critic_state_token` / `target_state_token`) are
        # nn.Embedding sub-modules; their `.weight` is a missing-from-checkpoint key, so
        # _init_weights IS invoked on them here (post-materialization, pre-FSDP). Without this
        # they'd materialize as NaN under meta-device _fast_init → NaN query → NaN critic.
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # SupportSACTraining interface
    # ------------------------------------------------------------------

    def sac_init(self):
        # Production path: heads already built in __init__ (sac_enable=True) and
        # FSDP-wrapped. Lazy fallback covers the no-FSDP smoke test where the
        # model was loaded without sac_enable on the config.
        if self.critic_heads is None:
            self._build_sac_heads()

        from torch.distributed.fsdp import register_fsdp_forward_method

        # Method names aligned with the pi0 interface the worker drives.
        register_fsdp_forward_method(self, "bc_loss")
        register_fsdp_forward_method(self, "sac_sample_actions")
        register_fsdp_forward_method(self, "sac_forward_critic")
        register_fsdp_forward_method(self, "sac_forward_actor")
        register_fsdp_forward_method(self, "sac_forward_state_features")
        register_fsdp_forward_method(self, "sac_update_target_network")

    def sac_get_critic_parameters(self) -> list[torch.nn.Parameter]:
        assert self.critic_heads is not None, "Call sac_init() first"
        params = list(self.critic_heads.parameters())
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
    # State features (backbone)
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_to_state_dict(obs: DataProto) -> dict[str, torch.Tensor]:
        """Pull the GR00T eagle tensors the rollout packed into ``obs.batch``.

        The replay schema stores the packed N1.6 eagle tensors under the fixed
        state-key names (see naive_rollout_gr00t.py):

            images      ← eagle pixel_values   (B, n_patches, C, H, W)
            lang_tokens ← eagle input_ids      (B, L) int64
            lang_masks  ← eagle attention_mask (B, L)
            states      ← normalised state     (B, state_horizon, max_state_dim)
        """
        batch = obs.batch
        out = {
            "images": batch["images"],
            "lang_tokens": batch["lang_tokens"],
            "lang_masks": batch["lang_masks"],
            "states": batch["states"],
        }
        # Asymmetric-AC privileged critic obs (arena only): thread it through when the env
        # emitted it (rides as obs.priv_obs → stripped to `priv_obs` here). Critic-only.
        if "priv_obs" in batch:
            out["priv_obs"] = batch["priv_obs"]
        return out

    def sac_forward_state_features(self, obs: DataProto, tokenizer=None) -> dict[str, torch.Tensor]:
        """Registered FSDP forward entry for standalone state-feature computation.

        Unpacks the eagle tensors from ``obs.batch`` and runs ``_state_features_impl``.
        ``tokenizer`` is accepted for interface parity with pi0 but ignored — GR00T
        runs its own processor on the rollout side and packs the result into ``obs``.

        ``sac_sample_actions`` / ``bc_loss`` instead call ``_state_features_impl``
        directly to avoid a NESTED registered-forward-method boundary — the inner
        method's FSDP exit re-shards the root params, leaving the action head's
        root-level weights (position_embedding, vlln, CategorySpecific*) as DTensors
        when ``_denoise`` runs, which breaks ops like embedding/bmm.
        """
        return self._state_features_impl(self._obs_to_state_dict(obs))

    def _state_features_impl(self, s: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the Eagle backbone on the stored inputs and encode the state.

        See ``_obs_to_state_dict`` for the packed-tensor layout. ``embodiment_id``
        is constant per run, so it is rebuilt from config rather than stored.
        ``image_mask`` is recomputed by the backbone.

        **None-free contract**: the returned dict must NOT contain ``None`` values.
        The worker runs ``split_nested_dicts_or_tuples(state_features, 2)`` on the
        result, which raises ``TypeError`` on ``None``. The backbone may yield a
        ``None`` ``image_mask``; in that case the key is **dropped** entirely (the
        downstream ``_run_flow`` reads it with ``sf.get("image_mask", None)``, so a
        missing key is equivalent to the original ``None``).
        """
        ah = self.action_head
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype

        # The Eagle backbone expects pixel_values as a list of per-sample
        # (n_patches, C, H, W) tensors. The replay buffer can only stack tensors,
        # so the rollout stores them stacked as (B, n_patches, C, H, W); restore
        # the per-sample list here. (Also accept a raw list, e.g. smoke test.)
        imgs = s["images"]
        if isinstance(imgs, torch.Tensor):
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

        raw_features = backbone_outputs["backbone_features"]               # (B, S, D)
        attn_mask = backbone_outputs["backbone_attention_mask"]            # (B, S) bool
        image_mask = backbone_outputs.get("image_mask", None)             # (B, S) bool or None

        vl_embeds = ah.vlln(raw_features)                                  # match action-head vlln

        # Masked mean-pool for the critic. Use torch.where (NOT vl_embeds * mask)
        # because padding positions can carry NaN/inf (the action head masks them
        # in attention, but ``0 * NaN = NaN`` would poison the pool → NaN critic).
        mask_b = attn_mask.unsqueeze(-1).bool()
        vl_safe = torch.where(mask_b, vl_embeds, torch.zeros_like(vl_embeds))
        denom = mask_b.sum(dim=1).clamp(min=1).to(vl_embeds.dtype)
        pooled = vl_safe.sum(dim=1) / denom

        state = s["states"].to(device, dtype=dtype)                        # (B, T, max_state_dim)
        B = state.shape[0]
        embodiment_id = torch.full((B,), self.embodiment_id, dtype=torch.long, device=device)
        state_features = ah.state_encoder(state, embodiment_id)            # (B, T, input_emb_dim)

        sf = {
            "pooled": pooled,
            "backbone_features": vl_embeds,
            "backbone_attention_mask": attn_mask,
            "state_features": state_features,
            "state": state,
            "embodiment_id": embodiment_id,
        }
        # None-free contract: only add image_mask when the backbone produced one.
        if image_mask is not None:
            sf["image_mask"] = image_mask
        # Asymmetric-AC privileged critic obs: carry through so split_nested_dicts keeps the
        # per-(s0,s1) alignment and _critic_input can append it. Critic-only; actor ignores it.
        if self.critic_privileged_obs and "priv_obs" in s:
            sf["priv_obs"] = s["priv_obs"].to(device, dtype=dtype)
        return sf

    # ------------------------------------------------------------------
    # Grad-enabled flow-matching denoiser (the actor sampler)
    # ------------------------------------------------------------------

    def _gaussian_log_prob(
        self, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Diagonal-Gaussian log-prob, mean-reduced over (horizon, action_dim).

        Matches pi0_torch._gaussian_log_prob exactly so the SAC entropy term has
        the same scale across backends. NB: the mean (not sum) reduction means the
        log-prob is O(num_steps); ``target_entropy`` must be tuned accordingly.
        """
        std_safe = std.clamp_min(1e-6)
        log_prob = -0.5 * (((sample - mean) / std_safe) ** 2 + 2.0 * torch.log(std_safe) + math.log(2.0 * math.pi))
        if self.sac_action_train_all:
            return log_prob.mean(dim=(-1, -2))
        # Average only over trainable action dims (the frozen dims carry no noise and
        # their per-dim log-prob is meaningless); keep the mean-over-(horizon, dims) scale.
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

    def _denoise(
        self,
        sf: dict[str, torch.Tensor],
        *,
        noise_scale: float,
        requires_grad: bool,
        return_log_prob: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample a (B, action_horizon, max_action_dim) action via the flow sampler.

        When training ALL dims this is a single trajectory. When only a subset is
        trainable (``sac_action_train_dims``, e.g. right_arm+right_hand) it runs TWO
        decoupled passes from the SAME initial noise:

          * explored pass  — full Flow-SDE trajectory; we keep its TRAINABLE dims
            (these carry the actor gradient + log-prob);
          * base pass      — fully deterministic (no noise, no grad); we keep its
            FROZEN dims.

        This is what stops the frozen dims (left_arm/left_hand) from being jittered by
        the right-side exploration: a single shared trajectory couples all dims through
        the flow DiT velocity, so the only way to keep the left arm at the clean base
        output is to read it from a separate noise-free trajectory.
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
        **noise (t=0) → action (t=1)** with ``x = x + dt*v`` (opposite to pi0_torch); we
        map onto the Flow-SDE formulas via ``s = 1 - t``. With ``noise_scale == 0`` (or
        flow-SDE disabled) the SDE mean collapses *exactly* to the deterministic Euler
        step, so the eval path is bit-identical to the original implementation.

        ``image_mask`` is read with ``.get`` so the None-free state-features dict
        (which drops the key when the backbone yields no mask) still works — a
        missing key is treated as ``None``, exactly as before.
        """
        ah = self.action_head
        vl_embeds = sf["backbone_features"]
        attn_mask = sf["backbone_attention_mask"]
        image_mask = sf.get("image_mask", None)
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
                v = pred[:, -self.action_horizon:]

                if use_sde:
                    # Linear-path endpoints implied by the (locally constant) velocity:
                    #   data_pred  = action endpoint (t=1),  noise_pred = noise endpoint (t=0).
                    data_pred = x + v * (1.0 - t_cur)
                    noise_pred = x - v * t_cur
                    # s = 1 - t  ⇒  pi0_torch's "t_cur"/"t_next" become s_cur/s_next.
                    s_cur = min(max(1.0 - t_cur, 1e-4), 1.0 - 1e-4)
                    s_next = 1.0 - t_next
                    sigma_schedule = self.flow_sde_noise_level * noise_scale * math.sqrt(s_cur / (1.0 - s_cur))
                    sigma = beta * sigma_schedule                          # 0-dim tensor
                    data_w = 1.0 - s_next                                  # == t_next
                    noise_w = s_next - (sigma.pow(2) * delta) / (2.0 * s_cur)
                    x_mean = data_pred * data_w + noise_pred * noise_w
                    sigma_t = (torch.sqrt(torch.as_tensor(delta, device=device, dtype=dtype)) * sigma).view(1, 1, 1)
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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, float]]:
        """Grad-enabled action sampling for the actor update.

        ``task_ids`` is accepted for interface parity with the multi-task pi0
        actor but **ignored** — Arena is single-task.

        With flow-SDE enabled this returns a tractable per-sample ``log_prob`` so
        the SAC actor loss is the full ``alpha*log_prob - Q`` (maximum entropy)
        and the critic's soft target / alpha auto-tuning become active. With
        flow-SDE disabled ``log_probs`` is None and the loss reduces to ``-Q``
        (DDPG-style deterministic policy gradient).
        """
        del task_ids  # single-task arena: ignored (signature parity only)
        noise_scale = self.flow_sde_train_noise_scale if self.flow_sde_enable else 0.0
        actions, log_probs = self._denoise(
            state_features,
            noise_scale=noise_scale,
            requires_grad=True,
            return_log_prob=self.flow_sde_enable,
        )
        # Advance the beta schedule once per actor optimiser step (first micro batch).
        if is_first_micro_batch and self.flow_sde_enable:
            self.flow_sde_step.add_(1)
        metrics: dict[str, float] = {}
        if self.flow_sde_enable:
            metrics = {
                "flow_sde_beta": float(self.flow_sde_beta().item()),
                "flow_sde_step": float(self.flow_sde_step.item()),
            }
        return actions, log_probs, metrics

    @torch.no_grad()
    def sac_sample_actions(
        self,
        obs: DataProto,
        tokenizer=None,
        validate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Rollout-time action sampling (no grad); mirrors pi0's ``sac_sample_actions``.

        ``tokenizer`` is ignored (GR00T preprocessing is on the rollout side); the
        eagle tensors are read from ``obs.batch`` via ``_obs_to_state_dict``.

        Uses the Flow-SDE exploration noise when enabled and ``validate=False``
        (off-policy data collection); reduces to the deterministic ODE otherwise.

        **Return contract (GR00T)**: a plain ``dict`` (NOT a ``Pi0Output`` wrapper):
            - ``"action"``    : (B, action_horizon, max_action_dim) — the **raw,
                                normalised** model action. Decoding to the real
                                26-DOF action is the rollout's job (the GR00T
                                processor / ``GR00TN16Adapter.decode_actions_flat``);
                                this model does NOT un-normalise.
            - ``"log_probs"`` : (B,) — sampler log-prob (zeros when flow-SDE off).
        Phase 3 rollout should adopt this dict shape and the single ``action`` key.
        """
        # Call the impl directly (NOT the registered sac_forward_state_features) so
        # root params stay all-gathered through _denoise — sac_sample_actions is
        # itself the registered FSDP forward entry. See sac_forward_state_features.
        del tokenizer  # ignored (GR00T processor runs on the rollout side)
        state_features = self._state_features_impl(self._obs_to_state_dict(obs))
        noise_scale = 0.0 if validate else (self.flow_sde_rollout_noise_scale if self.flow_sde_enable else 0.0)
        action, log_probs = self._denoise(
            state_features,
            noise_scale=noise_scale,
            requires_grad=False,
            return_log_prob=self.flow_sde_enable and not validate,
        )
        if log_probs is None:
            log_probs = torch.zeros(action.shape[0], device=action.device, dtype=torch.float32)
        return {"action": action, "log_probs": log_probs}

    @torch.no_grad()
    def sac_get_critic_value(
        self,
        obs: DataProto,
        actions,
        tokenizer=None,
    ) -> torch.Tensor:
        """Min-over-heads Q for the given (obs, actions); mirrors pi0's method.

        ``actions`` may be the dict returned by ``sac_sample_actions``
        (``{"action": ...}``) or any object exposing a ``.action`` tensor.
        ``tokenizer`` / ``task_ids`` are ignored (single-task arena).
        """
        del tokenizer
        state_features = self.sac_forward_state_features(obs)
        action_tensor = actions["action"] if isinstance(actions, dict) else actions.action
        q = self.sac_forward_critic(
            {"action": action_tensor},
            state_features,
            task_ids=None,
            use_target_network=False,
            method="min",
            requires_grad=False,
        )
        return q.detach().float()

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
        # Padded token positions of vl_embeds can carry NaN/inf. key_padding_mask zeros their
        # softmax weight, but MHA still projects their values and `0 * NaN = NaN` would poison
        # the pooled output → NaN critic. Zero them first (mirrors the masked mean-pool guard).
        mask_b = attn_mask.unsqueeze(-1).to(torch.bool)
        vl_safe = torch.where(mask_b, vl_embeds, torch.zeros_like(vl_embeds))
        query = state_token.view(1, 1, -1).expand(B, -1, -1)
        key_padding_mask = ~attn_mask.to(dtype=torch.bool)     # True ⇒ ignore (padding)
        pooled, _ = cross_attn(
            query=query, key=vl_safe, value=vl_safe,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        return pooled.squeeze(1)                               # (B, D)

    def _critic_input(
        self, a: dict[str, torch.Tensor], sf: dict[str, torch.Tensor], use_target_network: bool = False
    ) -> torch.Tensor:
        if self.critic_pooling == "attn":
            pooled = self._cross_attention_pool(
                sf["backbone_features"], sf["backbone_attention_mask"], use_target_network
            )                                                  # (B, D)
        else:
            pooled = sf["pooled"]                              # (B, D)
        B = pooled.shape[0]
        state_src = sf["state_features"] if self.critic_use_encoded_state else sf["state"]
        state_flat = state_src.reshape(B, -1)                  # (B, T*state_width)
        action = a["action"]                                   # (B, horizon, max_action_dim)
        act = action[:, : self.critic_action_horizon, : self.critic_action_dim].reshape(B, -1)
        if os.environ.get("GR00T_SAC_NAN_DEBUG"):
            for name, t in (("pooled", pooled), ("state", state_flat), ("action", act)):
                if not torch.isfinite(t).all():
                    logger.error(
                        "[gr00t-sac][nan] critic_input '%s' has %d/%d non-finite "
                        "(pooling=%s, encoded_state=%s, target=%s); state_token_finite=%s",
                        name, int((~torch.isfinite(t)).sum().item()), t.numel(),
                        self.critic_pooling, self.critic_use_encoded_state, use_target_network,
                        None if self.critic_pooling != "attn"
                        else bool(torch.isfinite(
                            (self.target_state_token if use_target_network else self.critic_state_token).weight
                        ).all().item()),
                    )
        parts = [pooled, state_flat, act]
        if self.critic_privileged_obs:
            # Asymmetric AC: append privileged obs (object pose + fridge joint) to the critic
            # input only. Zero-fill if unexpectedly absent so critic_input_dim stays fixed.
            priv = sf.get("priv_obs")
            priv_flat = (
                priv.reshape(B, -1).to(pooled.dtype)
                if priv is not None
                else pooled.new_zeros(B, self.critic_privileged_obs_dim)
            )
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
        """Q-values for (a, state_features). ``task_ids`` ignored (single-task arena)."""
        del task_ids  # single-task arena: ignored (signature parity only)
        assert self.critic_heads is not None, "Call sac_init() first"
        heads = self.target_critic_heads if use_target_network else self.critic_heads
        # Toggle ONLY the critic params — never wrap in no_grad, so that
        # gradients still flow to the action input (needed for the actor update).
        for p in heads.parameters():
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

    def bc_loss(
        self,
        obs: DataProto,
        tokenizer,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between freshly sampled actions and stored (demo) actions.

        New interface: takes ``(obs, tokenizer, actions, valids)`` like pi0's
        ``bc_loss``. ``tokenizer`` is ignored. State features are computed inline
        via ``_state_features_impl`` (NOT the registered ``sac_forward_state_features``,
        to avoid a nested FSDP forward boundary). The demo action is read from the
        single ``actions["action"]`` key. Compares only the real (unpadded) dims.
        """
        del tokenizer  # ignored (GR00T processor runs on the rollout side)
        state_features = self._state_features_impl(self._obs_to_state_dict(obs))
        pred, _ = self._denoise(
            state_features, noise_scale=0.0, requires_grad=True, return_log_prob=False
        )
        demo = actions["action"].to(pred.dtype)
        d = self.action_dim
        loss = F.mse_loss(pred[..., :d], demo[..., :d], reduction="none").mean(dim=[1, 2])  # (B,)
        valid_f = valids.float().to(loss.device)
        return (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    def sac_update_target_network(self, tau: float):
        assert self.critic_heads is not None, "Call sac_init() first"
        for p_online, p_target in zip(
            self.critic_heads.parameters(), self.target_critic_heads.parameters(), strict=True
        ):
            p_target.data.lerp_(p_online.data, tau)
        if self.critic_pooling == "attn":
            for p_online, p_target in zip(
                self.critic_prefix_cross_attn.parameters(),
                self.target_prefix_cross_attn.parameters(),
                strict=True,
            ):
                p_target.data.lerp_(p_online.data, tau)
            for p_online, p_target in zip(
                self.critic_state_token.parameters(), self.target_state_token.parameters(), strict=True
            ):
                p_target.data.lerp_(p_online.data, tau)


def _patch_eagle_compat() -> None:
    """Make the Eagle remote code load under transformers 4.51.3 (idempotent).

    Mirrors the patch in eval_arena_gr00t / smoke_test_gr00t_arena so the model
    loads the same way in the SAC workers:
      * ``PretrainedConfig._attn_implementation_autoset`` shim (newer-attr access);
      * gr00t's ``eagle_backbone`` builds ``attn_implementation='flash_attention_2'``
        but never forwards it to ``AutoModel.from_config`` — inject it.
    """
    try:
        from transformers import PretrainedConfig
        if not hasattr(PretrainedConfig, "_attn_implementation_autoset"):
            PretrainedConfig._attn_implementation_autoset = False
        import gr00t.model.modules.eagle_backbone as _eb
        if not getattr(_eb.AutoModel.from_config, "_attn_patched", False):
            _orig = _eb.AutoModel.from_config

            def _patched(config, **kw):
                # Force FA2 for the Eagle backbone even if the top-level model is loaded
                # with attn_implementation="eager" (verl/from_pretrained may pass eager).
                kw["attn_implementation"] = "flash_attention_2"
                return _orig(config, **kw)

            _patched._attn_patched = True
            _eb.AutoModel.from_config = _patched
    except Exception as e:  # pragma: no cover - best-effort compat shim
        logger.warning("Eagle compat patch skipped: %s", e)


def _disable_cudnn_sdpa() -> None:
    """Force SDPA off the cuDNN attention backend (idempotent, process-wide).

    On Hopper (H20/H100, sm_90) PyTorch's scaled_dot_product_attention prefers the
    cuDNN attention backend, which fails to initialise here with
    ``cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`` (torch 2.10 / cuDNN under the CUDA
    12.8 forward-compat driver). The GR00T action-head DiT (diffusers
    ``AttnProcessor2_0``) hits this on the first ``_run_flow`` forward. Disabling the
    cuDNN SDP backend makes SDPA fall back to FlashAttention / mem-efficient / math —
    all well supported on Hopper — while leaving the L20 (Ada) path unchanged. No-op
    on torch builds without the toggle.
    """
    try:
        cuda_be = torch.backends.cuda
        if hasattr(cuda_be, "enable_cudnn_sdp"):
            cuda_be.enable_cudnn_sdp(False)
            # Keep the remaining backends available so SDPA still has a kernel to pick.
            if hasattr(cuda_be, "enable_flash_sdp"):
                cuda_be.enable_flash_sdp(True)
            if hasattr(cuda_be, "enable_mem_efficient_sdp"):
                cuda_be.enable_mem_efficient_sdp(True)
            if hasattr(cuda_be, "enable_math_sdp"):
                cuda_be.enable_math_sdp(True)
            logger.info("Disabled cuDNN SDPA backend (Hopper CUDNN_STATUS_NOT_INITIALIZED workaround)")
    except Exception as e:  # pragma: no cover - best-effort backend toggle
        logger.warning("Could not disable cuDNN SDPA backend: %s", e)


def register_gr00t_sac() -> None:
    """Register ``Gr00tN1d6ForSAC`` so ``AutoModel`` loads it for the gr00t config.

    Imports gr00t first (triggering gr00t's own ``Gr00tN1d6`` registration) and
    then overrides the mapping for that config class. Must be called only inside
    an image that has gr00t installed; ``register_vla_models`` guards the import.
    """
    from transformers import AutoModel

    _disable_cudnn_sdpa()
    _patch_eagle_compat()

    # gr00t registers Gr00tN1d6 under AutoModel at import time (auto_map is null
    # in the checkpoint, confirmed), and verl's loader falls back to AutoModel
    # for this config. Override that mapping with our subclass.
    config_cls = Gr00tN1d6.config_class
    try:
        AutoModel.register(config_cls, Gr00tN1d6ForSAC, exist_ok=True)
    except TypeError:
        # Older transformers: register() has no exist_ok kwarg. Drop the existing
        # entry first, then register.
        AutoModel._model_mapping._extra_content.pop(config_cls, None)
        AutoModel.register(config_cls, Gr00tN1d6ForSAC)
    logger.info("Registered Gr00tN1d6ForSAC under AutoModel for %s", config_cls.__name__)
