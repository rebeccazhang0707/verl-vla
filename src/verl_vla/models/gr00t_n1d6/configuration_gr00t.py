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

"""SAC config surface for GR00T N1.6.

At runtime ``Gr00tN1d6ForSAC`` is loaded from a gr00t checkpoint whose config is
gr00t's own ``Gr00tN1d6Config``; the SAC / critic / Flow-SDE / policy fields below
are merged onto that config via the ``model.override_config`` yaml
(``model/override/gr00t.yaml``) and read with :func:`cfg_get`.

:class:`Gr00tSACConfig` documents that contract as a standalone, gr00t-free
``PretrainedConfig`` (usable for typing / defaults) so the fields live in one place
and this module imports without gr00t installed.
"""

from transformers import PretrainedConfig

# Default SAC / critic / Flow-SDE / policy fields. Mirrors ``model/override/gr00t.yaml``
# and the ``_cfg(...)`` fallbacks in ``modeling_gr00t_sac.py`` so both stay in sync.
GR00T_SAC_CONFIG_DEFAULTS: dict = {
    # --- routing ---
    "sac_enable": False,
    "policy_type": "arena",
    "critic_type": "cross_attn",  # -> internal critic_pooling="attn"; "mean_pool" -> "mean"
    "embodiment_tag": "gr1",
    # env action chunk length executed per interaction (<= action_horizon)
    "num_action_chunks": 16,
    # real (unpadded) env action width + projector index (not in the gr00t config)
    "action_dim": 26,
    "embodiment_id": 20,
    # --- critic dims ---
    "critic_head_num": 10,
    "critic_prefix_attn_heads": 8,
    "critic_action_dim": None,      # default: action_dim
    "critic_action_horizon": None,  # default: num_action_chunks (executed chunk)
    "critic_layernorm": False,
    # --- vision / io freezing ---
    "freeze_vision_tower": True,
    "freeze_action_io": False,
    # --- Flow-SDE (stochastic flow sampler) ---
    "flow_sde_enable": False,
    "flow_sde_noise_level": 0.065,
    "flow_sde_rollout_noise_scale": 1.0,
    "flow_sde_train_noise_scale": 1.0,
    "flow_sde_initial_beta": 1.0,
    "flow_sde_beta_min": 0.02,
    "flow_sde_beta_schedule_T": 4000,
}


def cfg_get(config, name: str, default=None):
    """Read a (possibly ``None``) attribute off an HF config with a fallback.

    Treats an explicit ``None`` on the config as "unset" so gr00t configs (which
    do not carry SAC fields) fall through to ``default`` / the yaml override.
    """
    val = getattr(config, name, None)
    return default if val is None else val


class Gr00tSACConfig(PretrainedConfig):
    """Standalone documentation/defaults holder for the GR00T SAC fields.

    Not the runtime config class (that stays gr00t's ``Gr00tN1d6Config``); provided
    so the SAC contract is importable and greppable without gr00t installed.
    """

    model_type = "gr00t_sac"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, default in GR00T_SAC_CONFIG_DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default))
