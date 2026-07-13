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

"""Process-wide compatibility shims for loading GR00T N1.6 under stock verl / transformers.

All patches are idempotent and best-effort; call :func:`apply_gr00t_compat_patches`
from ``load_gr00t_n1d6_policy`` before model instantiation.

Retained patches (and why):

* :func:`patch_eagle_compat` — Eagle remote code vs transformers 4.51.3; required to load.
* :func:`patch_fsdp2_interpolate_for_dtensor` — FSDP2 DTensor + SigLIP2 antialias upsample.
* :func:`disable_cudnn_sdpa` — Hopper/H100 cuDNN SDPA ``CUDNN_STATUS_NOT_INITIALIZED``.

Removed (not on the current VLA path):

* ``patch_hf_tokenizer_for_processor_only_models`` — official GR00T scripts set
  ``load_tokenizer=False``; tokenizer comes from the processor.
* ``patch_verl_monkey_patch_for_custom_vla_configs`` — ``VLAFSDPEngine._build_module``
  never calls stock verl ``apply_monkey_patch``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def patch_eagle_compat() -> None:
    """Make the Eagle remote code load under transformers 4.51.3 (idempotent).

    GR00T N1.6's vision-language backbone (Eagle) is loaded via gr00t remote code
    (``gr00t.model.modules.eagle_backbone``) and was written against older
    transformers. On 4.51.3 two incompatibilities show up without this shim:

    * ``PretrainedConfig._attn_implementation_autoset`` is read internally but not
      set by the Eagle code -> ``AttributeError``.
    * ``AutoModel.from_config`` must be forced to ``attn_implementation="flash_attention_2"``
      so Eagle submodules do not fall back to an incompatible or slower attention path.
    """
    try:
        from transformers import PretrainedConfig

        if not hasattr(PretrainedConfig, "_attn_implementation_autoset"):
            PretrainedConfig._attn_implementation_autoset = False
        import gr00t.model.modules.eagle_backbone as _eb

        if not getattr(_eb.AutoModel.from_config, "_attn_patched", False):
            _orig = _eb.AutoModel.from_config

            def _patched(config, **kw):
                kw["attn_implementation"] = "flash_attention_2"
                return _orig(config, **kw)

            _patched._attn_patched = True
            _eb.AutoModel.from_config = _patched
    except Exception as e:  # pragma: no cover - best-effort compat shim
        logger.warning("Eagle compat patch skipped: %s", e)


def disable_cudnn_sdpa() -> None:
    """Disable the cuDNN SDPA backend; keep flash / mem-efficient / math enabled.

    PyTorch ``scaled_dot_product_attention`` can route through several backends
    (cuDNN, flash, memory-efficient, math). On Hopper GPUs (H100, etc.) the cuDNN
    SDPA path in our torch/CUDA stack can fail with
    ``CUDNN_STATUS_NOT_INITIALIZED``. This toggles backends process-wide:

    * ``enable_cudnn_sdp(False)``
    * ``enable_flash_sdp`` / ``enable_mem_efficient_sdp`` / ``enable_math_sdp`` -> True
    """
    try:
        cuda_be = torch.backends.cuda
        if hasattr(cuda_be, "enable_cudnn_sdp"):
            cuda_be.enable_cudnn_sdp(False)
            if hasattr(cuda_be, "enable_flash_sdp"):
                cuda_be.enable_flash_sdp(True)
            if hasattr(cuda_be, "enable_mem_efficient_sdp"):
                cuda_be.enable_mem_efficient_sdp(True)
            if hasattr(cuda_be, "enable_math_sdp"):
                cuda_be.enable_math_sdp(True)
            logger.info("Disabled cuDNN SDPA backend (Hopper CUDNN_STATUS_NOT_INITIALIZED workaround)")
    except Exception as e:  # pragma: no cover - best-effort backend toggle
        logger.warning("Could not disable cuDNN SDPA backend: %s", e)


def patch_fsdp2_interpolate_for_dtensor() -> None:
    """Materialize FSDP2 DTensors before antialiased bilinear upsample.

    Eagle/SigLIP2 ``resize_positional_embeddings`` calls
    ``F.interpolate(..., mode="bilinear", antialias=True)``. Under FSDP2 the
    positional embedding weight is a DTensor, and PyTorch 2.7 has no sharding
    strategy for ``aten._upsample_bilinear2d_aa``, which aborts the first
    training step. All-gathering the tiny embedding preserves numerics.
    """
    import torch.nn.functional as F

    if getattr(F.interpolate, "_verl_vla_dtensor_patch", False):
        return

    _orig = F.interpolate

    def interpolate(input, *args, **kwargs):  # noqa: A001 - match torch API
        if kwargs.get("antialias", False) and hasattr(input, "full_tensor"):
            input = input.full_tensor()
        return _orig(input, *args, **kwargs)

    interpolate._verl_vla_dtensor_patch = True
    F.interpolate = interpolate
    logger.info("Patched F.interpolate for FSDP2 DTensor antialias upsample (GR00T/SigLIP2)")


def apply_gr00t_compat_patches() -> None:
    """Apply all GR00T load-time shims (idempotent). Call before model registration."""
    disable_cudnn_sdpa()
    patch_eagle_compat()
    patch_fsdp2_interpolate_for_dtensor()
