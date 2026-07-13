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

Extracted from ``modeling_gr00t_sac`` so the model file stays focused on SAC / flow
logic. All patches are idempotent and best-effort; call :func:`apply_gr00t_compat_patches`
from :func:`register_gr00t_sac` before model instantiation.
"""

from __future__ import annotations

import contextlib
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


def patch_verl_monkey_patch_for_custom_vla_configs() -> None:
    """Skip ``apply_monkey_patch`` when no Ulysses / padding / fused path is active.

    Stock ``verl==0.7.1`` always reads ``config.num_attention_heads`` (then
    ``config.text_config.*``) inside ``apply_monkey_patch``. ``Gr00tN1d6Config``
    has neither attribute, so FSDP ``_build_module`` crashes after a successful
    weight load. The legacy verl fork early-returns when nothing needs patching;
    mirror that here (idempotent).
    """
    try:
        import verl.models.transformers.monkey_patch as mp
    except ImportError:  # pragma: no cover
        return

    if getattr(mp, "_verl_vla_gr00t_monkey_patch", False):
        return

    _orig = mp.apply_monkey_patch

    def apply_monkey_patch(
        model,
        ulysses_sp_size: int = 1,
        use_remove_padding: bool = True,
        use_fused_kernels: bool = False,
        fused_kernels_backend: str = None,
        use_prefix_grouper: bool = False,
        use_tiled_mlp: bool = False,
        tiled_mlp_shards: int = 4,
        **kwargs,
    ):
        # Nothing to patch, or config lacks the LLM layout stock verl assumes
        # (Gr00tN1d6Config has neither num_attention_heads nor text_config).
        cfg = getattr(model, "config", None)
        has_llm_heads = cfg is not None and (
            hasattr(cfg, "num_attention_heads") or hasattr(cfg, "text_config")
        )
        if (
            (
                ulysses_sp_size <= 1
                and not use_remove_padding
                and not use_fused_kernels
                and not use_prefix_grouper
                and not use_tiled_mlp
            )
            or not has_llm_heads
        ):
            return None
        return _orig(
            model,
            ulysses_sp_size=ulysses_sp_size,
            use_remove_padding=use_remove_padding,
            use_fused_kernels=use_fused_kernels,
            fused_kernels_backend=fused_kernels_backend,
            use_prefix_grouper=use_prefix_grouper,
            use_tiled_mlp=use_tiled_mlp,
            tiled_mlp_shards=tiled_mlp_shards,
            **kwargs,
        )

    mp.apply_monkey_patch = apply_monkey_patch
    # Rebind import-time aliases used by the FSDP engine.
    import importlib

    for mod_name in (
        "verl.models.transformers.monkey_patch",
        "verl.workers.engine.fsdp.transformer_impl",
    ):
        with contextlib.suppress(ImportError):
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "apply_monkey_patch"):
                setattr(mod, "apply_monkey_patch", apply_monkey_patch)

    mp._verl_vla_gr00t_monkey_patch = True
    logger.info("Patched verl apply_monkey_patch for custom VLA configs (GR00T)")


def patch_hf_tokenizer_for_processor_only_models() -> None:
    """Tolerate processor-only checkpoints in stock ``verl`` ``hf_tokenizer``.

    ``verl==0.7.1`` has no ``AutoProcessor`` fallback; GR00T exports only
    ``processor_config.json``. Recover the inner Qwen2 tokenizer, and rebind
    import-time aliases (``HFModelConfig`` does ``from verl.utils import
    hf_tokenizer``).
    """
    import importlib
    import warnings

    import verl.utils.tokenizer as tok_mod

    if getattr(tok_mod, "_verl_vla_gr00t_tokenizer_patch", False):
        return

    _orig = tok_mod.hf_tokenizer

    def _from_processor(name_or_path, **kwargs):
        from transformers import AutoProcessor

        kwargs.setdefault("trust_remote_code", True)
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        return getattr(proc, "tokenizer", None) or getattr(
            getattr(proc, "processor", None), "tokenizer", None
        )

    def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
        try:
            return _orig(
                name_or_path,
                correct_pad_token=correct_pad_token,
                correct_gemma2=correct_gemma2,
                **kwargs,
            )
        except (KeyError, ValueError, OSError) as e:
            tok = None
            with contextlib.suppress(Exception):
                tok = _from_processor(name_or_path, **kwargs)
            if tok is None:
                warnings.warn(
                    f"Could not load a tokenizer for {name_or_path!r} "
                    f"({type(e).__name__}: {e}); returning None "
                    "(expected for processor-only models such as GR00T).",
                    stacklevel=1,
                )
                return None
            if correct_pad_token:
                tok_mod.set_pad_token_id(tok)
            return tok

    for mod_name in ("verl.utils.tokenizer", "verl.utils", "verl.workers.config.model"):
        with contextlib.suppress(ImportError):
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "hf_tokenizer"):
                setattr(mod, "hf_tokenizer", hf_tokenizer)

    tok_mod._verl_vla_gr00t_tokenizer_patch = True
    logger.info("Patched verl hf_tokenizer for processor-only (GR00T) checkpoints")


def apply_gr00t_compat_patches() -> None:
    """Apply all GR00T load-time shims (idempotent). Call before model registration."""
    disable_cudnn_sdpa()
    patch_eagle_compat()
    patch_hf_tokenizer_for_processor_only_models()
    patch_verl_monkey_patch_for_custom_vla_configs()
