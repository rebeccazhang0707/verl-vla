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

from __future__ import annotations

import glob
import os
from pathlib import Path

import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers import AutoTokenizer, Gemma3ForCausalLM, GemmaForCausalLM, PreTrainedModel, SiglipVisionModel
from transformers.models.auto import CONFIG_MAPPING
from verl import DataProto

from verl_vla.models.base import SupportSFTTraining
from verl_vla.models.pi0_torch.model.paligemma_with_expert import RoPEEmbedding
from verl_vla.utils.dtype import precision_to_torch_dtype
from verl_vla.utils.models.value_head import DistributionalValueHead

from .configuration_recap_value_critic import ReCapValueCriticConfig, get_gemma_expert_config


def _make_joint_attention_mask(
    prefix_pad_masks: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_ar_masks: torch.Tensor,
) -> torch.Tensor:
    prefix_att_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.long)
    att_masks = torch.cat([prefix_att_masks, suffix_ar_masks.to(dtype=torch.long)], dim=1)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected batched image tensor with 4 dims, got shape={tuple(image.shape)}.")
    if image.shape[1] == 3:
        return image
    if image.shape[-1] == 3:
        return image.permute(0, 3, 1, 2)
    raise ValueError(f"Expected image tensor in BCHW or BHWC format, got shape={tuple(image.shape)}.")


def _normalize_image(image: torch.Tensor, image_size: int) -> torch.Tensor:
    image = _to_bchw(image).float()
    if image.max() > 1.0:
        image = image / 255.0
    image = image * 2.0 - 1.0
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return image


class ReCapValueDualStreamLayer(nn.Module):
    def __init__(
        self, prefix_layer: nn.Module, suffix_layer: nn.Module, head_dim: int, rope_theta: int, rope_max_seq_len: int
    ):
        super().__init__()
        self.prefix_layer = prefix_layer
        self.suffix_layer = suffix_layer
        self.rope_embedding = RoPEEmbedding(dim=head_dim, max_wavelength=rope_theta, max_seq_len=rope_max_seq_len)

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, num_groups: int) -> torch.Tensor:
        if num_groups == 1:
            return hidden_states
        return torch.repeat_interleave(hidden_states, num_groups, dim=2)

    @staticmethod
    def _project_qkv(layer, hidden_states: torch.Tensor, num_heads: int, num_kv_heads: int, head_dim: int):
        input_shape = hidden_states.shape[:-1]
        query_states = layer.self_attn.q_proj(hidden_states).view(*input_shape, num_heads, head_dim)
        key_states = layer.self_attn.k_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim)
        value_states = layer.self_attn.v_proj(hidden_states).view(*input_shape, num_kv_heads, head_dim)
        return query_states, key_states, value_states

    def forward(
        self,
        prefix_hidden_states: torch.Tensor,
        suffix_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_layer = self.prefix_layer
        suffix_layer = self.suffix_layer
        prefix_residual = prefix_hidden_states
        suffix_residual = suffix_hidden_states
        prefix_normed = prefix_layer.input_layernorm(prefix_hidden_states)
        suffix_normed = suffix_layer.input_layernorm(suffix_hidden_states)

        prefix_num_heads = int(prefix_layer.self_attn.config.num_attention_heads)
        prefix_num_kv_heads = int(prefix_layer.self_attn.config.num_key_value_heads)
        prefix_head_dim = int(getattr(prefix_layer.self_attn.config, "head_dim", prefix_layer.self_attn.head_dim))
        suffix_num_heads = int(suffix_layer.self_attn.config.num_attention_heads)
        suffix_num_kv_heads = int(suffix_layer.self_attn.config.num_key_value_heads)
        suffix_head_dim = int(getattr(suffix_layer.self_attn.config, "head_dim", suffix_layer.self_attn.head_dim))
        if (prefix_num_heads, prefix_num_kv_heads, prefix_head_dim) != (
            suffix_num_heads,
            suffix_num_kv_heads,
            suffix_head_dim,
        ):
            raise ValueError(
                "Prefix Gemma3 and suffix value expert attention shapes must match for single-stage value forward: "
                f"prefix={(prefix_num_heads, prefix_num_kv_heads, prefix_head_dim)}, "
                f"suffix={(suffix_num_heads, suffix_num_kv_heads, suffix_head_dim)}."
            )

        prefix_q, prefix_k, prefix_v = self._project_qkv(
            prefix_layer, prefix_normed, prefix_num_heads, prefix_num_kv_heads, prefix_head_dim
        )
        suffix_q, suffix_k, suffix_v = self._project_qkv(
            suffix_layer, suffix_normed, suffix_num_heads, suffix_num_kv_heads, suffix_head_dim
        )

        query_states = torch.cat([prefix_q, suffix_q], dim=1)
        key_states = torch.cat([prefix_k, suffix_k], dim=1)
        value_states = torch.cat([prefix_v, suffix_v], dim=1)
        query_states = self.rope_embedding(query_states, position_ids)
        key_states = self.rope_embedding(key_states, position_ids)
        key_states = self._repeat_kv(key_states, prefix_num_heads // prefix_num_kv_heads)
        value_states = self._repeat_kv(value_states, prefix_num_heads // prefix_num_kv_heads)

        batch_size = query_states.shape[0]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask[:, None, :, :],
            is_causal=False,
        )
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, -1, prefix_num_heads * prefix_head_dim)

        prefix_len = prefix_hidden_states.shape[1]
        prefix_attn = attn_output[:, :prefix_len].to(dtype=prefix_layer.self_attn.o_proj.weight.dtype)
        suffix_attn = attn_output[:, prefix_len:].to(dtype=suffix_layer.self_attn.o_proj.weight.dtype)
        prefix_hidden_states = prefix_residual + prefix_layer.self_attn.o_proj(prefix_attn)
        suffix_hidden_states = suffix_residual + suffix_layer.self_attn.o_proj(suffix_attn)

        prefix_residual = prefix_hidden_states
        suffix_residual = suffix_hidden_states
        prefix_hidden_states = prefix_residual + prefix_layer.mlp(
            prefix_layer.post_attention_layernorm(prefix_hidden_states)
        )
        suffix_hidden_states = suffix_residual + suffix_layer.mlp(
            suffix_layer.post_attention_layernorm(suffix_hidden_states)
        )
        return prefix_hidden_states, suffix_hidden_states


class ReCapValueExpert(nn.Module):
    def __init__(self, config: ReCapValueCriticConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel.from_pretrained(config.siglip_path)
        self.gemma3 = Gemma3ForCausalLM.from_pretrained(config.gemma3_path)

        vision_hidden_size = self.vision_tower.config.hidden_size
        gemma_hidden_size = self.gemma3.config.hidden_size
        self.multi_modal_proj = nn.Linear(vision_hidden_size, gemma_hidden_size)
        nn.init.normal_(self.multi_modal_proj.weight, std=0.02)
        nn.init.zeros_(self.multi_modal_proj.bias)

        expert_config = get_gemma_expert_config(config.critic_expert_variant)
        gemma3_head_dim = getattr(self.gemma3.config, "head_dim", expert_config.head_dim)
        if expert_config.head_dim != gemma3_head_dim:
            raise ValueError(
                f"Value expert head_dim={expert_config.head_dim} must match Gemma3 head_dim={gemma3_head_dim}."
            )
        expert_hf_config = CONFIG_MAPPING["gemma"](
            head_dim=expert_config.head_dim,
            hidden_size=expert_config.width,
            intermediate_size=expert_config.mlp_dim,
            num_attention_heads=expert_config.num_heads,
            num_hidden_layers=expert_config.depth,
            num_key_value_heads=expert_config.num_kv_heads,
            vocab_size=int(getattr(self.gemma3.config, "vocab_size", 257152)),
            hidden_activation="gelu_pytorch_tanh",
        )
        self.value_expert = GemmaForCausalLM(config=expert_hf_config)
        self.value_expert.model.embed_tokens = None
        self.dual_layers = nn.ModuleList(
            [
                ReCapValueDualStreamLayer(
                    prefix_layer=prefix_layer,
                    suffix_layer=suffix_layer,
                    head_dim=expert_config.head_dim,
                    rope_theta=int(getattr(self.gemma3.config, "rope_theta", 10_000)),
                    rope_max_seq_len=int(getattr(self.gemma3.config, "max_position_embeddings", 8192)),
                )
                for prefix_layer, suffix_layer in zip(
                    self.gemma3.model.layers,
                    self.value_expert.model.layers,
                    strict=True,
                )
            ]
        )
        self.gemma3.model.layers = nn.ModuleList()
        self.value_expert.model.layers = nn.ModuleList()

        self._apply_precision(config.precision)
        self._set_requires_grad()

    def _apply_precision(self, precision: str) -> None:
        dtype = precision_to_torch_dtype(precision)
        self.vision_tower.to(dtype=dtype)
        self.gemma3.to(dtype=dtype)
        self.value_expert.to(dtype=dtype)

    def _set_requires_grad(self) -> None:
        if self.config.freeze_vision_encoder:
            self.vision_tower.requires_grad_(False)
            self.vision_tower.eval()
        if self.config.freeze_vlm:
            self.gemma3.requires_grad_(False)
            for dual_layer in self.dual_layers:
                dual_layer.prefix_layer.requires_grad_(False)
            self.gemma3.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.vision_tower.eval()
        if self.config.freeze_vlm:
            self.gemma3.eval()
            for dual_layer in self.dual_layers:
                dual_layer.prefix_layer.eval()
        return self

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.contiguous()
        if self.config.freeze_vision_encoder:
            with torch.no_grad():
                output = self.vision_tower(pixel_values=image).last_hidden_state
        else:
            output = self.vision_tower(pixel_values=image).last_hidden_state
        return self.multi_modal_proj(output)

    def embed_language_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_vlm:
            with torch.no_grad():
                return self.gemma3.model.embed_tokens(token_ids)
        return self.gemma3.model.embed_tokens(token_ids)

    def forward(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_ar_masks: torch.Tensor,
    ) -> torch.Tensor:
        prefix_hidden_states = prefix_embs
        suffix_hidden_states = suffix_embs
        if self.config.freeze_vlm:
            prefix_hidden_states = prefix_hidden_states.detach()

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        attention_mask = _make_joint_attention_mask(prefix_pad_masks, suffix_pad_masks, suffix_ar_masks)
        for dual_layer in self.dual_layers:
            prefix_hidden_states, suffix_hidden_states = dual_layer(
                prefix_hidden_states=prefix_hidden_states,
                suffix_hidden_states=suffix_hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        return self.value_expert.model.norm(suffix_hidden_states)


class ReCapValueCriticTrainableModel(PreTrainedModel, SupportSFTTraining):
    config_class = ReCapValueCriticConfig
    base_model_prefix = "recap_value_critic"

    def __init__(self, config: ReCapValueCriticConfig):
        config.refresh_derived_fields()
        PreTrainedModel.__init__(self, config)
        SupportSFTTraining.__init__(self, config)
        self.value_expert = ReCapValueExpert(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.cls_embedding = nn.Embedding(1, config.value_head_input_dim)
        nn.init.normal_(self.cls_embedding.weight, std=0.02)
        self.state_proj = nn.Linear(config.max_state_dim, config.value_head_input_dim)
        nn.init.normal_(self.state_proj.weight, std=0.02)
        nn.init.zeros_(self.state_proj.bias)
        self.value_head = self._build_value_head(config)

    @staticmethod
    def _build_value_head(config: ReCapValueCriticConfig) -> DistributionalValueHead:
        hidden_dim = config.value_head_hidden_dim
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
        return DistributionalValueHead(
            input_dim=int(config.value_head_input_dim),
            hidden_dim=hidden_dim,
            num_bins=int(config.value_head_num_bins),
            v_min=float(config.value_head_v_min),
            v_max=float(config.value_head_v_max),
            dropout=float(config.value_head_dropout),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        del model_args
        config = kwargs.pop("config", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        kwargs.pop("trust_remote_code", None)
        config_override_keys = {
            "siglip_path",
            "gemma3_path",
            "tokenizer_path",
            "critic_expert_variant",
            "image_size",
            "max_token_len",
            "image_keys",
            "include_state_in_prompt",
            "max_state_dim",
            "freeze_vision_encoder",
            "freeze_vlm",
            "stop_gradient_to_vlm",
            "precision",
            "sft_type",
            "value_head_hidden_dim",
            "value_head_num_bins",
            "value_head_v_min",
            "value_head_v_max",
            "value_head_dropout",
        }
        config_overrides = {key: kwargs.pop(key) for key in list(kwargs) if key in config_override_keys}
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs, **config_overrides)
        elif config_overrides:
            config_dict = config.to_dict()
            config_dict.update(config_overrides)
            config = cls.config_class(**config_dict)
        if torch_dtype is not None and getattr(config, "precision", None) in {None, "auto"}:
            config.precision = str(torch_dtype).replace("torch.", "")
        model = cls(config)
        state_dict = cls._load_optional_state_dict(pretrained_model_name_or_path)
        if state_dict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if len(unexpected) > 0:
                print(f"[ReCapValueCritic] ignored unexpected checkpoint keys: {len(unexpected)}")
            if len(missing) > 0:
                print(f"[ReCapValueCritic] initialized missing checkpoint keys: {len(missing)}")
        return model

    @staticmethod
    def _load_optional_state_dict(path_or_repo: str | os.PathLike) -> dict[str, torch.Tensor]:
        path = Path(path_or_repo)
        if not path.exists() or not path.is_dir():
            return {}
        state_dict: dict[str, torch.Tensor] = {}
        safetensor_paths = sorted(glob.glob(str(path / "*.safetensors")))
        if safetensor_paths:
            for weight_path in safetensor_paths:
                state_dict.update(safetensors.torch.load_file(weight_path, device="cpu"))
            return state_dict
        for name in ["pytorch_model.bin", "model.pt", "full_weights.pt"]:
            weight_path = path / name
            if weight_path.exists():
                return torch.load(weight_path, map_location="cpu", weights_only=False)
        return {}

    def sft_init(self):
        super().sft_init()

    def _extract_images_and_masks(self, obs: DataProto) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        image_key, wrist_image_key = self.config.image_keys
        image = _normalize_image(obs.batch[image_key], self.config.image_size)
        wrist_image = _normalize_image(obs.batch[wrist_image_key], self.config.image_size)
        batch_size = image.shape[0]
        device = image.device
        empty = torch.zeros_like(image)
        images = [image, wrist_image, empty]
        masks = [
            torch.ones(batch_size, dtype=torch.bool, device=device),
            torch.ones(batch_size, dtype=torch.bool, device=device),
            torch.zeros(batch_size, dtype=torch.bool, device=device),
        ]
        return images, masks

    def _build_prompts(self, obs: DataProto) -> list[str]:
        tasks = [str(task) for task in obs.non_tensor_batch["task"]]
        return [f"Task: {task}." for task in tasks]

    def _embed_state(self, obs: DataProto, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        state = obs.batch["observation.state"].to(device=device, dtype=self.state_proj.weight.dtype)
        if state.ndim > 2:
            state = state.reshape(state.shape[0], -1)
        if state.shape[-1] > self.config.max_state_dim:
            state = state[..., : self.config.max_state_dim]
        elif state.shape[-1] < self.config.max_state_dim:
            state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        return self.state_proj(state).to(dtype=dtype)

    def _embed_prefix(self, obs: DataProto) -> tuple[torch.Tensor, torch.Tensor]:
        images, image_masks = self._extract_images_and_masks(obs)
        param = next(self.value_expert.parameters())
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        for image, image_mask in zip(images, image_masks, strict=True):
            image = image.to(device=param.device, dtype=param.dtype)
            image_emb = self.value_expert.embed_image(image)
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(image_emb.shape[0], image_emb.shape[1]))

        tokenized = self.tokenizer(
            self._build_prompts(obs),
            padding=True,
            truncation=True,
            max_length=self.config.max_token_len,
            return_tensors="pt",
        )
        lang_tokens = tokenized["input_ids"].to(device=embs[0].device)
        lang_masks = tokenized["attention_mask"].to(device=embs[0].device, dtype=torch.bool)
        lang_emb = self.value_expert.embed_language_tokens(lang_tokens).to(dtype=embs[0].dtype)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1)

    def _embed_suffix(
        self, obs: DataProto, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        cls_emb = self.cls_embedding(token_ids).to(dtype=next(self.value_expert.value_expert.parameters()).dtype)
        state_emb = self._embed_state(obs, device=device, dtype=cls_emb.dtype)
        cls_emb = cls_emb + state_emb[:, None, :]
        pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        ar_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        return cls_emb, pad_mask, ar_mask

    def value_model_forward_features(self, obs: DataProto, tokenizer: torch.nn.Module | None = None) -> torch.Tensor:
        del tokenizer
        prefix_embs, prefix_pad_masks = self._embed_prefix(obs)
        batch_size = prefix_embs.shape[0]
        device = prefix_embs.device
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self._embed_suffix(obs, batch_size, device)

        model_dtype = next(self.value_expert.gemma3.parameters()).dtype
        prefix_embs = prefix_embs.to(dtype=model_dtype)
        suffix_embs = suffix_embs.to(dtype=next(self.value_expert.value_expert.parameters()).dtype)
        suffix_out = self.value_expert(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            suffix_embs=suffix_embs,
            suffix_pad_masks=suffix_pad_masks,
            suffix_ar_masks=suffix_ar_masks,
        )
        return suffix_out[:, -1, :]

    def forward(self, obs: DataProto, tokenizer: torch.nn.Module | None = None) -> torch.Tensor:
        features = self.value_model_forward_features(obs=obs, tokenizer=tokenizer)
        values, _, _ = self.value_head(features)
        return values

    def sft_loss(
        self,
        obs: DataProto,
        tokenizer: torch.nn.Module | None,
        actions: dict[str, torch.Tensor],
        valids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        target_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Override SupportSFTTraining.sft_loss for ReCap value-model training."""

        del actions, action_mask
        if target_values is None:
            raise ValueError("ReCap value-model SFT requires target_values.")

        features = self.value_model_forward_features(obs=obs, tokenizer=tokenizer)
        _, logits, _ = self.value_head(features)
        per_sample_loss, metrics = self.value_head.loss(
            logits, target_values.to(device=logits.device).float(), reduction="none"
        )
        self.sft_metrics = {f"value/{name}": value for name, value in metrics.items()}

        valid_mask = valids.float().view(-1).to(device=per_sample_loss.device)
        return (per_sample_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
