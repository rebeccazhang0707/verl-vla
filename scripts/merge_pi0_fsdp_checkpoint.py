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

"""Merge a verl FSDP PI0 checkpoint into loadable Hugging Face/diffusers weights.

This script handles the PI0 checkpoint layout used by VLA training:

- only `model.*` entries are exported as model weights;
- training-only entries such as `flow_sde_step` are skipped;
- DTensor metadata is used to recover the real FSDP mesh;
- the output directory contains both HF wrapper shards (`model-*.safetensors`)
  and diffusers shards (`diffusion_pytorch_model-*.safetensors`).

Example:
    python scripts/merge_pi0_fsdp_checkpoint.py \
        --local-dir /file_system/liujincheng/output/pi05_lerobot_sft/global_step_128000/actor \
        --target-dir /file_system/liujincheng/models/pi05_lerobot_sft_global_step_128000_hf \
        --verify
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils.transformers_compat import get_auto_model_for_vision2seq

from verl_vla.models import register_vla_models
from verl_vla.models.pi0_torch import PI0TorchConfig
from verl_vla.models.pi0_torch.model.modeling_pi0 import PI0Model


class PI0FSDPModelMerger(FSDPModelMerger):
    """FSDP merger variant for PI0 checkpoints with training-only state entries."""

    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        for weight in state_dict.values():
            if isinstance(weight, DTensor):
                return weight.device_mesh.mesh, weight.device_mesh.mesh_dim_names
        return np.array([world_size], dtype=np.int64), ("fsdp",)

    def _load_and_merge_state_dicts(
        self,
        world_size: int,
        total_shards: int,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...],
    ) -> dict[str, torch.Tensor]:
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, state_dicts: list[dict | None]) -> dict:
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dicts[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        first_state_dict = model_state_dict_lst[0]
        if first_state_dict is None:
            raise RuntimeError("Failed to load rank-0 checkpoint shard.")

        model_keys = [key for key in first_state_dict if key.startswith("model.")]
        skipped = len(first_state_dict) - len(model_keys)
        print(f"Merging {len(model_keys)} model weights; skipping {skipped} training-only keys")

        merged_state_dict: dict[str, list[torch.Tensor] | torch.Tensor] = {}
        param_placements: dict[str, tuple] = {}

        for key in model_keys:
            merged_state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                if model_state_shard is None:
                    raise RuntimeError(f"Missing checkpoint shard while merging {key}.")

                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    merged_state_dict[key].append(tensor._local_tensor.bfloat16())
                    placements = tuple(tensor.placements)
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]
                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    value = tensor.bfloat16() if tensor.is_floating_point() else tensor
                    merged_state_dict[key].append(value)

        del model_state_dict_lst

        for key in sorted(merged_state_dict):
            tensors = merged_state_dict[key]
            if not isinstance(tensors, list):
                continue

            if key in param_placements:
                placements = param_placements[key]
                if len(mesh_shape) != 1:
                    raise NotImplementedError("FSDP + TP is not supported yet")
                assert len(placements) == 1
                merged_state_dict[key] = self._merge_by_placement(tensors, placements[0])
            else:
                merged_state_dict[key] = tensors[0] if tensors[0].dim() == 0 else torch.cat(tensors, dim=0)

        return merged_state_dict

    def get_transformers_auto_model_class(self):
        if isinstance(self.model_config, PI0TorchConfig):
            return get_auto_model_for_vision2seq()
        return super().get_transformers_auto_model_class()


def find_latest_actor_dir(checkpoint_root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("global_step_*/actor"):
        match = re.fullmatch(r"global_step_(\d+)", path.parent.name)
        if match and (path / "fsdp_config.json").exists():
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(f"No global_step_*/actor FSDP checkpoints found under {checkpoint_root}")

    return max(candidates, key=lambda item: item[0])[1]


def write_diffusers_shards(target_dir: Path) -> None:
    hf_index_path = target_dir / "model.safetensors.index.json"
    if hf_index_path.exists():
        with hf_index_path.open(encoding="utf-8") as f:
            hf_index = json.load(f)
        hf_files = sorted(set(hf_index["weight_map"].values()))
    else:
        hf_files = ["model.safetensors"]

    new_weight_map: dict[str, str] = {}
    total_size = 0
    width = max(5, len(str(len(hf_files))))

    for idx, src_name in enumerate(hf_files, start=1):
        src_path = target_dir / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"Cannot find HF shard {src_path}")

        if len(hf_files) == 1:
            dst_name = "diffusion_pytorch_model.safetensors"
        else:
            dst_name = f"diffusion_pytorch_model-{idx:0{width}d}-of-{len(hf_files):0{width}d}.safetensors"

        tensors = load_file(src_path)
        stripped = {}
        for key, value in tensors.items():
            if not key.startswith("model."):
                continue
            new_key = key.removeprefix("model.")
            stripped[new_key] = value
            new_weight_map[new_key] = dst_name
            total_size += value.numel() * value.element_size()

        print(f"{src_name}: {len(stripped)} tensors -> {dst_name}")
        save_file(stripped, target_dir / dst_name, metadata={"format": "pt"})

    if len(hf_files) > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(new_weight_map.items())),
        }
        with (target_dir / "diffusion_pytorch_model.safetensors.index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")


def merge_checkpoint(local_dir: Path, target_dir: Path) -> None:
    register_vla_models()

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(target_dir),
        trust_remote_code=True,
        local_dir=str(local_dir),
        hf_model_config_path=str(local_dir / "huggingface"),
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    merger = PI0FSDPModelMerger(config)
    merger.merge_and_save()
    write_diffusers_shards(target_dir)


def verify_load(target_dir: Path) -> None:
    model = PI0Model.from_pretrained(str(target_dir), low_cpu_mem_usage=True)
    num_params = sum(param.numel() for param in model.parameters())
    print(f"Verified PI0Model.from_pretrained: {num_params} parameters")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-dir", type=Path, help="Path to a checkpoint actor directory.")
    source.add_argument("--checkpoint-root", type=Path, help="Root containing global_step_*/actor checkpoints.")
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory to write merged weights.")
    parser.add_argument("--verify", action="store_true", help="Load the exported diffusers model after merging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir if args.local_dir is not None else find_latest_actor_dir(args.checkpoint_root)
    local_dir = local_dir.resolve()
    target_dir = args.target_dir.resolve()

    print(f"Merging checkpoint: {local_dir}")
    print(f"Writing merged model: {target_dir}")
    merge_checkpoint(local_dir, target_dir)
    if args.verify:
        verify_load(target_dir)


if __name__ == "__main__":
    main()
