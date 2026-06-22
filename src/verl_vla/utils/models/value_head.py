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

import torch
import torch.nn.functional as F  # noqa: N812


class DistributionalValueHead(torch.nn.Module):
    """Distributional value head for normalized ReCap returns."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        num_bins: int = 201,
        v_min: float = -1.0,
        v_max: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be at least 2, got {num_bins}.")
        if v_min >= v_max:
            raise ValueError(f"v_min must be smaller than v_max, got {v_min} >= {v_max}.")

        if hidden_dim is None:
            hidden_dim = input_dim
        if hidden_dim <= 0:
            self.value_proj = torch.nn.Linear(input_dim, num_bins)
        else:
            self.value_proj = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity(),
                torch.nn.Linear(hidden_dim, num_bins),
            )
        self.register_buffer("atoms", torch.linspace(v_min, v_max, num_bins), persistent=False)
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_bins - 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value_dtype = self.atoms.dtype
        for module in self.value_proj.modules():
            if isinstance(module, torch.nn.Linear):
                value_dtype = module.weight.dtype
                break
        features = features.to(dtype=value_dtype)
        logits = self.value_proj(features)
        probs = F.softmax(logits, dim=-1)
        values = (probs * self.atoms.to(device=logits.device, dtype=probs.dtype)).sum(dim=-1)
        return values, logits, probs

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}.")
        targets = targets.float().view(-1).clamp(self.v_min, self.v_max)
        if targets.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Value target batch size does not match logits: targets={targets.shape[0]}, logits={logits.shape[0]}."
            )

        b = (targets - self.v_min) / self.delta_z
        lower = b.floor().long().clamp(0, self.num_bins - 1)
        upper = b.ceil().long().clamp(0, self.num_bins - 1)
        upper_weight = (b - lower.float()).clamp(0.0, 1.0)
        lower_weight = 1.0 - upper_weight

        target_probs = torch.zeros(
            targets.shape[0],
            self.num_bins,
            device=targets.device,
            dtype=logits.dtype,
        )
        batch_idx = torch.arange(targets.shape[0], device=targets.device)
        target_probs[batch_idx, lower] += lower_weight.to(dtype=logits.dtype)
        target_probs[batch_idx, upper] += upper_weight.to(dtype=logits.dtype)

        per_sample_loss = -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        pred_bins = logits.argmax(dim=-1)
        best_target_bins = torch.where(lower_weight >= upper_weight, lower, upper)
        pred_values = (F.softmax(logits, dim=-1) * self.atoms.to(logits.device, logits.dtype)).sum(dim=-1)
        metrics = {
            "loss": per_sample_loss.mean().detach(),
            "mae": (pred_values.float() - targets).abs().mean().detach(),
            "pred_mean": pred_values.float().mean().detach(),
            "target_mean": targets.float().mean().detach(),
            "pred_std": pred_values.float().std(unbiased=False).detach(),
            "target_std": targets.float().std(unbiased=False).detach(),
            "bin_acc": (pred_bins == best_target_bins).float().mean().detach(),
            "neighbor_acc": ((pred_bins == lower) | (pred_bins == upper)).float().mean().detach(),
        }
        loss = per_sample_loss if reduction == "none" else per_sample_loss.mean()
        return loss, metrics
