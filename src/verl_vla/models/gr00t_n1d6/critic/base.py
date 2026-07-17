# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch


class CriticBackend(ABC):
    # Arena GR00T critics are single-task; set True only if a multitask backend
    # (pi05-style multi_cross_attn) is added later.
    uses_task_ids = False

    @abstractmethod
    def init(self, model) -> None:
        pass

    @abstractmethod
    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_critic_parameters(self, model) -> list[torch.nn.Parameter]:
        pass

    @abstractmethod
    def update_target_network(self, model, tau: float) -> None:
        pass

    def resolve_action(self, a: dict[str, Any]) -> torch.Tensor:
        """Critic scores the NORMALISED ``full_action`` when present."""
        return a.get("full_action", a["action"])
