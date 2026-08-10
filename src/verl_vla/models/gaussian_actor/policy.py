# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
from verl import DataProto

from ..base import ModelOutput


class GaussianActorOutput(ModelOutput):
    def __init__(self, action: torch.Tensor, log_prob: torch.Tensor | None) -> None:
        self.action = action
        self.log_prob = log_prob

    def to_data_proto(self) -> DataProto:
        tensors = {"action": self.action.float()}
        if self.log_prob is not None:
            tensors["log_prob"] = self.log_prob.float()
        return DataProto.from_dict(tensors=tensors)
