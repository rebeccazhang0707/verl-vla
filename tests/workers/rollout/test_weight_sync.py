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

import asyncio

import torch
import torch.nn as nn

from verl_vla.workers.rollout.hf_rollout import HFRollout


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 4)


def test_update_weights_loads_matching_tensors_and_skips_unmatched_tensors() -> None:
    source_model = SimpleModel()
    target_model = SimpleModel()
    for parameter in target_model.parameters():
        nn.init.zeros_(parameter)

    async def iter_weights():
        for name, parameter in source_model.state_dict().items():
            yield f"_fsdp_wrapped_module.{name}", parameter.clone()
        yield "_fsdp_wrapped_module.critic_backend.target_network.weight", torch.randn(8, 8)

    rollout = object.__new__(HFRollout)
    rollout.module = target_model
    rollout.engine = None

    asyncio.run(rollout.update_weights(iter_weights()))

    target_state = target_model.state_dict()
    for name, parameter in source_model.state_dict().items():
        torch.testing.assert_close(target_state[name], parameter)
