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

import torch


def precision_to_torch_dtype(precision: str) -> torch.dtype:
    if precision in {"bf16", "bfloat16", "bf16-mixed"}:
        return torch.bfloat16
    if precision in {"fp16", "float16", "16", "16-mixed"}:
        return torch.float16
    return torch.float32
