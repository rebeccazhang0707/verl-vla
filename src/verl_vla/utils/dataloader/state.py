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

import logging
from pathlib import Path
from typing import Any

import torch

__all__ = ["load_dataloader_state"]

logger = logging.getLogger(__name__)


def load_dataloader_state(dataloader: Any, path: str | Path | None, filename: str = "data.pt") -> bool:
    if path is None:
        logger.warning("No dataloader state path provided; starting dataloader from scratch.")
        return False

    state_path = Path(path)
    if state_path.is_dir() or state_path.suffix == "":
        state_path = state_path / filename

    if not state_path.exists():
        logger.warning("No dataloader state found at %s; starting dataloader from scratch.", state_path)
        return False

    dataloader.load_state_dict(torch.load(state_path, weights_only=False))
    return True
