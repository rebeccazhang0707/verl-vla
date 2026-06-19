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

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def load_lerobot_feature_names(dataset_root: str | Path) -> set[str]:
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        return set()
    with open(info_path) as f:
        info = json.load(f)
    return set(info.get("features", {}).keys())


def collect_lerobot_columns(data_files: list[str | Path]) -> set[str]:
    columns: set[str] = set()
    for parquet_path in data_files:
        columns.update(pq.ParquetFile(parquet_path).schema_arrow.names)
    return columns


def update_lerobot_feature_metadata(dataset_root: str | Path, feature_infos: dict[str, dict[str, Any]]) -> None:
    try:
        from lerobot.datasets.utils import load_info, write_info
    except ImportError:
        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"LeRobot info.json not found: {info_path}") from None
        with open(info_path) as f:
            info = json.load(f)
        features = info.setdefault("features", {})
        for feature_name, feature_info in feature_infos.items():
            features[feature_name] = feature_info
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        return

    info = load_info(Path(dataset_root))
    for feature_name, feature_info in feature_infos.items():
        info["features"][feature_name] = {
            "dtype": feature_info["dtype"],
            "shape": tuple(feature_info["shape"]),
            "names": feature_info.get("names"),
        }
    write_info(info, Path(dataset_root))
