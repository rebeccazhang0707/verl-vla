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

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from verl_vla.utils.lerobot import (
    collect_lerobot_columns,
    list_lerobot_data_files,
    load_lerobot_feature_names,
    update_lerobot_feature_metadata,
    write_parquet_columns,
)

DatasetInfo = dict[str, str | Path]
CollectedDatasets = dict[str, DatasetInfo]

RECAP_RETURN_FIELD = "recap.return"
RECAP_VALUE_FIELD = "recap.value"
RECAP_ADVANTAGE_FIELD = "recap.advantage"
RECAP_INDICATOR_FIELD = "recap.indicator"
RECAP_FIELDS = {
    RECAP_RETURN_FIELD: {"dtype": "float32", "shape": [1], "names": None},
    RECAP_VALUE_FIELD: {"dtype": "float32", "shape": [1], "names": None},
    RECAP_ADVANTAGE_FIELD: {"dtype": "float32", "shape": [1], "names": None},
    RECAP_INDICATOR_FIELD: {"dtype": "int64", "shape": [1], "names": None},
}


def _get_field_names(return_cfg) -> dict[str, str]:
    fields_cfg = return_cfg.fields
    return {
        "return": str(fields_cfg["return"]),
        "value": str(fields_cfg.value),
        "advantage": str(fields_cfg.advantage),
        "indicator": str(fields_cfg.indicator),
    }


def _get_recap_fields(field_names: dict[str, str]) -> dict[str, dict[str, object]]:
    return {
        field_names["return"]: {"dtype": "float32", "shape": [1], "names": None},
        field_names["value"]: {"dtype": "float32", "shape": [1], "names": None},
        field_names["advantage"]: {"dtype": "float32", "shape": [1], "names": None},
        field_names["indicator"]: {"dtype": "int64", "shape": [1], "names": None},
    }


def _ensure_recap_fields_for_dataset(
    dataset: DatasetInfo,
    *,
    field_names: dict[str, str],
    c_fail_coef: float = 1.0,
    clip_min: float = -1.0,
    clip_max: float = 0.0,
) -> None:
    """Ensure a LeRobot dataset has all RECAP columns and compute normalized returns."""
    dataset_root = Path(dataset["root"])
    data_files = list_lerobot_data_files(dataset_root)

    recap_fields = _get_recap_fields(field_names)
    existing_fields = load_lerobot_feature_names(dataset_root)
    missing_fields = [field for field in recap_fields if field not in existing_fields]
    existing_columns = collect_lerobot_columns(data_files)
    missing_columns = [field for field in recap_fields if field not in existing_columns]
    return_lookup = _compute_return_lookup(
        data_files,
        c_fail_coef=c_fail_coef,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    for parquet_path in data_files:
        _ensure_recap_columns_for_file(
            parquet_path,
            missing_columns,
            return_lookup,
            field_names=field_names,
            recap_fields=recap_fields,
        )
    if missing_fields:
        update_lerobot_feature_metadata(dataset_root, {field: recap_fields[field] for field in missing_fields})


def ensure_recap_fields(config, collected_datasets: CollectedDatasets) -> CollectedDatasets:
    return_cfg = config.recap.compute_return
    field_names = _get_field_names(return_cfg)
    c_fail_coef = float(return_cfg.c_fail_coef)
    clip_min = float(return_cfg.clip_min)
    clip_max = float(return_cfg.clip_max)
    for dataset in collected_datasets.values():
        _ensure_recap_fields_for_dataset(
            dataset,
            field_names=field_names,
            c_fail_coef=c_fail_coef,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    return collected_datasets


def _compute_return_lookup(
    data_files: list[Path],
    *,
    c_fail_coef: float,
    clip_min: float,
    clip_max: float,
) -> dict[int, np.float32]:
    if c_fail_coef < 0:
        raise ValueError("recap.compute_return.c_fail_coef must be non-negative.")
    if clip_min >= clip_max:
        raise ValueError("recap.compute_return.clip_min must be smaller than recap.compute_return.clip_max.")

    records: list[dict[str, object]] = []
    for parquet_path in data_files:
        table = pq.read_table(
            parquet_path,
            columns=[
                "index",
                "episode_index",
                "frame_index",
                "task_index",
                "next.reward",
                "next.done",
                "next.truncated",
            ],
        )
        indices = table["index"].to_numpy().astype(np.int64, copy=False)
        episode_indices = table["episode_index"].to_numpy().astype(np.int64, copy=False)
        frame_indices = table["frame_index"].to_numpy().astype(np.int64, copy=False)
        task_keys = table["task_index"].to_numpy().astype(np.int64, copy=False).tolist()
        rewards = table["next.reward"].to_numpy().astype(np.float32, copy=False)
        dones = table["next.done"].to_numpy().astype(bool, copy=False)
        truncateds = table["next.truncated"].to_numpy().astype(bool, copy=False)

        records.extend(
            {
                "index": int(index),
                "episode_index": int(ep),
                "frame_index": int(frame),
                "reward": float(reward),
                "done": bool(done),
                "truncated": bool(truncated),
                "task_key": task_key,
            }
            for index, ep, frame, reward, done, truncated, task_key in zip(
                indices,
                episode_indices,
                frame_indices,
                rewards,
                dones,
                truncateds,
                task_keys,
                strict=True,
            )
        )

    records.sort(key=lambda item: (item["episode_index"], item["frame_index"], item["index"]))

    episodes: dict[int, list[dict[str, object]]] = {}
    for record in records:
        episodes.setdefault(int(record["episode_index"]), []).append(record)

    task_max_lengths: dict[object, int] = {}
    for episode_records in episodes.values():
        task_key = episode_records[0]["task_key"]
        task_max_lengths[task_key] = max(task_max_lengths.get(task_key, 0), len(episode_records))

    lookup: dict[int, np.float32] = {}
    for episode_records in episodes.values():
        task_key = episode_records[0]["task_key"]
        task_max = task_max_lengths[task_key]
        c_fail = float(task_max) * c_fail_coef
        denom = float(task_max) + c_fail
        if denom <= 0:
            raise ValueError(f"Invalid return normalization denominator for task={task_key}: {denom}.")

        final_record = episode_records[-1]
        success = (bool(final_record["done"]) and not bool(final_record["truncated"])) or float(
            final_record["reward"]
        ) > 0.0
        episode_length = len(episode_records)
        for offset, record in enumerate(episode_records):
            remaining_steps = episode_length - offset - 1
            target = -float(remaining_steps)
            if not success:
                target -= c_fail
            target = np.clip(target / denom, clip_min, clip_max)
            lookup[int(record["index"])] = np.float32(target)

    return lookup


def _ensure_recap_columns_for_file(
    parquet_path: Path,
    missing_columns: list[str],
    return_lookup: dict[int, np.float32],
    *,
    field_names: dict[str, str],
    recap_fields: dict[str, dict[str, object]],
) -> None:
    table = pq.read_table(parquet_path)
    indices = table["index"].to_numpy().astype(np.int64, copy=False)
    return_field = field_names["return"]
    fields_to_write = [return_field, *[field for field in missing_columns if field != return_field]]
    columns = {}
    for field in fields_to_write:
        if field == return_field:
            columns[field] = np.asarray([return_lookup[int(index)] for index in indices], dtype=np.float32)
        elif recap_fields[field]["dtype"] == "float32":
            columns[field] = np.full(len(indices), np.nan, dtype=np.float32)
        elif recap_fields[field]["dtype"] == "int64":
            columns[field] = np.zeros(len(indices), dtype=np.int64)
        else:
            raise ValueError(f"Unsupported RECAP field dtype: {field} -> {recap_fields[field]['dtype']}")

    write_parquet_columns(parquet_path=parquet_path, columns=columns)
