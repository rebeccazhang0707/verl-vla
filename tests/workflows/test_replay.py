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

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from verl_vla.workflows.replay import load_lerobot_episode


def test_load_lerobot_episode_reads_ordered_actions_and_reset_state(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "fps": 20}),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table({"task_index": [4], "task": ["pick up the bowl"]}),
        root / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [1, 0, 1],
                "frame_index": [1, 0, 0],
                "task_index": [4, 4, 4],
                "action": [[3.0, 4.0], [9.0, 9.0], [1.0, 2.0]],
                "observation.state": [[0.3], [0.9], [0.1]],
                "info.reset_state_id": [[17], [99], [17]],
            }
        ),
        data_dir / "file-000.parquet",
    )

    episode = load_lerobot_episode(root, 1, extra_columns=("info.reset_state_id",))

    assert episode.episode_index == 1
    assert episode.fps == 20
    assert episode.task == "pick up the bowl"
    assert episode.actions.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    np.testing.assert_allclose(episode.states, [[0.1], [0.3]])
    assert episode.extra["info.reset_state_id"].tolist() == [[17], [17]]
