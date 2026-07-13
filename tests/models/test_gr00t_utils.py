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

"""Unit tests for ``models/gr00t/utils.py`` (gr00t-package-free helpers)."""

import json

import numpy as np
import pytest

gr00t_utils = pytest.importorskip("verl_vla.models.gr00t_n1d6.utils")


def test_gr1_group_dims():
    dims = gr00t_utils.GR1.state_group_dims
    assert sum(dims.values()) == 26
    assert list(dims.keys()) == ["left_arm", "right_arm", "left_hand", "right_hand"]


def test_split_flat_state_to_groups():
    B = 4
    dims = gr00t_utils.GR1.state_group_dims
    blocks = [np.full((B, d), float(i), dtype=np.float32) for i, d in enumerate(dims.values())]
    flat = np.concatenate(blocks, axis=-1)

    groups = gr00t_utils.split_flat_state_to_groups(flat, dims)

    assert list(groups.keys()) == list(dims.keys())
    for i, (key, d) in enumerate(dims.items()):
        assert groups[key].shape == (B, d)
        assert np.allclose(groups[key], float(i))


def test_embodiments_registry():
    assert gr00t_utils.EMBODIMENTS["gr1"] is gr00t_utils.GR1
    assert gr00t_utils.GR1.embodiment_id == 20
    assert gr00t_utils.GR1.action_dim == 26
    assert int(gr00t_utils.GR00TDim.MAX_STATE_DIM) == 128
    assert gr00t_utils.EMBODIMENTS["libero_panda"] is gr00t_utils.LIBERO_PANDA
    assert gr00t_utils.LIBERO_PANDA.embodiment_id == 2
    assert sum(gr00t_utils.LIBERO_PANDA.state_group_dims.values()) == 8


def test_load_embodiment_id_fallback_and_checkpoint(tmp_path):
    assert gr00t_utils.load_embodiment_id("gr1") == 20
    assert gr00t_utils.load_embodiment_id("libero_panda") == 2
    (tmp_path / "embodiment_id.json").write_text(json.dumps({"gr1": 99}))
    assert gr00t_utils.load_embodiment_id("gr1", str(tmp_path)) == 99
    assert gr00t_utils.load_embodiment_id("gr1", str(tmp_path / "missing")) == 20


def test_load_embodiment_id_unknown_raises():
    with pytest.raises(KeyError):
        gr00t_utils.load_embodiment_id("does_not_exist")
