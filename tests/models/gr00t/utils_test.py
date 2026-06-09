# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Unit tests for the GR00T N1.6 gr00t-free helpers in ``utils.py``.

These cover the embodiment specs + pure-numpy state helpers, which import without
the gr00t package or a checkpoint. The ``GR00TN16Adapter`` itself (gr00t_policy.py)
needs the gr00t package + a checkpoint and is validated on GPU, not here.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpy")

# Load utils.py directly by file path. Importing it via ``verl_vla.models.gr00t.utils``
# would first run ``verl_vla/models/__init__.py``, which eagerly calls
# ``register_vla_models`` (pulling in transformers / verl / the pi0+openvla stack) and
# would make this pure-numpy test un-runnable on a bare CPU env. ``utils.py`` itself
# only needs the stdlib + numpy, so we side-step the package ``__init__`` chain.
_UTILS_PATH = Path(__file__).resolve().parents[3] / "src" / "verl_vla" / "models" / "gr00t" / "utils.py"
_spec = importlib.util.spec_from_file_location("verl_vla_gr00t_utils_standalone", _UTILS_PATH)
gr00t_utils = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass forward-ref (``"OrderedDict[str, int]"``) resolution,
# which looks up ``sys.modules[cls.__module__]``, finds this module.
sys.modules[_spec.name] = gr00t_utils
_spec.loader.exec_module(gr00t_utils)


def test_gr1_state_group_dims_sum_to_26():
    assert sum(gr00t_utils.GR1_STATE_GROUP_DIMS.values()) == 26
    assert list(gr00t_utils.GR1_STATE_GROUP_DIMS.keys()) == [
        "left_arm", "right_arm", "left_hand", "right_hand"
    ]


def test_split_flat_state_to_groups_shapes_and_order():
    B = 4
    # build a flat state where each group is filled with a distinct constant
    dims = gr00t_utils.GR1_STATE_GROUP_DIMS
    blocks = []
    for i, d in enumerate(dims.values()):
        blocks.append(np.full((B, d), float(i), dtype=np.float32))
    flat = np.concatenate(blocks, axis=-1)  # (B, 26)

    groups = gr00t_utils.split_flat_state_to_groups(flat)

    assert list(groups.keys()) == list(dims.keys())
    for i, (key, d) in enumerate(dims.items()):
        assert groups[key].shape == (B, d)
        assert np.allclose(groups[key], float(i)), f"group {key} not in expected slice order"


def test_split_flat_state_rejects_wrong_width():
    bad = np.zeros((2, 25), dtype=np.float32)  # 25 != 26
    with pytest.raises(AssertionError):
        gr00t_utils.split_flat_state_to_groups(bad)


def test_embodiment_id_and_dims_defaults():
    # N1.6 fallback defaults (authoritative values come from the checkpoint config)
    # Embodiment-specific constants live on the EmbodimentSpec (GR1)...
    assert int(gr00t_utils.GR1.embodiment_id) == 20
    assert int(gr00t_utils.GR1.action_dim) == 26
    assert int(gr00t_utils.GR1.state_dim) == 26
    assert gr00t_utils.GR1.state_group_dims is gr00t_utils.GR1_STATE_GROUP_DIMS
    assert gr00t_utils.get_embodiment_spec("gr1") is gr00t_utils.GR1
    # ...while model-level dims stay on GR00TDim.
    assert int(gr00t_utils.GR00TDim.MAX_STATE_DIM) == 128
    assert int(gr00t_utils.GR00TDim.MAX_ACTION_DIM) == 128


def test_get_embodiment_spec_unknown_raises():
    with pytest.raises(KeyError):
        gr00t_utils.get_embodiment_spec("does_not_exist")


def test_load_embodiment_id_fallback_table():
    # No model_path -> resolve from the copied EMBODIMENT_ID_FALLBACK table.
    assert gr00t_utils.load_embodiment_id("gr1") == 20
    assert gr00t_utils.load_embodiment_id("gr1") == gr00t_utils.GR1.embodiment_id


def test_load_embodiment_id_unknown_tag_raises():
    with pytest.raises(KeyError):
        gr00t_utils.load_embodiment_id("does_not_exist")


def test_load_embodiment_id_prefers_checkpoint(tmp_path):
    import json

    # A checkpoint's embodiment_id.json is authoritative over the fallback table.
    (tmp_path / "embodiment_id.json").write_text(json.dumps({"gr1": 99}))
    assert gr00t_utils.load_embodiment_id("gr1", str(tmp_path)) == 99


def test_load_embodiment_id_missing_file_falls_back(tmp_path):
    # model_path given but no embodiment_id.json -> fall back to the table.
    assert gr00t_utils.load_embodiment_id("gr1", str(tmp_path)) == 20
