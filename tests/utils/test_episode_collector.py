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

"""Unit tests for the streaming EpisodeCollector (episodic SAC replay collection).

These tests use synthetic collated rollout DataProto objects and run on CPU
without Ray, models, or simulators. Observations encode ``(lane, time)`` so
transition continuity can be asserted exactly, including across rollout
windows where the legacy per-rollout masking path drops data.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl import DataProto

from verl_vla.trainer.sac.sac_ray_trainer import prepare_sac_actor_input
from verl_vla.utils.episode_collector import EpisodeCollector

NUM_SUBSTEPS = 2


def make_rollout(
    num_lanes: int,
    num_steps: int,
    *,
    t0: int = 0,
    terminated: set[tuple[int, int]] = frozenset(),
    truncated: set[tuple[int, int]] = frozenset(),
    success: set[tuple[int, int]] = frozenset(),
) -> DataProto:
    """Build a synthetic collated [B, S] rollout whose obs encode (lane, time)."""
    times = torch.arange(t0, t0 + num_steps, dtype=torch.float32).expand(num_lanes, num_steps)
    lanes = torch.arange(num_lanes, dtype=torch.float32).unsqueeze(1).expand(num_lanes, num_steps)

    state = torch.stack([lanes, times], dim=-1)  # [B, S, 2]
    image = (lanes * 1000 + times).unsqueeze(-1).unsqueeze(-1).expand(num_lanes, num_steps, 2, 2).clone()
    action = torch.stack([lanes, times, torch.full_like(times, 0.5)], dim=-1)  # [B, S, 3]

    terminated_steps = torch.zeros(num_lanes, num_steps, NUM_SUBSTEPS, dtype=torch.bool)
    truncated_steps = torch.zeros_like(terminated_steps)
    success_steps = torch.zeros_like(terminated_steps)
    reward_steps = torch.zeros(num_lanes, num_steps, NUM_SUBSTEPS, dtype=torch.float32)
    reward_steps[:, :, 0] = lanes * 100 + times

    for lane, step in terminated:
        terminated_steps[lane, step, -1] = True
    for lane, step in truncated:
        truncated_steps[lane, step, -1] = True
    for lane, step in success:
        success_steps[lane, step, 0] = True

    task_id = (np.arange(num_lanes, dtype=np.int64) + 7)[:, None].repeat(num_steps, axis=1)
    task = np.empty((num_lanes, num_steps), dtype=object)
    for lane in range(num_lanes):
        task[lane, :] = f"task-{lane}"

    return DataProto.from_dict(
        tensors={
            "obs.state": state,
            "obs.image": image,
            "action.action": action,
            "next.terminated": terminated_steps,
            "next.truncated": truncated_steps,
            "next.success": success_steps,
            "next.reward": reward_steps,
        },
        non_tensors={"obs.task_id": task_id, "obs.task": task},
    )


def sort_by_lane_time(data: DataProto) -> DataProto:
    key = data.batch["t0.obs.state"][:, 0] * 1_000_000 + data.batch["t0.obs.state"][:, 1]
    idx = torch.argsort(key)
    idx_np = idx.numpy()
    return DataProto.from_dict(
        tensors={k: v[idx] for k, v in data.batch.items()},
        non_tensors={k: v[idx_np] for k, v in data.non_tensor_batch.items()},
    )


def lane_time(data: DataProto, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    state = data.batch[f"{prefix}.obs.state"]
    return state[:, 0], state[:, 1]


def test_in_window_equivalence_with_legacy_path():
    """Episodes completing inside one window must match the legacy masking path."""
    kwargs = dict(
        terminated={(0, 1), (1, 2)},
        truncated={(0, 3)},
        success={(0, 1)},
    )
    trainer_config = SimpleNamespace(step_penalty=0.25)

    legacy = prepare_sac_actor_input(
        make_rollout(3, 4, **kwargs), config=None, trainer_config=trainer_config, global_steps=0
    )
    legacy_valid_idx = torch.nonzero(legacy.batch["info.valids"] > 0).flatten()
    legacy_valid = DataProto.from_dict(
        tensors={k: v[legacy_valid_idx] for k, v in legacy.batch.items()},
        non_tensors={k: v[legacy_valid_idx.numpy()] for k, v in legacy.non_tensor_batch.items()},
    )

    collector = EpisodeCollector(step_penalty=0.25, max_open_len=100)
    ours = collector.ingest(make_rollout(3, 4, **kwargs))
    assert ours is not None

    # lane0: two full episodes (steps 0-1, 2-3); lane1: steps 0-2; lane2: no done -> stays open.
    assert len(ours) == len(legacy_valid) == 7

    legacy_valid = sort_by_lane_time(legacy_valid)
    ours = sort_by_lane_time(ours)

    for key in ("t0.obs.state", "t0.obs.image", "t0.action.action", "t1.obs.state", "t1.obs.image"):
        torch.testing.assert_close(ours.batch[key], legacy_valid.batch[key])
    for key in ("info.rewards", "info.terminateds", "info.valids", "info.success_mask"):
        torch.testing.assert_close(ours.batch[key], legacy_valid.batch[key])

    # t1.action matches legacy on non-final rows only: at segment ends the legacy
    # shift leaks the next episode's action while the collector self-copies.
    finals = {(0.0, 1.0), (0.0, 3.0), (1.0, 2.0)}
    lanes, times = lane_time(ours, "t0")
    non_final = torch.tensor(
        [(lane.item(), time.item()) not in finals for lane, time in zip(lanes, times, strict=True)]
    )
    torch.testing.assert_close(
        ours.batch["t1.action.action"][non_final], legacy_valid.batch["t1.action.action"][non_final]
    )

    assert ours.non_tensor_batch["t0.obs.task_id"].dtype == np.int64
    np.testing.assert_array_equal(
        ours.non_tensor_batch["t0.obs.task_id"], legacy_valid.non_tensor_batch["t0.obs.task_id"]
    )
    np.testing.assert_array_equal(ours.non_tensor_batch["t0.obs.task"], legacy_valid.non_tensor_batch["t0.obs.task"])


def test_cross_window_episode_is_fully_recovered():
    """An episode straddling two windows keeps its early transitions (the legacy path drops them)."""
    collector = EpisodeCollector(step_penalty=0.0, max_open_len=100)

    assert collector.ingest(make_rollout(1, 3, t0=0)) is None  # no done -> everything stays open

    out = collector.ingest(make_rollout(1, 3, t0=3, terminated={(0, 1)}, success={(0, 1)}))
    assert out is not None
    assert len(out) == 5  # times 0..4; window-2 step 2 stays open

    out = sort_by_lane_time(out)
    _, t0_times = lane_time(out, "t0")
    _, t1_times = lane_time(out, "t1")
    torch.testing.assert_close(t0_times, torch.arange(5, dtype=torch.float32))
    # Continuity across the rollout boundary: t1 is the true next obs, including 2 -> 3.
    torch.testing.assert_close(t1_times[:4], torch.arange(1, 5, dtype=torch.float32))
    assert t1_times[4].item() == t0_times[4].item()  # terminal self-copy

    torch.testing.assert_close(out.batch["info.terminateds"], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(out.batch["info.success_mask"], torch.ones(5))
    torch.testing.assert_close(out.batch["info.rewards"], torch.arange(5, dtype=torch.float32))

    metrics = collector.metrics()
    assert metrics["data/collector_online_success_rate"] == 1.0
    assert metrics["data/collector_transitions_emitted"] == 5.0
    assert metrics["data/collector_forced_flush_ratio"] == 0.0
    assert metrics["data/collector_open_fill_ratio"] == pytest.approx(1 / 100)  # window-2 step 2 stays open


def test_forced_flush_keeps_real_next_obs():
    collector = EpisodeCollector(step_penalty=0.0, max_open_len=3)

    out = collector.ingest(make_rollout(1, 4, t0=0, success={(0, 0)}))
    assert out is not None
    assert len(out) == 2  # buffer hits the cap at step 2: flush times 0-1, keep time 2

    out = sort_by_lane_time(out)
    _, t0_times = lane_time(out, "t0")
    _, t1_times = lane_time(out, "t1")
    torch.testing.assert_close(t0_times, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(t1_times, torch.tensor([1.0, 2.0]))  # real next obs, no self-copy
    torch.testing.assert_close(out.batch["info.terminateds"], torch.zeros(2))  # truncation: keep bootstrapping
    torch.testing.assert_close(out.batch["info.success_mask"], torch.ones(2))  # success seen before the cut

    # The retained slot chains into the next window and flushes with the real done.
    out2 = collector.ingest(make_rollout(1, 2, t0=4, terminated={(0, 0)}))
    assert out2 is not None and len(out2) == 3  # times 2, 3, 4
    out2 = sort_by_lane_time(out2)
    _, t0_times = lane_time(out2, "t0")
    _, t1_times = lane_time(out2, "t1")
    torch.testing.assert_close(t0_times, torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(t1_times, torch.tensor([3.0, 4.0, 4.0]))
    metrics = collector.metrics()
    assert metrics["data/collector_forced_flush_ratio"] == 0.5  # one forced flush, one real episode end
    assert metrics["data/collector_open_fill_ratio"] == pytest.approx(1 / 3)  # window-2 step 1 remains


def test_force_flush_all_empties_buffers_and_drops_newest_slot():
    collector = EpisodeCollector(step_penalty=0.0, max_open_len=100)
    assert collector.ingest(make_rollout(2, 3, t0=0)) is None

    flushed = collector.force_flush_all()
    assert flushed is not None
    assert len(flushed) == 4  # per lane: 3 open slots -> 2 transitions, newest dropped
    torch.testing.assert_close(flushed.batch["info.terminateds"], torch.zeros(4))

    assert collector.force_flush_all() is None
    assert collector.metrics()["data/collector_open_fill_ratio"] == 0.0
    assert collector.metrics()["data/collector_drop_ratio"] == pytest.approx(2 / 6)  # 2 dropped vs 4 emitted

    # Collection restarts cleanly after the break.
    out = collector.ingest(make_rollout(2, 2, t0=10, terminated={(0, 1), (1, 1)}))
    assert out is not None and len(out) == 4


def test_lane_count_change_raises():
    collector = EpisodeCollector(step_penalty=0.0, max_open_len=100)
    collector.ingest(make_rollout(2, 2))
    with pytest.raises(ValueError, match="Lane count changed"):
        collector.ingest(make_rollout(3, 2))


def test_input_batch_is_not_mutated():
    rollout = make_rollout(1, 2, terminated={(0, 1)})
    collector = EpisodeCollector(step_penalty=0.0, max_open_len=100)
    out = collector.ingest(rollout)
    assert out is not None
    assert "next.terminated" in rollout.batch.keys()
    assert "obs.state" in rollout.batch.keys()


if __name__ == "__main__":
    test_in_window_equivalence_with_legacy_path()
    test_cross_window_episode_is_fully_recovered()
    test_forced_flush_keeps_real_next_obs()
    test_force_flush_all_empties_buffers_and_drops_newest_slot()
    test_lane_count_change_raises()
    test_input_batch_is_not_mutated()
    print("all episode collector tests passed")
