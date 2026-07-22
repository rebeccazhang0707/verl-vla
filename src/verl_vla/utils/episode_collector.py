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

"""Streaming per-env episode collection for SAC under auto-reset rollouts.

With ``auto_reset=True`` consecutive rollouts are physically continuous per
batch row, but the legacy per-rollout masking (``_build_sac_transition_masks``)
drops every transition after the last ``done`` of a rollout window. Episodes
that straddle rollout boundaries therefore lose their early/middle transitions,
biasing the replay pool towards near-termination states (see
``docs/reinforcement-learning/episodic-replay.md``).

``EpisodeCollector`` fixes this by treating the collated ``[B, S]`` rollout
output as ``B`` continuous per-lane transition streams. Slots accumulate in a
per-lane open buffer; a complete episode is flushed the moment its ``done``
slot arrives (the terminal ``t1`` observation is a self-copy by convention, so
nothing from the future is needed). Only the residual after the last ``done``
stays buffered, and its transitions are emitted on later ingests once their
next observations arrive — no transition is ever dropped for lack of a
``done`` in the same window.

Lane identity relies on the deterministic env-loop collate keeping batch row
``b`` mapped to the same physical env across rollouts; ``ingest`` asserts the
lane count never changes. Any event that breaks physical continuity (an eval
window reusing the training envs, a checkpoint restart) must be preceded by
``force_flush_all`` or a fresh collector.
"""

from dataclasses import dataclass

import numpy as np
import torch
from verl import DataProto

from verl_vla.utils.data import reduce_substep_dims
from verl_vla.utils.keys import ACTION_KEY, FEEDBACK_KEY, OBS_KEY


@dataclass
class _Slot:
    """One macro-step of a single lane: obs/action storage plus chunk-level feedback.

    The slot — not the transition — is the unit of buffering, because a
    transition cannot be formed when a step arrives: its ``t1`` observation and
    its episode-level ``success`` are only known later, possibly in a future
    rollout window. Slots therefore carry every per-step signal needed at that
    deferred flush: ``terminated``/``truncated`` play distinct roles (``done``
    decides where a segment ends; only ``terminated`` disables Q bootstrapping),
    and ``success`` may fire on any step, so it must be remembered until the
    segment's ``any``-reduction in ``_flush_segment``.
    """

    tensors: dict[str, torch.Tensor]
    non_tensors: dict[str, object]
    reward: float
    terminated: bool
    truncated: bool
    success: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass
class _Transition:
    """A ``(t0, t1)`` slot pairing created at flush time, once ``t1`` is known.

    Holds references (not copies) into the slot storage: a slot shared as one
    transition's ``t1`` and the next one's ``t0`` is stored once. ``success``
    is the episode-level outcome broadcast over the segment.
    """

    t0: _Slot
    t1: _Slot
    success: bool


class EpisodeCollector:
    """Stateful per-lane episode collector emitting flat SAC transition batches.

    Output schema per transition (every row valid):

    - ``t0.<key>`` / ``t1.<key>`` for every ``obs.*`` and ``action.*`` key;
      ``t1`` comes from the next slot of the same episode. The terminal
      transition of a segment self-copies ``t1`` (matching the
      ``add_transition_prefixes`` boundary rule) except on forced flushes,
      where the retained newest slot provides a real next observation.
    - ``info.rewards`` (chunk reward minus ``step_penalty``),
      ``info.terminateds`` (per-slot terminated flag for bootstrap masking),
      ``info.valids`` (all ones), and ``info.success_mask`` (episode-level
      success broadcast over the segment).
    """

    def __init__(self, *, step_penalty: float = 0.0, max_open_len: int = 128):
        if max_open_len < 2:
            raise ValueError(f"max_open_len must be >= 2, got {max_open_len}")
        self.step_penalty = float(step_penalty)
        self.max_open_len = int(max_open_len)

        self._open: list[list[_Slot]] | None = None
        self._non_tensor_dtypes: dict[str, np.dtype] = {}
        self._episodes_flushed = 0
        self._success_episodes_flushed = 0
        self._forced_flushes = 0
        self._transitions_emitted = 0
        self._slots_dropped = 0

    def ingest(self, rollout_output: DataProto) -> DataProto | None:
        """Consume one collated ``[B, S]`` rollout and return flushed transitions.

        Returns ``None`` when no segment completed (no ``done`` seen and no
        buffer exceeded ``max_open_len``). The input batch is only read, never
        mutated; slot storage is cloned so the rollout tensors are not retained
        by the open buffers.
        """
        batch = rollout_output.batch
        terminated = reduce_substep_dims(batch[f"{FEEDBACK_KEY}.terminated"].bool(), reduction="any").cpu()
        truncated = reduce_substep_dims(batch[f"{FEEDBACK_KEY}.truncated"].bool(), reduction="any").cpu()
        success = reduce_substep_dims(batch[f"{FEEDBACK_KEY}.success"].bool(), reduction="any").cpu()
        reward = reduce_substep_dims(batch[f"{FEEDBACK_KEY}.reward"].float(), reduction="sum").cpu()
        num_lanes, num_steps = terminated.shape

        if self._open is None:
            self._open = [[] for _ in range(num_lanes)]
        elif len(self._open) != num_lanes:
            raise ValueError(
                f"Lane count changed from {len(self._open)} to {num_lanes}: the row<->env mapping is no "
                "longer stable, so cross-rollout episode collection would splice unrelated streams. "
                "Episodic replay requires a fixed env topology within a run."
            )

        prefixes = (f"{OBS_KEY}.", f"{ACTION_KEY}.")
        tensor_keys = [key for key in batch.keys() if key.startswith(prefixes)]
        non_tensor_keys = [key for key in rollout_output.non_tensor_batch.keys() if key.startswith(prefixes)]
        for key in non_tensor_keys:
            self._non_tensor_dtypes[key] = rollout_output.non_tensor_batch[key].dtype

        transitions: list[_Transition] = []
        for lane in range(num_lanes):
            open_slots = self._open[lane]
            for step in range(num_steps):
                slot = _Slot(
                    tensors={key: batch[key][lane, step].clone() for key in tensor_keys},
                    non_tensors={
                        key: self._copy_non_tensor(rollout_output.non_tensor_batch[key][lane, step])
                        for key in non_tensor_keys
                    },
                    reward=float(reward[lane, step]),
                    terminated=bool(terminated[lane, step]),
                    truncated=bool(truncated[lane, step]),
                    success=bool(success[lane, step]),
                )
                open_slots.append(slot)
                if slot.done:
                    transitions.extend(self._flush_segment(open_slots, next_slot=None))
                    open_slots.clear()
                elif len(open_slots) >= self.max_open_len:
                    # Forced truncation: keep the newest slot so every emitted
                    # transition bootstraps from a real next observation.
                    transitions.extend(self._flush_segment(open_slots[:-1], next_slot=open_slots[-1]))
                    del open_slots[:-1]
                    self._forced_flushes += 1

        return self._transitions_to_dataproto(transitions)

    def force_flush_all(self) -> DataProto | None:
        """Flush every open buffer ahead of a continuity break (eval, shutdown).

        The newest slot of each lane is dropped: once continuity breaks, its
        true next observation never arrives, so no valid transition can be
        formed from it. Everything older is emitted as a truncated segment.
        """
        if self._open is None:
            return None

        transitions: list[_Transition] = []
        for open_slots in self._open:
            if len(open_slots) >= 2:
                transitions.extend(self._flush_segment(open_slots[:-1], next_slot=open_slots[-1]))
                self._forced_flushes += 1
            self._slots_dropped += min(len(open_slots), 1)
            open_slots.clear()

        return self._transitions_to_dataproto(transitions)

    def metrics(self) -> dict[str, float]:
        open_lens = [len(slots) for slots in self._open] if self._open is not None else []
        total_flushes = self._episodes_flushed + self._forced_flushes
        total_collected = self._transitions_emitted + self._slots_dropped
        return {
            # Train-time success rate over completed episodes; compare against eval SR.
            "data/collector_online_success_rate": (
                self._success_episodes_flushed / self._episodes_flushed if self._episodes_flushed else 0.0
            ),
            # Cumulative transitions handed to the replay path; the slope is the
            # effective data throughput per rollout window.
            "data/collector_transitions_emitted": float(self._transitions_emitted),
            # Share of segment flushes that were forced truncations rather than real
            # episode ends. Growth outside eval boundaries means episodic_max_open_len
            # is too small for the episode horizon and episodes are being chopped up.
            "data/collector_forced_flush_ratio": (self._forced_flushes / total_flushes if total_flushes else 0.0),
            # Fraction of collected steps discarded at continuity breaks
            # (force_flush_all drops one slot per lane). Should stay near zero and
            # only tick up at eval/shutdown.
            "data/collector_drop_ratio": (self._slots_dropped / total_collected if total_collected else 0.0),
            # Longest open episode relative to the forced-truncation cap; sustained
            # values near 1.0 predict imminent forced flushes.
            "data/collector_open_fill_ratio": float(max(open_lens, default=0)) / self.max_open_len,
        }

    def _flush_segment(self, slots: list[_Slot], *, next_slot: _Slot | None) -> list[_Transition]:
        """Build transitions for one segment.

        ``next_slot=None`` marks a real episode end (last slot is ``done``, its
        ``t1`` self-copies); a forced flush passes the retained newest slot as
        the final transition's ``t1`` source instead.
        """
        if not slots:
            return []
        success = any(slot.success for slot in slots)
        transitions = []
        for idx, slot in enumerate(slots):
            if idx + 1 < len(slots):
                t1 = slots[idx + 1]
            elif next_slot is not None:
                t1 = next_slot
            else:
                t1 = slot
            transitions.append(_Transition(t0=slot, t1=t1, success=success))

        if next_slot is None:
            self._episodes_flushed += 1
            self._success_episodes_flushed += int(success)
        self._transitions_emitted += len(transitions)
        return transitions

    def _transitions_to_dataproto(self, transitions: list[_Transition]) -> DataProto | None:
        if not transitions:
            return None

        tensors: dict[str, torch.Tensor] = {}
        non_tensors: dict[str, np.ndarray] = {}
        template = transitions[0].t0

        for key in template.tensors.keys():
            tensors[f"t0.{key}"] = torch.stack([transition.t0.tensors[key] for transition in transitions])
            tensors[f"t1.{key}"] = torch.stack([transition.t1.tensors[key] for transition in transitions])
        for key in template.non_tensors.keys():
            dtype = self._non_tensor_dtypes[key]
            non_tensors[f"t0.{key}"] = self._stack_non_tensor(
                [transition.t0.non_tensors[key] for transition in transitions], dtype
            )
            non_tensors[f"t1.{key}"] = self._stack_non_tensor(
                [transition.t1.non_tensors[key] for transition in transitions], dtype
            )

        tensors["info.rewards"] = torch.tensor(
            [transition.t0.reward - self.step_penalty for transition in transitions], dtype=torch.float32
        )
        tensors["info.terminateds"] = torch.tensor(
            [float(transition.t0.terminated) for transition in transitions], dtype=torch.float32
        )
        tensors["info.valids"] = torch.ones(len(transitions), dtype=torch.float32)
        tensors["info.success_mask"] = torch.tensor(
            [float(transition.success) for transition in transitions], dtype=torch.float32
        )

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    @staticmethod
    def _copy_non_tensor(value: object) -> object:
        return value.copy() if isinstance(value, np.ndarray) else value

    @staticmethod
    def _stack_non_tensor(values: list, dtype: np.dtype) -> np.ndarray:
        first = values[0]
        if isinstance(first, np.ndarray) and first.ndim > 0:
            return np.stack(values).astype(dtype, copy=False)
        return np.array(values, dtype=dtype)
