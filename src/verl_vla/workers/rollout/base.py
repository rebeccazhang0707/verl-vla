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

from verl.workers.rollout.base import _ROLLOUT_REGISTRY

__all__ = ["register_vla_rollouts"]


def register_vla_rollouts():
    registrations = {
        ("hf", "sync"): "verl_vla.workers.rollout.hf_rollout.HFRollout",
        ("hf", "async"): "verl_vla.workers.rollout.hf_rollout.HFRollout",
        ("hf", "async_envloop"): "verl_vla.workers.rollout.hf_rollout.HFRollout",
        # GR00T N1.6 + Arena SAC: dedicated rollout (HFRollout cannot be reused — the
        # GR00T ``sac_sample_actions`` returns a plain dict with no ``.to_data_proto``,
        # and GR00TRolloutRob skips NaiveRolloutRob.__init__'s disk checkpoint load,
        # adding embodiment metadata + action-horizon validation). Obs packing and
        # action decoding live in the env (scheme Y), not in this rollout. Registered
        # as a **string path** (like ``hf``) so ``get_rollout_class`` imports it lazily
        # and a gr00t-free host never triggers the import. Selected via ``rollout.name=gr00t``.
        ("gr00t", "sync"): "verl_vla.workers.rollout.naive_rollout_gr00t.GR00TRolloutRob",
        ("gr00t", "async"): "verl_vla.workers.rollout.naive_rollout_gr00t.GR00TRolloutRob",
        ("gr00t", "async_envloop"): "verl_vla.workers.rollout.naive_rollout_gr00t.GR00TRolloutRob",
    }
    _ROLLOUT_REGISTRY.update(registrations)
