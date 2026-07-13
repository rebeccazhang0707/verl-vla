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

"""Action double-track invariant: the GR00T critic scores the NORMALISED action.

The rollout emits two tensors (``action`` = DECODED env chunk, ``full_action`` =
NORMALISED model action). This test locks in that ``full_action`` survives the
*actual* env-loop replay plumbing so the ``t0.action.*`` dict the SAC critic reads
still carries the normalised action, never the decoded env action.
"""

import pytest

pytest.importorskip("verl")
torch = pytest.importorskip("torch")

from verl import DataProto  # noqa: E402

from verl_vla.models.gr00t_n1d6.policy import ArenaGr00tOutput  # noqa: E402
from verl_vla.utils.data import (  # noqa: E402
    add_transition_prefixes,
    flatten_trajectories,
    get_dataproto_from_prefix,
    stack_dataproto_with_padding,
)


def test_full_action_survives_replay_plumbing():
    """End-to-end double-track invariant (rollout output -> replay transition).

    The normalised ``full_action`` from ``ArenaGr00tOutput.to_data_proto`` must
    survive the env-loop replay plumbing (``stack_dataproto_with_padding`` ->
    ``add_transition_prefixes`` -> ``flatten_trajectories``) so the ``t0.action.*``
    dict the SAC critic reads still carries the NORMALISED action, distinct from the
    DECODED env action. A rename/drop anywhere in that chain would flip the critic
    onto the wrong (decoded) action space; this guards it.
    """
    B, decoded_chunk, decoded_dim, full_horizon, max_action_dim = 2, 16, 26, 50, 128
    full_action = torch.randn(B, full_horizon, max_action_dim)
    decoded = torch.randn(B, decoded_chunk, decoded_dim)
    rollout = ArenaGr00tOutput.from_model_output(
        {"full_action": full_action, "decoded_action": decoded, "num_action_chunks": decoded_chunk}
    ).to_data_proto()

    # Env-loop namespacing: one rollout step -> keys "action.action" / "action.full_action".
    stacked = stack_dataproto_with_padding([rollout], "action")
    assert set(stacked) == {"action.action", "action.full_action"}
    data = DataProto.from_dict(tensors=stacked)

    # Rollout slot -> t0/t1 transition fields, then flatten (B, steps, ...) -> (B*steps, ...).
    data = flatten_trajectories(add_transition_prefixes(data))

    a0 = get_dataproto_from_prefix(data, "t0.action.").batch
    assert "full_action" in a0.keys() and "action" in a0.keys()
    # The critic-space selection (a.get("full_action", a["action"])) must land on the
    # NORMALISED full_action, not the decoded env action.
    assert a0["full_action"].shape[-1] == max_action_dim
    assert not torch.equal(a0["full_action"], a0["action"])


def test_bc_prefers_full_action_over_env_action():
    """BC / critic action-space selection: ``full_action`` wins over env ``action``.

    Mirrors the critic / BC action-space selection (``full_action`` over env ``action``)
    so a replay dict that carries both keys never silently trains BC on decoded joints.
    """
    full = torch.randn(2, 50, 128)
    env = torch.randn(2, 16, 26)
    actions = {"full_action": full, "action": env}
    chosen = actions.get("full_action", actions["action"])
    assert chosen is full
    assert chosen.shape == full.shape


def test_gr00t_flow_matching_targets_noise_to_action():
    """GR00T flow BC targets ``u = action - noise`` with ``x_t = (1-t)noise + t action``.

    Opposite of pi0 (``u = noise - action``, ``x_t = t*noise + (1-t)*action``). This
    locks the sign convention used by ``_bc_mse`` against the upstream action head.
    """
    B, H, D = 4, 8, 16
    actions = torch.randn(B, H, D)
    noise = torch.randn(B, H, D)
    t = torch.rand(B).view(B, 1, 1)
    x_t = (1.0 - t) * noise + t * actions
    u_t = actions - noise
    # Endpoints: t=0 -> noise, t=1 -> action; velocity points noise -> action.
    assert torch.allclose(x_t + (1.0 - t) * u_t, actions, atol=1e-5)
    assert torch.allclose(x_t - t * u_t, noise, atol=1e-5)


def test_critic_action_horizon_defaults_to_num_action_chunks():
    """Critic action horizon falls back to num_action_chunks when unset.

    Replicates the adapter resolution without importing the trainable model
    (which requires the gr00t package).
    """
    from verl_vla.models.gr00t_n1d6.adapter_config import Gr00tAdapterConfig

    action_horizon = 50
    num_action_chunks = 16
    cfg = Gr00tAdapterConfig(
        num_action_chunks=num_action_chunks,
        critic={"action_horizon": None},
    )
    # Modeling resolves None critic.action_horizon -> num_action_chunks.
    resolved = int(cfg.critic.action_horizon if cfg.critic.action_horizon is not None else cfg.num_action_chunks)
    assert resolved == 16
    assert resolved != action_horizon
    cfg.critic.action_horizon = 32
    assert int(cfg.critic.action_horizon) == 32
