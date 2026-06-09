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

"""In-process GR00T N1.6 + verl-vla SAC smoke test (Workflow A — NO simulator).

Loads the checkpoint, builds inputs from **synthetic** obs (no Isaac Sim), and
exercises the full verl-vla SAC action path on the real model:

  1. ``GR00TN16Adapter.build_inputs``  (synthetic image+state → eagle tensors)
  2. ``sac_sample_actions``            (rollout-time NORMALISED action chunk)
  3. ``decode_actions_flat``           (NORMALISED chunk → 26-DOF joints, fixed base)
  4. ``sac_get_critic_value``          (rollout-time min-over-heads Q)
  5. ``sac_forward_state_features``    (registered FSDP state-feature entry)
  6. ``sac_forward_actor``             (grad-enabled action sampling)
  7. ``sac_forward_critic``            (critic Q on sampled action)
  8. ``bc_loss`` + grad-enabled actor backward

Expect ``8/8 PASS``. This is **docker-only** (needs gr00t + transformers 4.51.3 +
the checkpoint); it is NOT runnable on the CPU dev host. The CPU contract is only
that this module *imports* and its argparse parses — every gr00t / transformers
import is deferred into ``main`` so a gr00t-free host can import the file.

Run inside the Arena GR00T docker (isaaclab_arena:cuda_gr00t_gn16):

    PYTHONPATH=/opt/groot_deps:/code/verl-vla/src /isaac-sim/python.sh \
        examples/arena_sac/smoke_test_gr00t_arena.py \
        --ckpt /models/checkpoint-5000-export --num-envs 2 --denoise-steps 2

Useful flags: ``--image-size``, ``--critic-heads``, ``--chunk``,
``--compare-gr00t-policy`` (also loads ``gr00t.policy.Gr00tPolicy`` for a
numerical cross-check of the decoded action, ~2x memory).
"""

import argparse
import sys
import traceback

import numpy as np
import torch

# Checkpoint action horizon (config.json action_horizon=50); guard upper bound anchor.
CKPT_ACTION_HORIZON = 50


def _patch_transformers_eagle():
    """Compat shims so the Eagle remote code loads (delegates to the Phase-2 patch)."""
    try:
        from isaaclab_arena_gr00t.utils.eagle_config_compat import apply_eagle_config_compat

        apply_eagle_config_compat()
        return
    except Exception:
        pass
    from verl_vla.models.gr00t.modeling_gr00t_sac import _patch_eagle_compat

    _patch_eagle_compat()


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke-test argparse parser (split out so it is unit-testable on CPU)."""
    ap = argparse.ArgumentParser(description="In-process GR00T N1.6 + verl-vla SAC smoke test (no simulator)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embodiment-tag", default="gr1")
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=16, help="num_action_chunks executed per query")
    ap.add_argument(
        "--action-horizon",
        type=int,
        default=CKPT_ACTION_HORIZON,
        help="model action horizon (checkpoint config.json = 50); guard upper bound",
    )
    ap.add_argument("--critic-heads", type=int, default=10)
    ap.add_argument("--denoise-steps", type=int, default=2, help="num_inference_timesteps for the flow sampler")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-gr00t-policy", action="store_true", help="also load Gr00tPolicy for a numerical check")
    return ap


def _synthetic_obs(num_envs: int, image_size: int, seed: int):
    """Build a synthetic (full_image[B,H,W,C] uint8, state26[B,26] float32) batch."""
    rng = np.random.default_rng(seed)
    full_image = rng.integers(0, 255, size=(num_envs, image_size, image_size, 3), dtype=np.uint8)
    state26 = rng.standard_normal(size=(num_envs, 26)).astype(np.float32)
    tasks = ["Place the sauce bottle on the top shelf of the fridge, and close the fridge door."] * num_envs
    return full_image, state26, tasks


def main() -> int:
    args = build_parser().parse_args()

    # Guard (#2) anchor check (pure helper; CPU-importable).
    from verl_vla.workers.rollout.naive_rollout_gr00t import assert_action_horizon_invariant

    assert_action_horizon_invariant(
        num_action_chunks=args.chunk,
        critic_action_horizon=args.chunk,
        action_horizon=args.action_horizon,
    )

    device = args.device
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    from transformers import AutoConfig, AutoModel
    from verl import DataProto

    from verl_vla.models.gr00t.gr00t_policy import GR00TN16Adapter
    from verl_vla.models.gr00t.modeling_gr00t_sac import register_gr00t_sac
    from verl_vla.models.gr00t.utils import load_embodiment_id

    _patch_transformers_eagle()
    register_gr00t_sac()

    passed, failed = 0, 0

    def _check(name, fn):
        nonlocal passed, failed
        try:
            detail = fn()
            passed += 1
            print(f"[PASS] {name}: {detail}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {e}", flush=True)
            traceback.print_exc()

    # ---- load model + adapter ----
    scfg = AutoConfig.from_pretrained(args.ckpt)
    scfg.sac_enable = True
    scfg.action_dim = 26
    scfg.embodiment_id = load_embodiment_id(args.embodiment_tag, args.ckpt)
    scfg.critic_head_num = args.critic_heads
    scfg.num_inference_timesteps = args.denoise_steps
    model = AutoModel.from_pretrained(args.ckpt, config=scfg).to(device=device, dtype=dtype)
    adapter = GR00TN16Adapter(args.ckpt, embodiment_tag=args.embodiment_tag)

    full_image, state26, tasks = _synthetic_obs(args.num_envs, args.image_size, args.seed)
    B = args.num_envs

    # 1. build_inputs
    inputs = {}
    raw_groups = {}

    def _c1():
        nonlocal inputs, raw_groups
        inputs, raw_groups = adapter.build_inputs(full_image, state26, tasks)
        return f"input_ids={tuple(inputs['input_ids'].shape)} state={tuple(inputs['state'].shape)}"

    _check("1.build_inputs", _c1)

    # Pack the env-side obs DataProto exactly like the env (scheme Y).
    def _packed_prompts():
        pv = inputs["pixel_values"]
        pv = torch.stack(pv, dim=0) if isinstance(pv, list) else pv
        return DataProto.from_dict(
            tensors={
                "images": pv.to(device),
                "lang_tokens": inputs["input_ids"].to(device),
                "lang_masks": inputs["attention_mask"].to(device),
                "states": inputs["state"].to(device),
            }
        )

    chunk_norm = {}

    # 2. sac_sample_actions (rollout path)
    def _c2():
        nonlocal chunk_norm
        with torch.autocast(device_type="cuda", dtype=dtype), torch.no_grad():
            out = model.sac_sample_actions(_packed_prompts(), validate=True)
        full_action = out["action"].detach().float()
        assert full_action.shape[0] == B
        chunk_norm["full"] = full_action
        chunk_norm["chunk"] = full_action[:, : args.chunk]
        return f"action={tuple(full_action.shape)} log_probs={tuple(out['log_probs'].shape)}"

    _check("2.sac_sample_actions", _c2)

    # 3. decode_actions_flat (NORMALISED chunk → 26-DOF joints, fixed base)
    def _c3():
        decoded = adapter.decode_actions_flat(chunk_norm["chunk"].cpu().numpy(), raw_groups)
        decoded = np.asarray(decoded)
        assert decoded.shape[0] == B and decoded.shape[-1] == 26
        return f"decoded={decoded.shape}"

    _check("3.decode_actions_flat", _c3)

    # 4. sac_get_critic_value (rollout path)
    def _c4():
        with torch.autocast(device_type="cuda", dtype=dtype), torch.no_grad():
            q = model.sac_get_critic_value(_packed_prompts(), {"action": chunk_norm["chunk"].to(device)})
        assert q.reshape(-1).shape[0] == B
        return f"critic_value={tuple(q.reshape(-1).shape)}"

    _check("4.sac_get_critic_value", _c4)

    sf = {}

    # 5. sac_forward_state_features (registered FSDP entry)
    def _c5():
        nonlocal sf
        with torch.autocast(device_type="cuda", dtype=dtype):
            sf = model.sac_forward_state_features(_packed_prompts())
        return f"keys={sorted(sf.keys())}"

    _check("5.sac_forward_state_features", _c5)

    actor_out = {}
    task_ids = torch.zeros(B, dtype=torch.long, device=device)

    # 6. sac_forward_actor (grad-enabled sampling)
    def _c6():
        nonlocal actor_out
        with torch.autocast(device_type="cuda", dtype=dtype):
            a0, logp, _ = model.sac_forward_actor(sf, task_ids=task_ids)
        actor_out["a0"] = a0
        actor_out["logp"] = logp
        return f"a0={tuple(a0.shape)} log_probs={None if logp is None else tuple(logp.shape)}"

    _check("6.sac_forward_actor", _c6)

    # 7. sac_forward_critic on the sampled action
    def _c7():
        with torch.autocast(device_type="cuda", dtype=dtype):
            q = model.sac_forward_critic(
                {"action": actor_out["a0"]}, sf, task_ids=task_ids, use_target_network=False, method="min"
            )
        return f"q={tuple(q.reshape(-1).shape)}"

    _check("7.sac_forward_critic", _c7)

    # 8. bc_loss + grad-enabled actor backward (BC anchor path; bc_loss_coef recipe = 0.05)
    def _c8():
        valids = torch.ones(B, device=device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss = model.bc_loss(
                obs=_packed_prompts(),
                tokenizer=None,
                actions={"action": chunk_norm["chunk"].to(device)},
                valids=valids,
            )
        loss.backward()
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        return f"bc_loss={loss.detach().float().item():.4f} grad_params={grad_params}"

    _check("8.bc_loss+backward", _c8)

    # optional: official Gr00tPolicy numerical cross-check of the decoded action
    if args.compare_gr00t_policy:

        def _c_cmp():
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.policy.gr00t_policy import Gr00tPolicy

            from verl_vla.models.gr00t.utils import GR1_STATE_GROUP_DIMS

            emb = EmbodimentTag[args.embodiment_tag.upper()]
            policy = Gr00tPolicy(embodiment_tag=emb, model_path=args.ckpt, device=device, strict=True)
            vkey = policy.modality_configs["video"].modality_keys[0]
            skeys = list(policy.modality_configs["state"].modality_keys)
            lkey = policy.language_key
            video = {vkey: full_image.reshape(B, 1, *full_image.shape[1:]).astype(np.uint8)}
            st = {}
            start = 0
            for k in skeys:
                d = GR1_STATE_GROUP_DIMS[k]
                st[k] = state26[:, start : start + d].reshape(B, 1, d).astype(np.float32)
                start += d
            lang = {lkey: [[tasks[i]] for i in range(B)]}
            action_dict, _ = policy.get_action({"video": video, "state": st, "language": lang})
            ref = np.concatenate([action_dict[k] for k in skeys], axis=-1)
            return f"gr00tpolicy_action={ref.shape}"

        _check("9.compare_gr00t_policy", _c_cmp)

    total = passed + failed
    print("\n" + "=" * 60)
    print(f"GR00T ARENA SMOKE: {passed}/{total} PASS")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
