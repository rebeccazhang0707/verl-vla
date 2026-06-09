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

"""Closed-loop GR00T N1.6 policy eval in the Arena simulator (NO server/client).

Migrated from ``verl.experimental.vla.eval_arena_gr00t`` to ``verl_vla.*`` and
aligned with the verl-vla **scheme-Y** Arena env (env owns obs packing + action
decoding; see ``verl_vla.envs.arena_env.arena_env``). Drives the policy through
``IsaacLabArenaEnv`` directly, in-process:

    env.reset() ─► obs(images_and_states[packed] + full_image + task)
        loop:
          sac:        model.sac_sample_actions(packed obs) → NORMALISED chunk
                      → env.chunk_step(chunk)              (env decodes once, fixed base)
          gr00tpolicy: Gr00tPolicy.get_action → ABSOLUTE joints → env.step(per-step)
        track success_once / return per episode

The ``--actor sac`` path is byte-for-byte the verl-vla SAC rollout action path
(``GR00TRolloutRob.generate_sequences``): emit the normalised model chunk and let
the env decode it (``chunk_step`` → ``decode_actions_flat`` → ``scatter_action``).
The ``--actor gr00tpolicy`` path is the official-policy reference; it outputs
already-decoded ABSOLUTE joints, so it is stepped via the env's per-step
``step`` (which expects decoded 26-DOF joints) rather than ``chunk_step``.

Anchors shared with training (kept here so eval does not drift from the run
scripts): ``embodiment_tag`` / ``num_action_chunks`` / ``action_horizon=50`` /
``override_config`` (sac_enable, action_dim, embodiment_id, critic_head_num).

Run inside the Arena GR00T docker (isaaclab_arena:cuda_gr00t_gn16), GPU + display:

    PYTHONPATH=/opt/groot_deps:/code/verl-vla/src /isaac-sim/python.sh \
        examples/arena_sac/eval_arena_gr00t.py \
        --ckpt /models/checkpoint-5000-export --num-envs 2 --episodes 2 \
        --max-steps 100 --chunk 16

Staged so failures localise: [launch] → [build scene] → [load policy] → [rollout].
This module is CPU-importable: every isaac / omni / transformers / gr00t import is
deferred into the function that needs it.
"""

import argparse
import sys
import traceback

import numpy as np
import torch

# Model action horizon for this checkpoint (config.json action_horizon=50). Kept as
# a module constant so the guard upper bound matches the run-script anchor
# (+actor_rollout_ref.model.override_config.action_horizon=50) and eval cannot drift.
CKPT_ACTION_HORIZON = 50


def _patch_transformers_eagle():
    """Compat shims so the Eagle remote code loads.

    Delegates to the Phase-2 migrated patch (``modeling_gr00t_sac._patch_eagle_compat``):
    PretrainedConfig._attn_implementation_autoset shim + flash_attention_2 injection
    into gr00t's eagle_backbone. Falls back to ``isaaclab_arena_gr00t``'s compat helper
    when available (preferred inside the Arena image).
    """
    try:
        from isaaclab_arena_gr00t.utils.eagle_config_compat import apply_eagle_config_compat

        apply_eagle_config_compat()
        return
    except Exception:
        pass
    from verl_vla.models.gr00t.modeling_gr00t_sac import _patch_eagle_compat

    _patch_eagle_compat()


def build_parser() -> argparse.ArgumentParser:
    """Build the eval argparse parser (split out so it is unit-testable on CPU)."""
    ap = argparse.ArgumentParser(description="Closed-loop GR00T N1.6 Arena eval (in-process)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embodiment-tag", default="gr1")
    ap.add_argument("--arena-env-name", default="put_item_in_fridge_and_close_door")
    ap.add_argument("--arena-object", default="ranch_dressing_hope_robolab")
    ap.add_argument(
        "--arena-embodiment",
        default="gr1_joint",
        help="env control embodiment; MUST be a joint-position one (gr1_joint) for GR00T",
    )
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=16, help="action chunk executed per policy query (num_action_chunks)")
    ap.add_argument(
        "--action-horizon",
        type=int,
        default=CKPT_ACTION_HORIZON,
        help="model action horizon (checkpoint config.json = 50); guard upper bound",
    )
    ap.add_argument("--critic-heads", type=int, default=10)
    ap.add_argument(
        "--actor",
        choices=["gr00tpolicy", "sac"],
        default="gr00tpolicy",
        help="gr00tpolicy = official Gr00tPolicy (eval reference); "
        "sac = Gr00tN1d6ForSAC.sac_sample_actions (the verl-vla SAC training action path)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env-spacing", type=float, default=10.0, help="match training (source recipe = 10.0)")
    ap.add_argument("--kitchen-style", type=int, default=2)
    ap.add_argument("--camera-name", default="robot_pov_cam_rgb")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-dir", default="/tmp/eval_arena_video")
    ap.add_argument("--reset-only", action="store_true", help="stop after first reset (scene/asset smoke)")
    return ap


def _build_cfg(args):
    """Build the OmegaConf cfg consumed by ``IsaacLabArenaEnv.__init__`` (target schema)."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "seed": args.seed,
            "num_envs": args.num_envs,
            "device": args.device,
            "action_dim": 26,
            "camera_name": args.camera_name,
            "arena_env_name": args.arena_env_name,
            "arena_object": args.arena_object,
            # GR00T outputs JOINT positions -> the env must use the joint-control
            # embodiment ("gr1_joint"). "gr1_pink" is Pink-IK (end-effector) control.
            "arena_embodiment": args.arena_embodiment,
            "object_set": None,
            "kitchen_style": args.kitchen_style,
            "task_description": "Place the sauce bottle on the top shelf of the fridge, and close the fridge door.",
            "max_episode_steps": args.max_steps,
            # Anchor: match the run-script env_spacing (source recipe = 10.0).
            "env_spacing": args.env_spacing,
            "disable_fabric": False,
            "solve_relations": True,
            "enable_pinocchio": True,
            "placement_seed": None,
            "resolve_on_reset": None,
            "presets": None,
            # The target env builds GR00TN16Adapter in __init__ from these.
            "gr00t_model_path": args.ckpt,
            "embodiment_tag": args.embodiment_tag,
            "render_on_chunk_boundary": not bool(args.video),
            "video_cfg": {"save_video": bool(args.video), "video_base_dir": args.video_dir, "fps": 50},
        }
    )


def _full_image_and_tasks(obs, env):
    """Pull (full_image[B,H,W,C] uint8, state26[B,26] float32, tasks[list]) for the policy path.

    Scheme-Y obs keep ``full_image`` at the top level (video only) and pack the
    model-ready eagle tensors under ``images_and_states``. The raw 26-DOF joint state
    fed to the last ``build_inputs`` is cached on the env as ``_last_state26``.
    """
    full_image = obs["full_image"]
    if isinstance(full_image, torch.Tensor):
        full_image = full_image.cpu().numpy()
    state26 = np.asarray(env._last_state26, dtype=np.float32)
    tasks = obs.get(
        "task_descriptions",
        ["Place the sauce bottle on the top shelf of the fridge, and close the fridge door."],
    )
    return full_image, state26, list(tasks)


def main() -> int:
    args = build_parser().parse_args()

    # Guard (#2): critic_action_horizon <= num_action_chunks <= action_horizon, so a too-small
    # chunk does not silently truncate the critic input and a too-large one does not exceed the
    # model decode horizon. Reuses the rollout's pure helper (CPU-importable).
    from verl_vla.workers.rollout.naive_rollout_gr00t import assert_action_horizon_invariant

    assert_action_horizon_invariant(
        num_action_chunks=args.chunk,
        critic_action_horizon=args.chunk,  # run scripts set critic_action_horizon == num_action_chunks
        action_horizon=args.action_horizon,
    )

    device = args.device
    dtype = torch.bfloat16

    # ---- [launch] construct env (launches Isaac Sim) ----
    print("[launch] constructing IsaacLabArenaEnv (launches Isaac Sim)...", flush=True)
    from verl_vla.envs.arena_env.arena_env import IsaacLabArenaEnv

    cfg = _build_cfg(args)
    env = IsaacLabArenaEnv(cfg, rank=0, world_size=1)
    print("[launch] Isaac Sim app up.", flush=True)

    # ---- [build scene] build the Arena env + first reset ----
    print(f"[build scene] _init_env(arena_env_name={args.arena_env_name}, object={args.arena_object})...", flush=True)
    env._init_env()
    obs, _ = env.reset()
    full_image, state, tasks = _full_image_and_tasks(obs, env)
    print(f"[build scene] reset OK. image={full_image.shape} state={state.shape} task={tasks[0][:60]!r}", flush=True)
    if args.reset_only:
        env.close()
        print("[reset-only] scene + reset succeeded; stopping.", flush=True)
        return 0

    if args.actor == "gr00tpolicy":
        # Official gr00t.policy.Gr00tPolicy — the eval reference. Outputs ABSOLUTE
        # joints (decode_action applied internally), so it is stepped per-step via
        # env.step (which expects already-decoded 26-DOF joints), NOT chunk_step
        # (which decodes a NORMALISED chunk).
        print("[load policy] loading official Gr00tPolicy...", flush=True)
        from gr00t.data.embodiment_tags import EmbodimentTag

        _patch_transformers_eagle()
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        from verl_vla.models.gr00t.utils import GR1_STATE_GROUP_DIMS

        emb = EmbodimentTag[args.embodiment_tag.upper()]
        policy = Gr00tPolicy(embodiment_tag=emb, model_path=args.ckpt, device=device, strict=True)
        _video_key = policy.modality_configs["video"].modality_keys[0]
        _state_keys = list(policy.modality_configs["state"].modality_keys)
        _lang_key = policy.language_key
        print(f"[load policy] ready. video={_video_key} state={_state_keys} lang={_lang_key}", flush=True)

        def _policy_obs(full_image, state, tasks):
            B, H, W, C = full_image.shape
            video = {_video_key: full_image.reshape(B, 1, H, W, C).astype(np.uint8)}
            st = {}
            start = 0
            for k in _state_keys:
                d = GR1_STATE_GROUP_DIMS[k]
                st[k] = state[:, start : start + d].reshape(B, 1, d).astype(np.float32)
                start += d
            lang = {_lang_key: [[tasks[i] if i < len(tasks) else tasks[-1]] for i in range(B)]}
            return {"video": video, "state": st, "language": lang}

        def predict_abs_chunk(full_image, state, tasks):
            action_dict, _ = policy.get_action(_policy_obs(full_image, state, tasks))
            return np.concatenate([action_dict[k] for k in _state_keys], axis=-1)  # (B, horizon, 26) ABSOLUTE

        is_normalised = False

    else:  # --actor sac : the EXACT verl-vla SAC rollout action path (Gr00tN1d6ForSAC.sac_sample_actions)
        print("[load policy] loading Gr00tN1d6ForSAC (training action path)...", flush=True)
        from transformers import AutoConfig, AutoModel
        from verl import DataProto

        from verl_vla.models.gr00t.modeling_gr00t_sac import register_gr00t_sac
        from verl_vla.models.gr00t.utils import load_embodiment_id

        _patch_transformers_eagle()
        register_gr00t_sac()
        scfg = AutoConfig.from_pretrained(args.ckpt)
        scfg.sac_enable = True
        scfg.action_dim = 26
        # Resolve from the checkpoint's embodiment_id.json (fallback table otherwise).
        scfg.embodiment_id = load_embodiment_id(args.embodiment_tag, args.ckpt)
        scfg.critic_head_num = args.critic_heads
        model = AutoModel.from_pretrained(args.ckpt, config=scfg).eval().to(device=device, dtype=dtype)
        print("[load policy] SAC model ready (env owns obs packing + action decoding).", flush=True)

        def predict_norm_chunk(obs):
            # Scheme Y: the env already packed the eagle tensors under ``images_and_states``
            # (exactly the slots ``sac_sample_actions`` reads). Mirror GR00TRolloutRob.
            packed = obs["images_and_states"]
            prompts = DataProto.from_dict(tensors={k: v.to(device) for k, v in packed.items()})
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                out = model.sac_sample_actions(prompts, validate=True)
            return out["action"].detach().float().cpu().numpy()  # (B, horizon, max_action_dim) NORMALISED

        is_normalised = True

    # ---- [rollout] closed-loop episodes ----
    successes, returns, ep_lens = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        steps = 0
        while steps < args.max_steps:
            if is_normalised:
                # NORMALISED chunk → env decodes once with a fixed base (chunk_step).
                full_chunk = predict_norm_chunk(obs)
                chunk = torch.from_numpy(np.asarray(full_chunk[:, : args.chunk], dtype=np.float32))
                obs, rew, term, trunc, infos = env.chunk_step(chunk)
                steps += chunk.shape[1]
                done = bool(torch.as_tensor(term).reshape(-1).all()) or bool(torch.as_tensor(trunc).reshape(-1).all())
            else:
                # ABSOLUTE joints → step per-step (env.step expects decoded 26-DOF).
                full_image, state, tasks = _full_image_and_tasks(obs, env)
                abs_chunk = predict_abs_chunk(full_image, state, tasks)  # (B, horizon, 26)
                done = False
                for i in range(min(args.chunk, abs_chunk.shape[1])):
                    step_act = torch.from_numpy(np.asarray(abs_chunk[:, i], dtype=np.float32))
                    obs, rew, term, trunc, infos = env.step(step_act)
                    steps += 1
                    done = bool(torch.as_tensor(term).reshape(-1).all()) or bool(
                        torch.as_tensor(trunc).reshape(-1).all()
                    )
                    if done or steps >= args.max_steps:
                        break
            if done:
                break
        succ = env.success_once.copy()
        ret = env.returns.copy()
        successes.append(succ)
        returns.append(ret)
        ep_lens.append(int(env.elapsed_steps.max()))
        print(
            f"[rollout] ep {ep}: success={succ.tolist()} return={np.round(ret, 3).tolist()} steps={steps}",
            flush=True,
        )
        if args.video:
            env.flush_video(video_sub_dir=f"ep{ep}")

    succ_all = np.concatenate(successes)
    print("\n" + "=" * 60)
    print(f"ARENA EVAL: {args.episodes} episodes × {args.num_envs} envs = {succ_all.size} rollouts")
    print(f"  success_rate = {succ_all.mean():.3f}  ({int(succ_all.sum())}/{succ_all.size})")
    print(f"  mean_return  = {np.concatenate(returns).mean():.4f}")
    print(f"  mean_ep_len  = {np.mean(ep_lens):.1f}")
    print("=" * 60)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
