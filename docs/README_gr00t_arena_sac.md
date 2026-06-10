# GR00T N1.6 + Isaac Lab Arena + SAC (verl-vla)

End-to-end guide for training a GR00T N1.6 policy in the Isaac Lab Arena simulator with Soft Actor-Critic (SAC) on **verl-vla**.

## 1. Overview

This recipe fine-tunes the GR00T N1.6 checkpoint `gr1_ranch_bottle_into_fridge` (single Arena task: place a sauce bottle on the fridge top shelf and close the door) with online SAC, using a **scheme-Y** data flow:

- **The env owns observation packing and action decoding.** `IsaacLabArenaEnv` runs `GR00TN16Adapter.build_inputs` to produce eagle tensors (`images / lang_tokens / lang_masks / states`) under `obs.images_and_states`, and decodes the policy's **normalised** action chunk back to 26-DOF joints with `decode_actions_flat` — **once per chunk against a single fixed base state** captured at chunk entry (the checkpoint uses `use_relative_action=true`, so a fixed base is correctness-critical).
- **The rollout is minimal.** `GR00TRolloutRob.generate_sequences` only samples the model and emits the normalised action chunk + `critic_value` + `log_probs`; it does **not** pack obs or decode actions.
- **SAC training** consumes `t0.obs.*` / `t0.action.action` from the replay buffer; the actor loss optionally adds a fixed-coefficient BC anchor (`bc_loss_coef`).

Entry module: `python -m verl_vla.trainer.main_sac` (driven by `examples/arena_sac/run_gr00t_arena_sac.sh`).

## 2. Environment preparation

```bash
# Target repo + branch
git clone https://github.com/rebeccazhang0707/verl-vla
cd verl-vla
git checkout migrate/gr00t-arena-sac  # package: src/verl_vla/, prefix verl_vla.*
```

**Checkpoint** (`action_horizon=50`, `use_relative_action=true`, `embodiment_id=20`):

```
_isaaclab_arena/checkpoints/gr1_ranch_bottle_into_fridge/checkpoint-5000-export/
```

Make it visible to the container (see §3); the run scripts point `SFT_MODEL_PATH` at the mounted path (default `/models/checkpoint-5000-export`).

**USD assets:** Arena scene/object USDs are resolved by the Isaac Lab Arena stack inside the image (Omniverse Nucleus / local asset root). Follow the `_isaaclab_arena` asset setup; for the LIBERO-style flow the analogous helper is `scripts/download_usd_assets.sh`.

## 3. Docker build & launch

The image is built/launched by the **Arena** helper (not the libero `Dockerfile.isaaclab402`):

```bash
git clone -b release/0.2.0 git@github.com:isaac-sim/IsaacLab-Arena.git _isaaclab_arena
cd _isaaclab_arena/docker
# -g installs GR00T N1.6 deps and tags the image cuda_gr00t_gn16
./run_docker.sh -g
# Image: isaaclab_arena:cuda_gr00t_gn16   Workdir: /workspaces/isaaclab_arena
# Mounts: $HOME/datasets→/datasets, $HOME/models→/models, $HOME/eval→/eval, /tmp→/tmp
# Python interpreter inside the image: /isaac-sim/python.sh
```

**Make verl-vla and the checkpoint reachable inside the container**, e.g.:

```bash
# Put the checkpoint under the auto-mounted models dir
cp -r _isaaclab_arena/checkpoints/gr1_ranch_bottle_into_fridge/checkpoint-5000-export \
  $HOME/models/checkpoint-5000-export   # → /models/checkpoint-5000-export

# Make verl-vla visible: either add a bind mount to run_docker.sh DOCKER_RUN_ARGS
#   -v <host>/verl-vla:/code/verl-vla
# or clone verl-vla under an already-mounted host dir (e.g. $HOME/models → /models).
```

**Mandatory `PYTHONPATH` (transformers 4.51.3 must win):**

```bash
export PYTHONPATH=/opt/groot_deps:/code/verl-vla/src
```

`/opt/groot_deps` holds the pinned gr00t + transformers-4.51.3 stack required by the eagle remote code; it **must precede** the verl-vla src so the patched transformers takes priority.

## 4. Data preparation

The Arena task is single-task, so the dataset is a constant-`task_ids=0` placeholder with dummy `state_ids`:

```bash
cd /code/verl-vla
python scripts/prepare_arena_dataset.py --num_train <BATCH_SIZE> --num_val <n>
```

Schema (same columns as the libero dataset):

| Column | Value |
|---|---|
| `data_source` | arena dataset tag |
| `prompt` | the Arena task description |
| `state_ids` | `0..N-1` (dummy; reset placement is env-driven) |
| `task_ids` | constant `0` (single task) |
| `ability` / `extra_info` | placeholders consumed by the trainer |

`<BATCH_SIZE>` must match the run script (see §5): single-node default `BATCH_SIZE = NUM_ENV_WORKERS·NUM_STAGE·NUM_ENV / ROLLOUT_N = 4·2·8/8 = 8`.

## 5. Training

```bash
cd /code/verl-vla
export PYTHONPATH=/opt/groot_deps:/code/verl-vla/src

# Single-node
bash examples/arena_sac/run_gr00t_arena_sac.sh

# Disaggregated (separate sim / train GPU pools; start Ray head+worker first, see script header)
bash examples/arena_sac/run_gr00t_arena_sac_disagg.sh
```

Key tunables (anchors as shipped):

| Knob | Default | Notes |
|---|---|---|
| `NUM_ACTION_CHUNKS` | `16` | single anchor → feeds env + model + `critic_action_horizon`; invariant `critic_action_horizon ≤ NUM_ACTION_CHUNKS ≤ action_horizon(50)` |
| `+override_config.action_horizon` | `50` | matches checkpoint `config.json`; guard upper bound |
| `NUM_ENV` | `8` | envs per env worker |
| `NUM_ENV_WORKERS` / `NUM_ENV_GPUS` | `4` | sim GPU workers (single / disagg) |
| `NUM_STAGE` | `2` | pipeline stages |
| `ROLLOUT_N` | `8` | responses per prompt (== `NUM_ENV` for Isaac) |
| `actor.sac.bc_loss_coef` | `0.0` | fixed-coefficient BC anchor; **arena now syncs source `09b0f07` → `0.0` = BC anchor OFF (pure SAC)**. Mechanism retained; `0.0` ⇒ `model.bc_loss` not called |
| `actor.sac.tau` | `1.0` | target-network coeff; source `09b0f07` (was 0.005). `1.0` = hard target copy every update |
| `actor.sac.initial_alpha` | `0.01` | fixed entropy temperature; source `09b0f07` (was 0.05) |
| `actor.sac.alpha_type` | `exp` | alpha parameterisation; source `09b0f07` (was softplus). Target supports `exp`/`softplus` |
| `actor.sac.actor_replay_positive_sample_ratio` | `0.8` | actor replay positive ratio; source `09b0f07` (was 0.5). Critic ratio stays `0.5` |
| `actor.replay_pool_single_size` | `2000` | rank-local replay capacity; source `09b0f07` (was 6000). Shallower ≈ near on-policy |
| `actor.replay_pool_save_interval` | `500` | replay snapshot interval; source `09b0f07` |
| `env.train.step_penalty` | `0.001` | per-step time penalty (source `09b0f07`; was `0.0`) — dense critic signal, à la LIBERO |
| `+env.train.env_spacing` | `10.0` | source recipe |
| `+override_config.freeze_vision_tower` | `True` | freeze Eagle vision tower + `mlp1` connector (source `09b0f07`; model default `False` = legacy) |
| `+override_config.critic_pooling` | `attn` | learnable cross-attention pool over VL tokens (source `09b0f07`; model default `mean` = legacy masked mean-pool). Output dim unchanged ⇒ `critic_input_dim` same as mean-pool |
| `+override_config.critic_use_encoded_state` | `True` | feed the actor's dense `state_encoder` output to the critic instead of the raw padded state (source `09b0f07`; model default `False`). **Changes `critic_input_dim`** ⇒ critic must train fresh (incompatible with a mean/raw critic ckpt) |
| `override_config.critic_prefix_attn_heads` | `8` | MHA heads for the cross-attention pool (only used when `critic_pooling=attn`) |
| `MICRO_BATCH_SIZE` (disagg only) | `32` | source `09b0f07` bumped disagg `16→32`; single-node stays `16` |
| `FLOW_SDE_NOISE_LEVEL` | `0.02` | source `09b0f07` enabled disagg exploration noise `0.0→0.02`; single-node also `0.02` (migration leftover `0.065` corrected to match source single-node) |
| `trainer.save_freq` | `500` | |
| `trainer.total_epochs` | `1000` | |

> **Source `09b0f07` sync.** The four critic/vision knobs are **config-gated** with a legacy default
> in the model (`mean` pool / raw state / no vision freeze), so existing `pi05` / `libero` and the
> prior gr00t path are bit-unchanged; the Arena recipe opts in via the run-script `override_config`
> overrides. The seven SAC tunables (`tau` / `bc_loss_coef` / `initial_alpha` / `alpha_type` /
> `actor_replay_positive_sample_ratio` / `replay_pool_single_size` / `replay_pool_save_interval`) are
> per-run `actor.*` overrides (declared dataclass fields → no `+`) and only affect the Arena run
> scripts, not the shared defaults. See `docs/MIGRATION_gr00t_arena.md §12` (§12.5 for the SAC bundle).

## 6. Mandatory gates before the first formal training run

> These are **docker-only** (real verl + Isaac Sim + gr00t + GPU). None run on a CPU dev host. Full detail: `docs/MIGRATION_gr00t_arena.md §11.C`.

- **Gate 0 — real compose.** `python -m verl_vla.trainer.main_sac --cfg job` with the run-script overrides; replaces the CPU proxy. (Also resolves the `actor.grad_clip` `+`-prefix question.)
- **Gate 1 — #1 decode numerical equivalence (HARD GATE).** With the real checkpoint + real `GR00TN16Adapter`: assert the env `chunk_step` decoded chunk equals one-shot `decode_actions_flat(full_chunk, base_groups)` element-wise (`atol=1e-5`); then perturb `env._last_policy_state` **after** chunk-start capture and assert decoded values are **unchanged** (proves fixed base — the §8.J P0 fix).
- **Gate 2 — end-to-end.** 1–2 `rollout_interval`s; assert replay carries `t0.obs.images`, `t0.action.action` (normalised), `feedback.terminations`, correctly-shaped `info.*`.
- **Gate 3 — FSDP2 forward.** `sac_sample_actions` / `sac_get_critic_value` / `bc_loss` run under FSDP2 without DTensor errors; `wrap_policy.transformer_layer_cls_to_wrap` covers all action-head submodules.
- **Gate 4 — smoke.** `smoke_test_gr00t_arena.py --num-envs 2 --denoise-steps 2` → `8/8 PASS`; then `eval_arena_gr00t.py --reset-only`; then a short `--actor sac` eval.

## 7. Evaluation

```bash
PYTHONPATH=/opt/groot_deps:/code/verl-vla/src /isaac-sim/python.sh \
  examples/arena_sac/eval_arena_gr00t.py \
  --ckpt /models/checkpoint-5000-export --num-envs 2 --episodes 2 \
  --max-steps 100 --chunk 16 --actor sac
```

- `--actor sac` — the **exact** verl-vla SAC rollout action path: model emits a normalised chunk, the env decodes it once with a fixed base (`chunk_step`). Use this to evaluate the trained policy.
- `--actor gr00tpolicy` — official `Gr00tPolicy` reference; outputs **absolute** joints, stepped per-step via `env.step`. Use this as the upstream baseline.
- `--reset-only` — scene/asset smoke (stops after first reset).

## 8. Differences vs source recipe

| Item | Source | Target (verl-vla) |
|---|---|---|
| BC anchor (`bc_loss_coef`) | `0.0` (source `09b0f07`) | run scripts `0.0` — **arena now closes the fixed-coefficient BC anchor (pure SAC)**, synced to source (was `0.05`). Mechanism retained (field default `0.0` + 3-branch `training_worker` logic) **+** optional TD3+BC (`td3_enabled`); mutually exclusive |
| SAC tunables (`tau` / `initial_alpha` / `alpha_type` / `actor_replay_positive_sample_ratio` / `replay_pool_single_size` / `replay_pool_save_interval`) | `1.0` / `0.01` / `exp` / `0.8` / `2000` / `500` (source `09b0f07`) | same (run-script `actor.*` overrides); `critic_replay_positive_sample_ratio` stays `0.5` |
| `step_penalty` | `0.001` (source `09b0f07`) | `0.001` (run scripts) — corrected from the earlier `0.0` recorded against the pre-`09b0f07` source |
| vision tower | frozen (source `09b0f07`, default `True`) | `freeze_vision_tower` — model default `False` (legacy: follows backbone `tune_visual`); run scripts set `True` |
| critic pooling | `attn` cross-attention pool (source `09b0f07`) | `critic_pooling` — model default `mean` (legacy masked mean-pool); run scripts set `attn` |
| critic state input | encoded `state_features` (source `09b0f07`) | `critic_use_encoded_state` — model default `False` (raw padded state); run scripts set `True` |
| disagg micro-batch / noise | `32` / `0.02` (source `09b0f07`) | `32` / `0.02` (disagg script); single-node `16` / `0.02` (FLOW_SDE drift `0.065→0.02` corrected) |
| `env_spacing` | `10.0` | `10.0` (run scripts) |
| `max_episode_steps` | run script `512` (yaml 1200) | `512` |
| `pipeline_stage_num` | run script `2` (yaml 1) | `2` |
| action horizon guard | ckpt `50` | `+override_config.action_horizon=50` |
| Data flow | rollout packs/decodes | **scheme Y**: env packs obs + decodes chunk (fixed base); rollout emits normalised actions |

## 9. Troubleshooting

- **`action_horizon` guard fires at rollout init** — ensure `critic_action_horizon ≤ NUM_ACTION_CHUNKS ≤ action_horizon`. The guard upper bound = checkpoint `config.json action_horizon=50`; keep `+override_config.action_horizon=50` set.
- **Action drift / corrupted trajectories** — the checkpoint is relative-action; the env must decode the whole chunk against the **chunk-start** base (`_decode_chunk_to_policy_actions`), not live state. Verify via Gate 1.
- **FSDP wrap misses the action head** — `wrap_policy.transformer_layer_cls_to_wrap` must list `Qwen3DecoderLayer`, `Siglip2EncoderLayer`, `BasicTransformerBlock`, `MultiEmbodimentActionEncoder`, `CategorySpecificMLP` (already in both run scripts).
- **Eagle remote code fails to load** — keep `PYTHONPATH=/opt/groot_deps:…` (transformers 4.51.3 first); the `_patch_eagle_compat` shim (`_attn_implementation_autoset` + forced FA2) is applied automatically by eval/smoke.
- **Critic Q-values collapse (positive and negative states get the same Q)** — this is a critic *representation* problem, **not** a reward problem. The param-free masked mean-pool averages task-relevant tokens away and the raw padded state is mostly zeros for GR1. Fix by enabling the learnable cross-attention pool + dense encoded state (source `09b0f07`): `+override_config.critic_pooling=attn`, `+override_config.critic_use_encoded_state=True` (both already set in the Arena run scripts). NB: `critic_use_encoded_state` changes `critic_input_dim`, so the critic must train fresh.
- **Critic outputs NaN with `critic_pooling=attn`** — padded VL tokens can carry `NaN`/`inf`; the cross-attention pool zeros padding positions (`vl_safe = where(mask, vl_embeds, 0)`) **before** the MHA call, because `key_padding_mask` only nulls the softmax weight while `nn.MultiheadAttention` still projects the masked values (`0 * NaN = NaN`). If NaNs still appear, set `GR00T_SAC_NAN_DEBUG=1` to log which tensor (`pooled`/`state`/`action`) is non-finite and the pooling/encoded-state/target context. (Default mean-pool path is unaffected.)
- **cuDNN SDPA crash on Hopper** — `_disable_cudnn_sdpa()` is applied by the Phase-2 model; do not re-enable cuDNN SDPA.
- **`pi05`/`libero` unaffected** — `bc_loss_coef` defaults to `0.0` → BC branch skipped, `model.bc_loss` never called.
