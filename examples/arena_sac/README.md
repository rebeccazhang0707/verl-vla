# GR00T N1.6 + Isaac Lab Arena — SAC Training (verl-vla)

Online **SAC** fine-tuning of the **GR00T N1.6** policy in **Isaac Lab Arena** (task
`put_item_in_fridge_and_close_door`), on the refactored **verl-vla** package
(`src/verl_vla`). Two physical layouts:

| Layout | Script | Use when |
| --- | --- | --- |
| **Single node (collocated)** | `examples/arena_sac/run_gr00t_arena_sac.sh` | one 8-GPU **L20** box (4 sim + 4 train) |
| **Disaggregated** | `examples/arena_sac/run_gr00t_arena_sac_disagg.sh` | **L20** sim node(s) + **H20** train node(s) |

Entry point for both: `python -m verl_vla.trainer.main_sac` with Hydra overrides
(`--config-name rob_sac_trainer`, env group `env=rob_sac_env`).

---

## 0. Get the code + the Arena submodule

The Arena task code (including the `critic_privileged` observation group used by the
asymmetric actor-critic) lives in the **`IsaacLab-Arena` git submodule**, pinned to
`git@github.com:rebeccazhang0707/IsaacLab-Arena.git` branch `reb/arena-verl`. It is **not**
fetched by a plain `git clone` — you must init the submodule.

```bash
# Fresh clone WITH the submodule in one shot (LFS skipped to avoid large USD assets):
GIT_LFS_SKIP_SMUDGE=1 git clone -b migrate/gr00t-arena-sac --recurse-submodules https://github.com/rebeccazhang0707/verl-vla.git
cd verl-vla

# OR, if you already cloned without --recurse-submodules:
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive IsaacLab-Arena
```

Verify the submodule is on the pinned commit:

```bash
git submodule status IsaacLab-Arena
# expect:  3500d454... IsaacLab-Arena (heads/reb/arena-verl)
```

Notes:
- The submodule URL is an **SSH** remote on a private fork — you need an SSH key with read
  access (`ssh -T git@github.com` should authenticate). For HTTPS, override the URL:
  `git submodule set-url IsaacLab-Arena https://github.com/rebeccazhang0707/IsaacLab-Arena.git`.
- `GIT_LFS_SKIP_SMUDGE=1` skips downloading LFS-tracked assets at clone time; the asset cache
  is populated lazily on first sim run from the lightwheel registry.
- To bump the pin later: `cd IsaacLab-Arena && git fetch && git checkout <commit> && cd .. &&
  git add IsaacLab-Arena && git commit`.

## Prerequisites

- **Docker image**: `isaaclab_arena:cuda_gr00t_gn16` (built from the IsaacLab-Arena repo,
  `IsaacLab-Arena/docker/run_docker.sh -g`). Bundles GR00T deps under `/opt/groot_deps`.
- **Checkpoint**: a GR00T N1.6 export dir (e.g. `checkpoint-5000-export`), mounted into the
  container (typically `/models/checkpoint-5000-export`).
- **verl-vla repo**: this tree, mounted into the container (e.g. `/eval`); launch scripts at
  `/eval/examples/arena_sac/...`.
- **CUDA forward-compat** (only if host driver < image CUDA 12.8): host `~/cuda128-compat`
  bind-mounted to `/opt/cuda128-compat`.

---

## Hardware: why L20 *and* H20

| Role | GPU | Reason |
| --- | --- | --- |
| **Sim** (Isaac Sim env workers) | **L20 / L40 (Ada)** | Isaac Sim RTX rendering **requires RT Cores**. Hopper (H20/H100) has **none** → cannot render. |
| **Train** (FSDP actor+critic + GR00T inference) | **L20 *or* H20** | Pure compute, no rendering. H20's 96 GB suits the policy update. |

A **single-node** run must be on **L20** (it renders the sim). To train on **H20**, use the
**disaggregated** layout (L20 sim node + H20 train node).

---

## 1. Start the container (from the IsaacLab-Arena build context)

```bash
cd IsaacLab-Arena
bash docker/run_docker.sh -g -r \
  -m <repo>/checkpoints/gr1_ranch_bottle_into_fridge \
  -e <repo>     # the verl-vla repo root -> /eval
```

`-m` → checkpoint parent dir → `/models`; `-e` → verl-vla repo → `/eval`. Pass `-r` only on
the first run (force-builds the image). Export `WANDB_API_KEY` on the host or set
`trainer.logger=['console']` to skip wandb.

---

## 2. Single node — 8× L20 (collocated)

The script auto-starts a local Ray cluster and injects the custom resources itself:

```bash
bash /eval/examples/arena_sac/run_gr00t_arena_sac.sh
```

Default topology: 4 sim GPUs + 4 train GPUs, 2 pipeline stages. Change the split via env vars,
e.g. `NUM_ENV_WORKERS=2 NUM_ROLLOUT_GPUS=6 bash .../run_gr00t_arena_sac.sh`.

Artifacts under `$OUTPUT_DIR` (default `~/models/vla_arena_gr00t_sac`): `video/rank_*/stage_*`,
`replay_pools/`, Hydra run dir.

---

## 3. Disaggregated — L20 sim + H20 train

Start Ray yourself on each node (one resource label per node), then launch on the train/head
node only.

**Train/head node (H20):**
```bash
PYTHONPATH=/opt/groot_deps:/eval /isaac-sim/python.sh -m ray.scripts.scripts start \
    --head --port=6379 --dashboard-host=0.0.0.0 --num-gpus=8 --num-cpus=48 \
    --resources='{"train_rollout": 8}'
```

**Each sim node (L20):**
```bash
PYTHONPATH=/opt/groot_deps:/eval /isaac-sim/python.sh -m ray.scripts.scripts start \
    --address='<head-ip>:6379' --num-gpus=8 --num-cpus=48 --resources='{"sim": 8}'
```

**Verify** both pools (`ray.cluster_resources()` shows `train_rollout` and `sim`), then:
```bash
bash /eval/examples/arena_sac/run_gr00t_arena_sac_disagg.sh
```

`env.disagg_sim.enable=True` means *separate GPU pools*, not separate nodes.

---

## 4. Configuration

### Reward, success & auto-reset (Arena RL adaptation)

The stock Arena `put_item_in_fridge_and_close_door` task defines **no reward term** — it only
exposes composite-success as a *termination*. That breaks online RL: verl derives
`done`/`success` from `reward > 0` (so a success would be invisible), and IsaacLab would
**auto-reset** on success (corrupting the fixed-length rollout).

`apply_rl_reward_and_disable_autoreset` (in `src/verl_vla/envs/arena_env/utils.py`, run at env
build, **RL-only**) fixes both, LIBERO-style:
- **success termination → `RewTerm`** with `weight = 1/step_dt` (Arena 50 Hz → +1.0/step on
  success). Toggle off with `env.train.rl_success_reward=False`.
- **all terminations → None** → no auto-reset; verl owns resets + horizon.

Healthy log: `[arena_env] RL patch: success->RewTerm weight=50.000 (step_dt=0.0200s); ...`.

> **`MAX_EPISODE_STEPS` must be a multiple of `num_action_chunks`** — `env_loop` floors
> `max_interactions = max_episode_steps // num_action_chunks`. With `num_action_chunks=16` use
> **512** (= 32×16 ≈ 10 s @ 50 Hz).

### Long-horizon credit + asymmetric critic (gated, arena-only)

| `env.train.…` | Default | Effect |
| --- | --- | --- |
| `subtask_reward` | `False` | graded subtask reward (0/0.5/1.0 = fraction of subtasks done) for earlier credit on the sequential task; supersedes `dense_success_reward`. Trainer emits `data/sr_subtask{k}` / `data/sr_composite`. |
| `dense_success_reward` | `False` | keep post-success steps valid (many +1 anchors); -1 timeout only on never-successful trajectories. |
| `num_subtasks` | `null` | subtask count for the `sr_subtask{k}` metrics; `null` = inferred from graded levels. |
| `critic_privileged_obs` | `False` | feed the task's `critic_privileged` obs group (object pose rel. to shelf + door joint) to the **critic only** (asymmetric AC). |

> **Privileged-obs dim must match between env and model.** The env resolves `PRIV_OBS_DIM`
> **dynamically** from the live `ObservationManager` (`_resolve_priv_obs_dim`), so it tracks
> whatever the Arena task declares. The **model** side is static — set
> `+actor_rollout_ref.model.override_config.critic_privileged_obs(_dim)` in the run script to
> the **same** width, or the critic MLP `critic_input_dim` won't match the concatenated input.
> The pinned submodule's `critic_privileged` group is **8-dim** (object pose 7 + door 1); the
> run scripts currently set `critic_privileged_obs_dim=4` — align these to the Arena group you
> actually run against.

### Replay pool / fresh start

`actor_rollout_ref.actor.load_replay_pool=False` + `trainer.resume_mode=disable` start every
arena run from an empty replay pool and skip checkpoint resume (the privileged-obs +
`critic_pooling=attn` settings change `critic_input_dim`, so the critic must train fresh).

### Render performance

`env.train.render_on_chunk_boundary=True` (default) renders the RTX camera **once per action
chunk** instead of every physics step (~32× fewer renders, lossless for the policy). Set
`False` when recording smooth video.

---

## 5. Pitfalls (handled in-repo / by the scripts)

- **Arena task has no reward term** → converted to a `RewTerm` + terminations nulled (§4).
- **`No available node types {'train_rollout': …}`** → disagg pools need Ray resources
  `sim`/`train_rollout`; single-node injects them, multi-node needs `ray start --resources=...`.
- **HDF5 `errno 11 Unable to lock file`** → `build_env_cfg_without_recorder` forces
  `dataset_export_mode=EXPORT_NONE` (metric terms still run in-memory).
- **`SSLCertVerificationError` (lightwheel)** → `disable_lightwheel_ssl_verify` patches
  `requests` to skip TLS verify for lightwheel hosts only.
- **NaN critic with `critic_pooling=attn`** → the cross-attn query token is an `nn.Embedding`
  (not a bare root `nn.Parameter`), so `from_pretrained`'s `_fast_init` initializes it instead
  of leaving NaN; padded VL tokens are zeroed before pooling.

---

## 6. File map

| File | Purpose |
| --- | --- |
| `examples/arena_sac/run_gr00t_arena_sac.sh` | single-node SAC launch (§2) |
| `examples/arena_sac/run_gr00t_arena_sac_disagg.sh` | disaggregated SAC launch (§3) |
| `src/verl_vla/trainer/main_sac.py` | entry point; builds Ray resource pools |
| `src/verl_vla/trainer/sac/sac_ray_trainer.py` | SAC trainer; reward branches (sparse/dense/subtask) + `sr_*` metrics |
| `src/verl_vla/envs/arena_env/arena_env.py` | `IsaacLabArenaEnv` (per-chunk render-interval, privileged-obs extraction, dynamic `PRIV_OBS_DIM`) |
| `src/verl_vla/envs/arena_env/utils.py` | `apply_rl_reward_and_disable_autoreset`, `build_env_cfg_without_recorder`, graded/sparse reward fns, lightwheel SSL patch |
| `src/verl_vla/models/gr00t/modeling_gr00t_sac.py` | `Gr00tN1d6ForSAC`: critic heads, cross-attn pool (`nn.Embedding` query), privileged-obs critic input, frozen vision tower |
| `src/verl_vla/trainer/config/env/rob_sac_env.yaml` | env defaults (`subtask_reward`/`dense_success_reward`/`num_subtasks`/`critic_privileged_obs`, all off by default) |
| `IsaacLab-Arena/` | Arena task code submodule (declares the `critic_privileged` obs group) — see §0 |
