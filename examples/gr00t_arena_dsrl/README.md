# GR00T Arena DSRL (latent-noise steering)

DSRL ([Diffusion Steering via Reinforcement Learning](https://arxiv.org/abs/2506.15799),
RLinf recipe: `libero_spatial_dsrl_openpi.yaml`) keeps the **whole VLA frozen** and
trains only a small SAC policy over the flow-matching **initial noise `x0`**:

```
obs ──frozen backbone──▶ pooled VL features ┐
obs ──processor────────▶ raw state          ┴─▶ noise actor (tanh Gaussian, ~0.5M params)
                                                   │  steering noise x0  (the SAC action)
                                                   ▼
                              frozen flow head, deterministic Euler ODE
                                                   │
                                                   ▼
                                              env action chunk
```

- **Actor**: `verl_vla.models.dsrl.DSRLNoiseActor` — MLP over the frozen pooled
  backbone features + raw state, outputs one tanh-bounded noise vector
  (`max_action_dim`, GR00T: 128) broadcast over the action horizon.
- **Critic**: the existing SAC critic ensemble, scoring the *steering noise*
  (defaults auto-switch to `action_dim=max_action_dim`, `action_horizon=1`).
- **Replay**: `full_action` stores the steering noise; `action` stays the
  decoded env chunk. Trainer / replay pool / env are unchanged.
- **Generic**: the same `adapter.dsrl.*` keys work for pi0/pi05
  (`model/adapter/pi0.yaml`); for pi0 also set `critic.input_dim`
  to `prefix_embed_dim + state_dim + max_action_dim` and `flow_sde_enable=False`.

## Launch

Same Docker / paths as `examples/gr00t_arena_sac` (see its README):

```bash
# GR1 fridge task
ARENA_TASK=gr1 INNER_SCRIPT=examples/gr00t_arena_dsrl/run_gr00t_arena_dsrl.sh \
  OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_dsrl \
  examples/gr00t_arena_sac/run_docker.sh

# Arena LIBERO (Franka)
ARENA_TASK=libero TASK_SUITE=libero_spatial TASK_ID=3 \
  INNER_SCRIPT=examples/gr00t_arena_dsrl/run_gr00t_arena_dsrl.sh \
  OUTPUT_ROOT=/eval/outputs/arena_gr00t_libero_dsrl \
  examples/gr00t_arena_sac/run_docker.sh
```

The single-node command above starts a **local** Ray cluster inside one container
(`--gpus all`) and co-locates the env workers and the model/actor+rollout workers
on the same machine. `NUM_ENV_GPUS` / `NUM_MODEL_GPUS` split the GPUs between the
two pools, but both pools live on that one node.

## Disaggregated launch (H20 train + L20 sim, multi-node Ray)

Isaac Sim env workers are render/physics-bound (fit **L20**), while the GR00T
model/actor+rollout workers are HBM/compute-bound (fit **H20**). `TrainCluster`
already puts the **env pool** and the **model pool** in separate Ray resource
pools; pinning each pool to a node type only needs (1) a Ray cluster that spans
both nodes and (2) a Ray custom-resource label per pool
(`cluster.resource.env.resource_label` / `cluster.resource.model.resource_label`,
both `null` by default). No script change is required — the labels are appended as
Hydra overrides through the inner script's `"$@"` passthrough.

```
 L20 node                              H20 node (Ray head)
 ┌───────────────────────┐             ┌────────────────────────────┐
 │ ray start --address    │            │ ray start --head            │
 │   --resources sim=N    │◀──6379────▶│   --resources train_rollout=M │
 │ Isaac Sim env workers  │            │ GR00T actor+rollout workers  │
 │  (env pool, L20 GPUs)  │            │  (model pool, H20 GPUs)      │
 └───────────────────────┘             │ + DSRL driver (ray.init)     │
                                       └────────────────────────────┘
```

**Prerequisites**

- The `isaaclab_arena:cuda_gr00t_gn16` image is built on **both** nodes (it bakes
  `verl` / `codetiming` that even the L20 `ray start` node imports).
- Repo, checkpoint (`MODELS_HOST`), Arena (`ARENA_HOST`) and — for LIBERO — the
  LIBERO assets (`LIBERO_IN_LAB_HOST`) exist at the **same host paths on both
  nodes**. A shared/NFS mount is simplest and also collects `videos/`,
  `checkpoints/`, `replay_pools/` in one place (otherwise env-side `videos/` land
  on the L20 node and `checkpoints/`+`replay_pools/` on the H20 node).
- The two nodes reach each other over the host network (the container runs with
  `--network=host`).

**Shared setup (run on the machine you drive from)**

```bash
export HEAD_IP=<H20 node IP>          # must be reachable from the L20 node
export RUN_ID="$(id -u):$(id -g)"     # the container runs as your host uid/gid
export CN=gr00t_dsrl_disagg          # shared container name on both nodes
# PYTHONPATH is set per-process by the inner script, but the remote raylet's
# workers inherit it from `ray start`, so pass it there too:
export INNER_PYTHONPATH=/opt/groot_deps:/eval/src:/workspaces/isaaclab_arena
COMMON=(
  ARENA_TASK=libero
  ARENA_HOST=/home/billyw/Projects/libero_rl_example/IsaacLab-Arena
  MODELS_HOST=$HOME/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec
  LIBERO_IN_LAB_HOST=/home/billyw/Projects/RobotLearningLab
  CONTAINER_NAME=$CN
)
```

**1. Start a long-lived container on _both_ nodes** (`--no-run` creates the
`--network=host --privileged` container and the host-matching user, but runs
nothing):

```bash
# run on the H20 node AND on the L20 node
env "${COMMON[@]}" examples/gr00t_arena_sac/run_docker.sh --no-run
```

**2. Start the Ray head on the H20 node** and label its GPUs `train_rollout`
(start Ray through `/isaac-sim/python.sh` so the raylet's workers inherit the
Isaac Sim + GR00T runtime; `--num-gpus` = H20 GPUs on this node):

```bash
docker exec -u "$RUN_ID" -w /eval \
  -e RAY_TMPDIR=/tmp/ray -e PYTHONPATH="$INNER_PYTHONPATH" \
  "$CN" /isaac-sim/python.sh -m ray.scripts.scripts start \
    --head --port=6379 --num-gpus=8 --resources='{"train_rollout": 8}'
```

**3. Join the L20 node to the cluster** and label its GPUs `sim`
(`--num-gpus` = L20 GPUs on this node):

```bash
docker exec -u "$RUN_ID" -w /eval \
  -e RAY_TMPDIR=/tmp/ray -e PYTHONPATH="$INNER_PYTHONPATH" \
  "$CN" /isaac-sim/python.sh -m ray.scripts.scripts start \
    --address="$HEAD_IP:6379" --num-gpus=4 --resources='{"sim": 4}'
```

**4. Launch the DSRL driver on the H20 node.** `RAY_ADDRESS` makes `ray.init()`
attach to the existing cluster instead of starting a local one; the two
`resource_label` overrides pin the env pool to the `sim` (L20) node and the model
pool to the `train_rollout` (H20) node. Keep `NUM_NODES=1` (each pool is one
node); set `NUM_ENV_GPUS` / `NUM_MODEL_GPUS` no larger than the GPUs you gave each
`ray start`:

```bash
docker exec -i -u "$RUN_ID" -w /eval \
  -e RAY_ADDRESS="$HEAD_IP:6379" -e RAY_TMPDIR=/tmp/ray \
  -e GROOT_MODEL_PATH=/models/checkpoint-10000 \
  -e ARENA_TASK=libero -e TASK_SUITE=libero_spatial -e TASK_ID=7 \
  -e LIBERO_IN_LAB_ROOT=/libero_in_lab \
  -e OUTPUT_ROOT=/eval/outputs/arena_gr00t_libero_spatial_task7_dsrl_disagg \
  -e EXPERIMENT_NAME=libero_spatial_task7_gr00t_dsrl_disagg \
  -e TRAINER_LOGGER='[console,wandb]' -e WANDB_API_KEY=<your-key> \
  -e NUM_NODES=1 -e NUM_ENV_GPUS=4 -e NUM_MODEL_GPUS=8 \
  -e MAX_INTERACTIONS=20 -e TOTAL_TRAINING_STEPS=10000 -e TEST_FREQ=500 \
  -e CRITIC_POOL_PROJ_DIM=256 -e CRITIC_LAYERNORM=True \
  -e ACTOR_POSITIVE_SAMPLE_RATIO=0.8 \
  "$CN" bash examples/gr00t_arena_dsrl/run_gr00t_arena_dsrl.sh \
    cluster.resource.env.resource_label=sim \
    cluster.resource.model.resource_label=train_rollout
```

**Teardown** (run on both nodes):

```bash
docker exec -u "$RUN_ID" "$CN" /isaac-sim/python.sh -m ray.scripts.scripts stop
docker rm -f "$CN"
```

**Notes**

- **Resume**: add `-e RESUME_MODE=resume_path` and
  `-e RESUME_FROM_PATH=$OUTPUT_ROOT/checkpoints/global_step_<N>` to the step-4
  driver `docker exec` (keep the same `OUTPUT_ROOT`/`EXPERIMENT_NAME`). The FSDP
  resume restores the noise actor + critic + optimizer/replay state that the HF
  export alone cannot; `RESUME_FROM_PATH` must contain `global_step_`.
- **More sim nodes**: join extra L20 workers with the same
  `--resources='{"sim": N}'` and append `cluster.resource.env.nnodes=<count>` to
  the driver command (step 4). The same pattern with
  `cluster.resource.model.nnodes` scales the H20 side.
- **LIBERO assets on the L20 node**: the inner script injects
  `LIBERO_IN_LAB_ROOT` / `LIBERO_CONFIG_DIR` into the Ray `runtime_env`, so remote
  env workers resolve the USD/HDF5 — but only if those assets are actually mounted
  at `/libero_in_lab` on the L20 node (hence the "same paths on both nodes"
  prerequisite).
- **Label counts are nominal**: each worker bundle only requests `1e-4` of its
  label, so `sim`/`train_rollout` just need to be `≥ 1` on the right nodes; using
  the GPU count keeps the intent readable.
- Everything else (SAC/DSRL knobs below) is unchanged from the single-node path.

## Key knobs (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `NOISE_ACTOR_LR` | `3e-4` | Noise-actor lr (`actor.optim.lr`; the VLA is frozen) |
| `CRITIC_LR` | `3e-4` | SAC critic lr |
| `CRITIC_TAU` | `0.005` | Target critic Polyak coefficient |
| `AUTO_ENTROPY` | `True` | SAC entropy auto-tuning |
| `TARGET_ENTROPY` | `-64.0` | Target entropy over the 128-dim steering noise (≈ −dim/2) |
| `BACKUP_ENTROPY` | `False` | Keep the −α·logπ term out of the critic TD target (RLinf parity) |
| `CRITIC_WARMUP_STEPS` | `100` | Critic-only steps before actor updates |
| `EMA_DECAY` | `null` | Actor EMA over the tiny noise actor (null = off) |
| `CRITIC_POOL_PROJ_DIM` | `0` | Critic pooled-feature projection (SAC baseline 256) |
| `CRITIC_LAYERNORM` | `True` | LayerNorm in critic heads (SAC baseline True) |
| `ACTOR_POSITIVE_SAMPLE_RATIO` | `0.8` | Positive replay ratio for actor batches |
| `EVAL_EPISODES` | `GPUs×NUM_ENV` | Trajectories averaged per eval SR |
| `EPISODIC_REPLAY` | `True` | Episodic replay collection |

The SAC launcher's `FREEZE_ACTION_IO` / `FLOW_SDE_*` knobs are intentionally
absent: DSRL freezes the whole VLA and owns the exploration noise
(`flow_sde_enable=true` raises at model init).

Adapter-level knobs live under `cluster.actor_rollout_ref.model.adapter.dsrl.*`
(`hidden_dims`, `feature_latent_dim`, `state_latent_dim`, `noise_per_step`,
`noise_bound`) — see `src/verl_vla/workflows/config/model/adapter/gr00t.yaml`.

## Caveats

- Mutually exclusive with `flow_sde_enable` (DSRL owns the exploration noise).
- TD3+BC and offline RLPD prefill are incompatible: demos are env actions,
  while the DSRL SAC action space is the steering noise.
- Eval (`eval=True`) uses the deterministic steering noise `tanh(mean)`.
- Checkpoints: the frozen policy is exported unchanged; the noise actor is
  saved alongside as `dsrl_noise_actor.pt` (critic as `critic.pt`).
