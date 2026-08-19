# GR00T N1.6 DSRL on Isaac Lab Arena LIBERO

DSRL ([arXiv:2506.15799](https://arxiv.org/abs/2506.15799)) freezes the whole
GR00T N1.6 policy and trains only a small Transformer noise actor over the
flow-matching initial noise `x0`, plus its SAC critic. The steering noise is the
SAC action: it seeds the frozen flow head, which decodes it into the
environment action with a deterministic Euler ODE.

`run_train.sh` reproduces the published LIBERO-Spatial runs. The
results, baselines, and known measurement gaps are in
[the recipe guide](../../../../../docs/reinforcement-learning/dsrl/gr00t/arena-libero-spatial.md).

## Setup

### 1. Build the image

Build from the repository root. The Dockerfile clones Isaac Lab Arena at a
pinned commit and installs its Isaac Lab and GR00T submodules, so no separate
Arena checkout is needed.

```bash
docker build \
  -f docker/Dockerfile.isaaclab_arena \
  -t isaaclab_arena:gr00t-runtime \
  .
```

Run everything below inside a container from this image. Arena lives at
`/workspace/arena` and this repository at `/workspace/project`.

### 2. Download the starting checkpoint

```bash
mkdir -p .data/models
hf download china-sae-robotics/gr00t_n16_arena_libero_all_suites_rel_rotvec \
  --include "checkpoint-10000/*" \
  --local-dir .data/models
```

The include pattern preserves the `checkpoint-10000/` directory, producing the
launcher's default `MODEL_PATH` of `.data/models/checkpoint-10000`.

### 3. Provide the LIBERO assets

Arena resolves the LIBERO scene USDs and the demonstration HDF5 files (used for
state reset) from `LIBERO_DATA_ROOT`. They are not in the image.

```bash
hf download china-sae-robotics/RobotLearningLab_Dataset \
  --repo-type dataset \
  --include "libero/USD/*" "libero/assembled_hdf5/*" \
  --local-dir .data
```

This lands the two directories at `.data/libero/{USD,assembled_hdf5}`. The
task-config JSONs are not on that dataset, but they ship with Arena inside the
image, so nothing else is needed.

## Run

The image does not contain the checkpoint or the LIBERO assets — `.data/` is in
`.dockerignore` — so bind-mount them, along with the repository itself:

```bash
docker run --rm --gpus all --network=host --ipc=host --privileged --shm-size=64g \
  -v "$PWD":/workspace/project \
  -v "$PWD/.data/models/checkpoint-10000":/workspace/project/.data/models/checkpoint-10000:ro \
  -v "$PWD/.data/libero/USD":/workspace/project/.data/libero/USD:ro \
  -v "$PWD/.data/libero/assembled_hdf5":/workspace/project/.data/libero/assembled_hdf5:ro \
  -w /workspace/project -e RAY_TMPDIR=/tmp/ray -e TASK_ID=7 \
  isaaclab_arena:gr00t-runtime \
  bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train.sh
```

Inside an already-running container it is just:

```bash
TASK_ID=7 \
  bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train.sh
```

The first run on a fresh image spends a long time before the first training
step: Isaac Sim downloads its assets and compiles warp kernels and shaders.
Those land in `.data/gr00t_arena/cache/`, which the launcher creates and the
image symlinks `/root/.cache` to, so later runs are much faster. Raise
`cluster.env.env_worker.simulator_start_timeout_s` with a Hydra override if the
cold start still times out.

`dsrl.yaml` owns the recipe — noise-actor and critic architecture, learning
rates, entropy schedule, replay sampling, rollout and evaluation cadence. The
launcher owns only what varies with the machine. Anything else is a Hydra
override passed as an argument:

```bash
TASK_ID=7 bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train.sh \
  cluster.actor_rollout_ref.actor.sac.target_entropy=-64.0
```

### Environment variables

The launcher exposes only what varies with the machine. Everything else lives
in `dsrl.yaml`.

| Var | Default | Meaning |
| --- | --- | --- |
| `TASK_ID` | `3` | LIBERO Spatial task, zero-based |
| `MODEL_PATH` | `.data/models/checkpoint-10000` | Starting GR00T checkpoint |
| `OUTPUT_DIR` | `outputs/rl/dsrl/gr00t/arena-libero-spatial-task<id>` | Output root; checkpoints, replay, videos, and TensorBoard derive from it |
| `NUM_ENV_GPUS` / `NUM_MODEL_GPUS` | `4` / `4` | Simulation and model GPU pools |
| `NUM_ENV` | `64` | Isaac environments **per env worker** |
| `LIBERO_DATA_ROOT` | `.data/libero` | LIBERO USD / HDF5 asset tree |
| `LIBERO_ASSETS_DATA_DIR` | `$LIBERO_DATA_ROOT/USD` | Scene / object USDs |
| `LIBERO_ASSEMBLED_DATASET_DIR` | `$LIBERO_DATA_ROOT/assembled_hdf5` | Demo HDF5 for state reset |
| `LIBERO_CONFIG_DIR` | under `$ARENA_ROOT` | Arena's LIBERO task-config JSONs |
| `ARENA_ROOT` | `/workspace/arena` | Arena checkout inside the image |

Recipe settings — noise-actor and critic architecture, learning rates, entropy
schedule, replay sampling, rollout and evaluation cadence, the training-step
budget — are in `dsrl.yaml`. Change one for a single run with a Hydra override
rather than editing the file:

```bash
TASK_ID=7 bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train.sh \
  trainer.total_training_steps=2000 \
  cluster.env.env_worker.simulator_start_timeout_s=3600
```

Inspect the fully composed job without launching training:

```bash
bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train.sh --cfg job
```

## Topology

The reference setup is one 8-GPU node: 4 simulation GPUs and 4 model GPUs. The
two Ray pools are **disjoint**, so this needs 8 GPUs, not 4. `NUM_ENV` is per
env worker, so the default gives 4 x 64 = 256 parallel auto-reset environments.

Checkpoints are FSDP shards saved at `world_size=4` and the checkpoint manager
does not reshard, so any later resume or evaluation must use the same
`NUM_MODEL_GPUS`.

## Storage

A run writes roughly 90 GB of replay shards and about 14 GB per saved
checkpoint. Point `OUTPUT_DIR` at node-local scratch; replay, checkpoints,
videos, and TensorBoard logs are all derived from that one run root. The replay
pool is regenerable and can be deleted after the run.

Use a fresh `OUTPUT_DIR` for every new experiment. SAC restores replay shards
found in an existing directory independently of model-checkpoint resume.

## Monitoring

```bash
tensorboard --logdir outputs/rl/dsrl/gr00t/arena-libero-spatial-task7/tensorboard --bind_all
```

Select checkpoints on `val/trajectory_success_rate`. The curve is strongly
non-monotonic on every task, so the final checkpoint is often not the best one.

## Notes

- **Replay is episodic by default.** `EpisodeBuffer`
  (`src/verl_vla/trainer/sac/episode_buffer.py`, added in
  [#17](https://github.com/verl-project/verl-vla/pull/17)) buffers raw rollout
  steps per environment lane until the episode completes, so an episode
  spanning several rollout windows keeps its early and middle transitions.
  There is no feature flag. It does require `auto_reset=true`, which this
  launcher sets: a non-auto-reset window that ends with unfinished lanes raises
  rather than splicing across the next real reset.
- **`target_entropy=0` anneals alpha to zero by construction.** The alpha loss
  is `-alpha * (log_pi + target_entropy)`, and the achievable entropy of the
  steering-noise distribution is far above zero, so the gradient always pushes
  alpha down. This is a schedule, not a controller that converges to a target.
- **DSRL is mutually exclusive with Flow-SDE, TD3+BC, and offline RLPD
  prefill.** Those paths operate on environment actions rather than steering
  noise. The verl checkpoint carries the noise actor and critic; the native
  Hugging Face export stays an unchanged upstream policy.
- The SAC and evaluation launchers for GR00T Arena live in
  [`examples/rl/sac/gr00t/`](../../../sac/gr00t/README.md) and still use
  `run_docker.sh`. This DSRL recipe is self-contained and does not.
