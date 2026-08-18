# GR00T N1.6 DSRL on Arena LIBERO Spatial

This guide records DSRL online RL recipes for GR00T N1.6 on all 10 LIBERO
Spatial tasks, simulated through Isaac Lab Arena with GPU-parallel simulation
and photorealistic rendering. Each task is trained as an independent
single-task run from the same GR00T N1.6 SFT checkpoint. The GR00T policy stays
frozen throughout; only a Transformer noise actor over the flow-matching
initial noise and its SAC critic are optimized.

Across the ten tasks the best DSRL checkpoint improves success by **15.2
percentage points** over the supervised starting policy, from 73.9% to 89.1%.

```{figure} ../../../_static/images/gr00t-libero-spatial-dsrl-grid.gif
:alt: Nine LIBERO Spatial rollouts tiled 3x3, each captioned with its SFT and DSRL success rates
:width: 620px
:align: center

One evaluation rollout per task, replayed from the archived best checkpoint.
Each cell is captioned with its task ID, its SFT success rate, the DSRL peak,
and the gain.
```

Unlike the [PI0.5 DSRL recipe](../pi05/libero-spatial.md), the actor and critic
here consume GR00T's own frozen mean-pooled vision-language prefix instead of
an independent CNN encoder, so the steering module is aligned with the flow
head it steers. The critic is a Transformer sequence critic over
`[obs_token, action_1..K]` with four min-reduced heads.

## Install the environment

Build and enter the Arena GR00T image before running this recipe. Build from
the repository root; the Dockerfile clones Isaac Lab Arena at a pinned commit
and installs its Isaac Lab and GR00T submodules itself, so no separate Arena
checkout is needed.

```bash
docker build \
  -f docker/Dockerfile.isaaclab_arena \
  -t isaaclab_arena:gr00t-runtime \
  .
```

Unless stated otherwise, run every command below inside a container built from
this image. Inside it, Arena lives at `/workspace/arena` and this repository at
`/workspace/project`.

The tested topology is a single node with eight GPUs: four environment GPUs
hosting 64 Isaac environments each, for 256 parallel auto-reset environments,
and four model GPUs running GR00T inference plus the SAC update.

### Starting policy

The runs start from a GR00T N1.6 checkpoint fine-tuned on all LIBERO suites
with relative end-effector actions in rotation-vector form, published at
[`china-sae-robotics/gr00t_n16_arena_libero_all_suites_rel_rotvec`](https://huggingface.co/china-sae-robotics/gr00t_n16_arena_libero_all_suites_rel_rotvec).
Download it from the repository root:

```bash
mkdir -p .data/models
hf download china-sae-robotics/gr00t_n16_arena_libero_all_suites_rel_rotvec \
  --include "checkpoint-10000/*" \
  --local-dir .data/models
```

The include pattern preserves the `checkpoint-10000/` directory, producing the
launcher's default path `.data/models/checkpoint-10000`. Its supervised
fine-tuning used the
[`china-sae-robotics/all_libero_suites_rel_rotvec`](https://huggingface.co/datasets/china-sae-robotics/all_libero_suites_rel_rotvec)
LeRobot dataset under `embodiment_tag: new_embodiment` with
`use_relative_action: true` and an action horizon of 16, which is why this
recipe sets `embodiment_id=10`, `action_dim=7`, and `num_action_chunks=16`. Its
actions are 7-DOF relative end-effector poses in rotation-vector form, not the
8-DOF absolute quaternion layout used by the per-suite LIBERO LeRobot datasets.
The published checkpoint is step 10,000 of a 20,000-step schedule, so it is
deliberately not a fully converged policy.

DSRL itself needs only the checkpoint; the dataset is required only to
reproduce the supervised starting policy.

## Start training

Run the launcher from the repository root, once per task:

```bash
TASK_ID=7 TOTAL_TRAINING_STEPS=8000 \
  bash examples/rl/dsrl/gr00t/run_gr00t_arena_libero_dsrl.sh
```

`TASK_ID` is the zero-based LIBERO Spatial task. The launcher
hard-codes the settings that define this recipe and exposes only the task,
schedule, topology, and paths as environment variables. Any extra argument is
forwarded to Hydra verbatim, so a one-off change needs no edit:

```bash
TASK_ID=7 bash examples/rl/dsrl/gr00t/run_gr00t_arena_libero_dsrl.sh \
  cluster.actor_rollout_ref.actor.sac.target_entropy=-64.0
```

The LIBERO USD and HDF5 assets are not in the image. Provide them on the host
and point `LIBERO_DATA_ROOT` at the tree:

```bash
hf download china-sae-robotics/RobotLearningLab_Dataset \
  --repo-type dataset \
  --include "libero/USD/*" "libero/assembled_hdf5/*" \
  --local-dir .data
```

This lands the two directories at `.data/libero/{USD,assembled_hdf5}`. The
task-config JSONs are not on that dataset, but they ship with Arena inside the
image.

A single run writes roughly 90 GB of replay shards and about 14 GB per saved
checkpoint, so point the replay and checkpoint directories at node-local
scratch that is bind-mounted from the host. Use an empty output directory for
every fresh run, because SAC restores replay shards found in an existing output
directory independently of model-checkpoint resume.

## Experiment configuration

All ten runs share one configuration; only `libero_task_id` differs.

| Setting | Value |
| --- | --- |
| Task suite | Arena `libero_spatial`, tasks 0-9 |
| Starting policy | [`gr00t_n16_arena_libero_all_suites_rel_rotvec`](https://huggingface.co/china-sae-robotics/gr00t_n16_arena_libero_all_suites_rel_rotvec) `checkpoint-10000` |
| SFT dataset | [`china-sae-robotics/all_libero_suites_rel_rotvec`](https://huggingface.co/datasets/china-sae-robotics/all_libero_suites_rel_rotvec) |
| Native GR00T policy | Frozen |
| Embodiment | `new_embodiment`, `embodiment_id=10` |
| Action dimension | 7 (`eef_pose`, relative rotvec) |
| Action chunk | 16 steps |
| DSRL actor | Transformer chunking noise actor, per-flow-step latent |
| Actor feature source | GR00T frozen mean-pooled VL prefix |
| Critic | Transformer sequence critic, 4 heads, min-reduced |
| Critic layernorm | Enabled |
| Actor learning rate | `5e-5`, constant schedule |
| Critic learning rate | `1e-4` |
| Critic target update `tau` | `0.005` |
| Critic warmup | 100 steps |
| Actor update interval | Every step |
| Entropy tuning | Auto, `softplus`, `initial_alpha=1.0` |
| Target entropy | `0.0` |
| Backup entropy in TD target | Disabled |
| Global / micro batch size | 128 / 32 |
| Actor replay sampling | 60% positive, 40% negative |
| Online replay buffer | 20,000 per pool, positive and negative (40,000 total) |
| Replay collection | Episodic: complete episodes buffered per lane |
| Rollout interval | Every 160 training steps |
| Warm rollout steps | 5 |
| Rollout window | 20 interactions x 16 chunks = 320 env steps |
| Simulation GPUs | 4 env workers, disjoint from the model GPUs |
| Environment parallelism | 4 env workers x 64 environments = 256 |
| Model GPUs | 4, FSDP `world_size=4` |
| Evaluation | 32 trajectories every 500 steps |
| Checkpoint interval | Every 500 steps |

Replay collection is episodic by default: `EpisodeBuffer` holds raw rollout
steps per environment lane until the episode completes, so an episode spanning
several 320-step rollout windows keeps its early and middle transitions. This
requires `auto_reset=true`, which the launcher sets.

The DSRL actor emits an independent latent per flow step and steers only the
seven real action dimensions. GR00T pads actions to `max_action_dim`, and the
padding columns are **tiled** from that steered seven-dimensional block rather
than drawn fresh: `steering_noise.repeat(...)` truncated to the padded width.
The flow head denoises every padded dimension jointly, so tiling keeps the
decode a deterministic function of the SAC action — none of the base policy's
own sampling noise leaks in — while the SAC action space stays at the robot's
seven DOF.

TD3+BC and offline RLPD prefill are incompatible with DSRL, because
demonstrations are environment actions rather than steering noise, and stay
disabled.

## Outputs and monitoring

Artifacts are written below the selected output root:

```text
<output_root>/
|-- checkpoints/
|-- replay_pools/
|-- tensorboard/
`-- videos/
```

Start TensorBoard against that directory:

```bash
tensorboard \
  --logdir outputs/rl/dsrl/gr00t/libero_spatial-task7/tensorboard \
  --bind_all \
  --port 6008
```

Use `val/trajectory_success_rate` to select a checkpoint. The validation curve
is strongly non-monotonic on every task, so the final checkpoint should not
automatically replace the best observed checkpoint.


## Results

The baseline is **not** the success rate the training log prints at step 0. The
DSRL entrypoint hardcodes `dsrl.enabled=true`, and an untrained noise actor
under deterministic evaluation collapses `x0` toward zero instead of sampling
from `N(0, I)`. The baselines below therefore come from the plain evaluation
entrypoint with DSRL absent entirely, measured with 128 environments, a 320-step
budget, and a 200-step task horizon. Every environment contributes exactly one
episode, so n = 128 and the standard error is near 4.4%.

Each run was stopped manually rather than at a fixed step budget, so the
horizons differ per task. "Peak" is the best logged evaluation over the run.
Every task ends by placing the bowl on the plate.

| Task | Instruction | SFT | Peak | Gain | Best step |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Pick up the black bowl between the plate and the ramekin | 76.6% | 96.9% | **+20.3** | 4,000 |
| 1 | Pick up the black bowl next to the ramekin | 93.0% | 96.9% | +3.9 | 500 |
| 2 | Pick up the black bowl from table center | 60.2% | 78.1% | **+18.0** | 1,000 |
| 3 | Pick up the black bowl on the cookie box | 88.3% | 100.0% | +11.7 | 2,500 |
| 4 | Pick up the black bowl in the top drawer of the wooden cabinet | 51.6% | 68.8% | **+17.2** | 500 |
| 5 | Pick up the black bowl on the ramekin | 81.2% | 96.9% | **+15.6** | 5,500 |
| 6 | Pick up the black bowl next to the cookie box | 88.3% | 93.8% | +5.5 | 4,000 |
| 7 | Pick up the black bowl on the stove | 30.5% | 81.2% | **+50.8** | 3,500 |
| 8 | Pick up the black bowl next to the plate | 93.0% | 93.8% | +0.8 | 9,500 |
| 9 | Pick up the black bowl on the wooden cabinet | 76.6% | 84.4% | +7.8 | 8,500 |
| | **Mean** | **73.9%** | **89.1%** | **+15.2** | |

The gain is inversely related to the starting policy's competence. The five
tasks starting below 78% gain 7.8 to 50.8 points, averaging +22.8; the five
starting at 81% or above gain 0.8 to 15.6, averaging +7.5. DSRL recovers
headroom the supervised policy left on the table rather than pushing an
already-strong policy further. Most of the gain arrives early: four of the ten
runs peak at or before step 2,500, and tasks 1 and 4 peak at their very first
evaluation.
