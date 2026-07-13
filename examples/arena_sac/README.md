# Arena eval / SAC runbook (`examples/arena_sac/`)

Everything you need to run GR00T N1.6 and PI0.5 policies on IsaacLab-Arena
tasks — eval, SAC training, and the PI0.5 teleop smoke — from the host.

The flow is always the same:

```
run_docker.sh   →   (re)creates the right container + mounts   →   runs an inner script
```

- **`run_docker.sh`** is the *single* container launcher. It picks the backend
  image, mounts the right host dirs, and `docker exec`s an inner script.
- The inner script (selected with `EVAL_SCRIPT`) builds the actual Hydra command
  and runs it with the in-container `python`.

## Files

| File | Role |
| --- | --- |
| `run_docker.sh` | **Single** container launcher. `BACKEND=gr00t\|pi05` selects image/mounts/user. Runs the inner script named by `EVAL_SCRIPT`. |
| `run_gr00t_arena_eval.sh` | GR00T RECAP `policy_eval`. `ARENA_TASK=gr1\|libero`. |
| `run_gr00t_arena_sac.sh` | GR00T SAC training. `ARENA_TASK=gr1\|libero`. |
| `run_pi05_arena_g1_eval.sh` | PI0.5 RECAP `policy_eval` (Arena G1). |
| `run_pi05_arena_g1_sac_smoke.sh` | PI0.5 teleop data-collection smoke (Arena G1). |

Inner scripts are meant to run *inside* the container, but `run_docker.sh`
launches them for you — you normally never call them directly.

## Backends (images / containers / mounts)

| | `BACKEND=gr00t` (default) | `BACKEND=pi05` |
| --- | --- | --- |
| Image | `isaaclab_arena:cuda_gr00t_gn16` | `vemlp-demo-cn-beijing.cr.volces.com/verl/verl-vla-arena:v0.2` |
| Container name | `isaaclab_arena-cuda_gr00t_gn16` | `verl-vla-arena` |
| Container user | root | `isaac-sim` (uid 1234) |
| Repo mount | host repo → `/eval` | host repo → `/workspaces/verl-vla` |
| GR00T deps | `/opt/groot_deps` (Eagle / transformers 4.51.3) | none |
| Extra mounts | Arena, `/models`, `/libero_in_lab`, `cuda128-compat` | none |
| Default inner script | `run_gr00t_arena_eval.sh` | `run_pi05_arena_g1_eval.sh` |

Only the GR00T backend needs the Arena checkout, checkpoints, and LIBERO assets
mounted. PI0.5 only bind-mounts the repo.

### GR00T host → container mounts

| Host path (default) | Container path | Set via |
| --- | --- | --- |
| `<verl-vla repo>` | `/eval` | (this repo) |
| `~/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out` | `/models` | `MODELS_HOST` |
| `~/Projects/libero_rl_example/IsaacLab-Arena` | `/workspaces/isaaclab_arena` | `ARENA_HOST` |
| `~/Projects/libero_rl_example/libero_in_lab` | `/libero_in_lab` | `LIBERO_IN_LAB_HOST` |
| `~/cuda128-compat` (if present) | `/opt/cuda128-compat` (ro) | auto |

> `ARENA_HOST` **must** exist — the host checkout provides the wrist-cam / env
> patches that the (often stale) baked image copy lacks. The launcher verifies
> `GR1T2WristCameraCfg` is present after mounting.

## Path / checkpoint defaults

| What | Default (container path unless noted) | Env var |
| --- | --- | --- |
| GR00T checkpoint | `/models/checkpoint-5000` | `GROOT_MODEL_PATH` |
| GR00T checkpoint parent (host) | `~/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out` | `MODELS_HOST` |
| PI0.5 checkpoint | `/workspaces/models/torch_pi05_base` | `MODEL_PATH` |
| GR1 joint-space YAMLs | `/workspaces/isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1` | `ARENA_GR1_JOINT_SPACE_DIR` |
| LIBERO assets root | `/libero_in_lab` | `LIBERO_IN_LAB_ROOT` |

### Suggested checkpoint parents per task

| Task | `MODELS_HOST` (host) | `GROOT_MODEL_PATH` (container) |
| --- | --- | --- |
| GR1 fridge | `~/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out` | `/models/checkpoint-5000` |
| LIBERO | `~/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec` | `/models/checkpoint-5000` |

## Output directories

Outputs land under `<repo>/outputs/…` on the host (the repo is bind-mounted).
`OUTPUT_ROOT` is a *container* path; for GR00T it defaults under `/eval/outputs`,
for PI0.5 under `/workspaces/verl-vla/outputs`.

| Run | Default `OUTPUT_ROOT` (basename under `outputs/`) | Contents |
| --- | --- | --- |
| GR00T GR1 eval | `arena_gr00t_gr1_eval` | `videos/`, `eval_metrics/` |
| GR00T LIBERO eval | `arena_gr00t_<suite>_task<id>_eval` | `videos/`, `eval_metrics/` |
| GR00T GR1 SAC | `arena_gr00t_gr1_sac` | `videos/`, `checkpoints/`, `replay_pools/` |
| GR00T LIBERO SAC | `arena_gr00t_libero_sac` | `videos/`, `checkpoints/`, `replay_pools/` |
| PI0.5 G1 eval | `arena_g1_eval` | `videos/`, `eval_metrics/` |
| PI0.5 G1 smoke | `arena_g1_smoke` | `lerobot/`, `videos/` |

## Environment variables

### `run_docker.sh` (launcher)

| Var | Default | Meaning |
| --- | --- | --- |
| `BACKEND` | `gr00t` | `gr00t` or `pi05` — selects image / mounts / user. |
| `EVAL_SCRIPT` | backend default | Inner script (path relative to the repo inside the container). |
| `IMAGE` | backend default | Override the docker image. |
| `CONTAINER_NAME` | backend default | Override the container name. |
| `RECREATE` | `0` | `1` forces remove + recreate of the container. |
| `MAX_EPISODES` | `10` | Episodes to evaluate (ignored by SAC training). |
| `OUTPUT_ROOT` | inner-script default | Eval/train output root (container path). |
| `ARENA_TASK` | `gr1` | Forwarded to GR00T inner scripts (`gr1`/`libero`). |
| `MODELS_HOST` | `~/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out` | (gr00t) checkpoint parent → `/models`. |
| `GROOT_MODEL_PATH` | `/models/checkpoint-5000` | (gr00t) checkpoint path inside the container. |
| `ARENA_HOST` | `~/Projects/libero_rl_example/IsaacLab-Arena` | (gr00t) Arena checkout → `/workspaces/isaaclab_arena`. |
| `LIBERO_IN_LAB_HOST` | `~/Projects/libero_rl_example/libero_in_lab` | (gr00t) LIBERO assets → `/libero_in_lab`. |
| `TASK_SUITE` / `TASK_ID` | (unset → inner default) | (gr00t libero) LIBERO suite / task id. |
| `MODEL_PATH` | `/workspaces/models/torch_pi05_base` | (pi05) policy checkpoint. |
| `MOUNT_REPO` | `1` | (pi05) `0` = use the image's baked repo copy instead of bind-mount. |

### GR00T inner scripts (`run_gr00t_arena_eval.sh`, `run_gr00t_arena_sac.sh`)

| Var | GR1 default | LIBERO default | Meaning |
| --- | --- | --- | --- |
| `ARENA_TASK` | `gr1` | `libero` | Selects simulator + embodiment. |
| `GROOT_EMBODIMENT_TAG` | `gr1` | `new_embodiment` | Embodiment tag. |
| `GROOT_EMBODIMENT_ID` | `20` | `10` | Projector index. |
| `ACTION_DIM` | `26` | `7` | Real (unpadded) env action width. |
| `NUM_ACTION_CHUNKS` | `16` | `16` | Executed action-chunk length (must match training). |
| `MAX_INTERACTIONS` | `32` | `10` | `env_loop` interactions per rollout. |
| `TASK_SUITE` / `TASK_ID` | — | `libero_spatial` / `3` | LIBERO task (libero only). |
| `MAX_EPISODES` (eval) | `10` | `10` | Episodes to evaluate. |

`run_gr00t_arena_sac.sh` additionally exposes SAC knobs (all overridable):
`NUM_ENV` (8), `NUM_ENV_GPUS`/`NUM_MODEL_GPUS` (1), `NUM_STAGE` (2),
`MINI_BATCH_SIZE` (128), `MICRO_BATCH_SIZE` (32), `TOTAL_EPOCHS` (1000),
`ROLLOUT_INTERVAL` (20), `WARM_ROLLOUT_STEPS` (5), `CRITIC_WARMUP_STEPS` (200),
`SAVE_FREQ` (500), `INITIAL_ALPHA` (0.01), `ALPHA_TYPE` (softplus),
`AUTO_ENTROPY` (False), `CRITIC_TAU` (0.01), `RESUME_MODE`/`RESUME_FROM_PATH`,
`PROJECT_NAME`, `EXPERIMENT_NAME`, `TRAINER_LOGGER` (`[console]`).

## Typical commands

Run all of these from the repo root on the host.

### GR00T — GR1 fridge eval (verified default)

```bash
examples/arena_sac/run_docker.sh
```

### GR00T — LIBERO spatial task 3 eval

```bash
EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_eval.sh \
ARENA_TASK=libero \
MODELS_HOST=~/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec \
  examples/arena_sac/run_docker.sh
```

### GR00T — GR1 SAC training

```bash
EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_sac.sh \
ARENA_TASK=gr1 \
MODELS_HOST=~/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out \
OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_sac \
  examples/arena_sac/run_docker.sh
```

### GR00T — LIBERO SAC training (separate container recommended)

```bash
EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_sac.sh \
ARENA_TASK=libero \
MODELS_HOST=~/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec \
OUTPUT_ROOT=/eval/outputs/arena_gr00t_libero_sac \
CONTAINER_NAME=isaaclab_arena-cuda_gr00t_gn16_sac \
  examples/arena_sac/run_docker.sh
```

### PI0.5 — G1 eval

```bash
BACKEND=pi05 examples/arena_sac/run_docker.sh
```

### PI0.5 — G1 teleop smoke

```bash
BACKEND=pi05 \
EVAL_SCRIPT=examples/arena_sac/run_pi05_arena_g1_sac_smoke.sh \
  examples/arena_sac/run_docker.sh
```

### Just start a container / shell

```bash
examples/arena_sac/run_docker.sh --no-run            # (re)start GR00T container only
examples/arena_sac/run_docker.sh --shell             # GR00T interactive shell
BACKEND=pi05 examples/arena_sac/run_docker.sh --shell # PI0.5 interactive shell
```

Extra Hydra overrides can be appended to any inner script and are forwarded via
`"$@"`, e.g. run inside `--shell`:

```bash
GROOT_MODEL_PATH=/models/checkpoint-5000 \
  bash examples/arena_sac/run_gr00t_arena_eval.sh recap.policy_eval.max_episodes=1
```

## Notes / gotchas

- **Pick the right backend.** `verl-vla-arena` has no GR00T deps; GR00T runs
  fail there. `isaaclab_arena:cuda_gr00t_gn16` is only for GR00T.
- **`EVAL_SCRIPT` is the interface.** `run_docker.sh` selects the inner script
  purely by `EVAL_SCRIPT`; keep the GR00T inner scripts under `examples/arena_sac/`.
- **verl bootstrap (GR00T only).** The GR00T image ships no `verl`; the inner
  scripts install `verl==0.7.1 --no-deps` on first run and pin
  `torch/transformers(4.51.3)/numpy` so Eagle is not upgraded. A `verl_constraints.txt`
  is written under `OUTPUT_ROOT`.
- **NCCL fix (GR00T only).** The inner scripts disable a stray cu13 NCCL under
  `/opt/groot_deps` so torch keeps its cu12 NCCL.
- **Output ownership.** GR00T runs as root; the launcher `chmod`s the output
  root so host-side files stay readable/writable.
- **LIBERO SAC + GR1 SAC in parallel:** use a distinct `CONTAINER_NAME` so the
  two runs get separate containers.
