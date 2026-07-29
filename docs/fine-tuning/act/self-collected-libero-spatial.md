# Fine-tune ACT on self-collected LIBERO Spatial demonstrations

This recipe fine-tunes an ACT policy on demonstrations collected from LIBERO
Spatial task 0 and evaluates the resulting policy on the same task. Follow the
[LIBERO keyboard data-collection guide](../../data-collection/libero/keyboard.md)
to prepare the demonstrations.

The commands below assume that the recorded LeRobot dataset is available at:

```text
./outputs/record/lerobot/local/libero_spatial
```

## Verified setup

| Component | Configuration |
| --- | --- |
| Policy | Native ACT, initialized from `assets/hf_models/act_libero` |
| Training data | Self-collected LeRobot demonstrations |
| Dataset identity | `local/libero_spatial` |
| Dataset root | `./outputs/record/lerobot/local/libero_spatial` |
| Embodiment | LIBERO Franka/Panda |
| Evaluation environment | LIBERO Spatial, task 0 |
| Default training resources | One CUDA-capable NVIDIA GPU |
| Action horizon | 10 steps |
| Output root | `./outputs/train/act-sft/libero-spatial` |

The initializer defines the ACT architecture but contains no policy weights.
It loads an ImageNet-pretrained ResNet18 backbone and randomly initializes the
remaining parameters.

## Prepare the environment

Use the environment from the
[Quick Start](../../getting-started/index.md), including its installation
checks. ACT training requires a CUDA-capable NVIDIA GPU.

## Prepare the dataset

The data-collection guide creates the `local/libero_spatial` dataset under
`./outputs/record/lerobot`.

### Dataset contract

The checked-in ACT initializer and the LIBERO recorder share the following
contract:

| Feature | Recorded data | ACT input |
| --- | --- | --- |
| `observation.images.image` | `uint8`, `(256, 256, 3)`, `[0, 255]` | `float32`, `(3, 256, 256)`, `[0, 1]` |
| `observation.images.wrist_image` | `uint8`, `(256, 256, 3)`, `[0, 255]` | `float32`, `(3, 256, 256)`, `[0, 1]` |
| `observation.state` | `float32`, `(8,)` | `float32`, `(8,)` |
| `action` | `float32`, `(7,)` | `float32`, `(7,)` |
| `task` | Text | Text |

LeRobot converts the recorded RGB frames from channel-last `uint8` arrays to
channel-first `float32` tensors before they reach ACT. The initializer uses a
10-step action chunk, so the recipe sets `data.action_delta_steps=10`. ACT
builds its state and action processors from the dataset statistics and saves
them with the exported policy.

### Replay a demonstration in LIBERO

Replay connects the recorded reset state and actions back to the matching
simulator:

```bash
vvla-replay \
  speed=1.0 \
  root="./outputs/record/lerobot/local/libero_spatial" \
  episode_indices='[0]'
```

Press Enter when prompted. A successful replay reports identical
`executed_frames` and `expected_frames`.

## Configure training

The recipe uses
`examples/fine_tuning/act/self_collected_libero_spatial/act_sft.yaml`, which
selects the shared SFT workflow, ACT model, and LIBERO adapter. The launcher
provides these defaults:

| Setting | Default | Purpose |
| --- | --- | --- |
| `data.repo_id` | `local/libero_spatial` | Logical LeRobot dataset identity |
| `data.root` | `./outputs/record/lerobot/local/libero_spatial` | Frames consumed by the SFT dataloader |
| `cluster.actor_rollout_ref.model.adapter.processor_dataset_root` | `./outputs/record/lerobot/local/libero_spatial` | Statistics used to initialize native ACT processors |
| `data.batch_size` | `32` | Global DataLoader batch size |
| `cluster.actor_rollout_ref.actor.mini_batch_size` | `32` | Actor mini-batch size |
| `cluster.actor_rollout_ref.actor.micro_batch_size` | `16` | Per-device micro-batch size |
| `cluster.actor_rollout_ref.actor.optim.lr` | `1e-4` | Learning rate |
| `trainer.total_epochs` | `100` | Number of full dataset passes |
| `cluster.resource.model.gpus_per_node` | `1` | GPUs used for training |
| `trainer.save_freq` | `500` | Checkpoint interval in optimizer steps |

Keep `data.root` and
`cluster.actor_rollout_ref.model.adapter.processor_dataset_root` pointed at the
same dataset when initializing ACT from `assets/hf_models/act_libero`.

To train on multiple GPUs, override the GPU count:

```bash
bash examples/fine_tuning/act/self_collected_libero_spatial/run_train.sh \
  cluster.resource.model.gpus_per_node=4
```

The global `data.batch_size` must be divisible by the number of training GPUs.

## Start full training

Run the recipe from the repository root:

```bash
bash examples/fine_tuning/act/self_collected_libero_spatial/run_train.sh
```

Checkpoints are written to:

```text
./outputs/train/act-sft/libero-spatial/checkpoints
```

`latest_checkpointed_iteration.txt` records the latest step. The evaluation
launcher loads its native ACT export from
`global_step_<N>/actor/huggingface`. Running the training command again resumes
from the latest checkpoint, including optimizer and dataloader state.

### Monitor training

The console reports `sft/loss`, gradient norm, epoch, and timing metrics.
TensorBoard event files are written to:

```text
./outputs/train/act-sft/libero-spatial/tensorboard
```

Start TensorBoard in another terminal:

```bash
tensorboard \
  --logdir "./outputs/train/act-sft/libero-spatial/tensorboard" \
  --bind_all \
  --port 6006
```

Open `http://localhost:6006`. The supervised loss shows how well the policy is
fitting the demonstrations; policy performance is measured by the LIBERO
evaluation below.

## Evaluate in LIBERO

Closed-loop evaluation uses the same ACT adapter and LIBERO Spatial task 0 that
produced the demonstrations. The evaluation launcher resolves the latest
native export automatically:

```bash
bash examples/fine_tuning/act/self_collected_libero_spatial/run_eval.sh
```

By default, it evaluates all 50 reset states for task 0 across 8 parallel
environments. LIBERO task IDs are zero-based, so `task_ids=[0]` selects task 1.

Results are written to:

```text
./outputs/eval/act-sft/libero-spatial/task-1-parallel/
├── metrics.json
└── videos/
```

Inspect `metrics.json` for episode counts, returns, and success rate. Use the
videos to review successful rollouts and diagnose failures. A successfully
trained checkpoint can still have a low success rate when the demonstrations
are insufficient or inconsistent.
