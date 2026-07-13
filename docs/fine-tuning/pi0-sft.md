# Fine-tune PI0.5 on LIBERO Spatial

This guide fine-tunes `Miical/pi05-base` on the
`lerobot/libero_spatial_image` dataset with supervised fine-tuning (SFT). The
provided launcher runs on one node with eight NVIDIA GPUs.

## Build the image

Run from the repository root:

```bash
docker build \
  -f docker/Dockerfile.pi0 \
  -t verl-vla-pi0:dev \
  .
```

The image contains the verified PI0.5, verl-vla, LeRobot, and LIBERO runtime,
including the LIBERO assets required for OSMesa rendering.

## Start training

```bash
bash examples/pi05_sft/run_docker.sh
```

The launcher uses `.data/pi05_sft` as the persistent data directory shared by
the host and the container. Source code is bind-mounted from the repository,
so Python changes are available without rebuilding the image.

On the first run, the launcher automatically:

1. downloads the LIBERO Spatial dataset through LeRobot;
2. computes the dataset normalization statistics;
3. downloads the PI0.5 checkpoint from Hugging Face; and
4. starts distributed SFT on all eight GPUs.

Downloaded files and training outputs remain under `.data/pi05_sft` and are
reused by later runs.

## Default configuration

| Setting | Value |
| --- | --- |
| Model | `Miical/pi05-base` |
| Dataset | `lerobot/libero_spatial_image` |
| Nodes | 1 |
| GPUs | 8 |
| Global batch size | 64 |
| Micro-batch size | 8 |
| DataLoader workers | 8 |
| Action horizon | 50 |
| Epochs | 10 (approximately 8,270 steps) |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Warmup ratio | `0.05` |
| Distributed strategy | FSDP2 |
| Model dtype | BF16 |
| Output | `.data/pi05_sft/output/pi05_libero_spatial_sft` |

## Monitor training

A running job reports loss and gradient metrics in the console:

```text
Training Progress: 1/8270 ... grad_pre=... sft_loss=...
```

The Docker launcher also starts TensorBoard automatically. Event files are
written to:

```text
.data/pi05_sft/output/pi05_libero_spatial_sft/tensorboard
```

Open the following address in a browser to view the training metrics:

```text
http://localhost:6006
```

When training on a remote machine, replace `localhost` with the machine address
or forward port `6006` over SSH.

GPU utilization can be inspected from another terminal:

```bash
watch -n 1 nvidia-smi
```

## Native installation

To build a local Python environment, follow the dependency versions and
installation order in
[`docker/Dockerfile.pi0`](https://github.com/verl-project/verl-vla/blob/main/docker/Dockerfile.pi0).
