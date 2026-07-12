# Pi0.5 SFT on LIBERO Spatial

This example fine-tunes the Torch Pi0.5 base model on
`lerobot/libero_spatial_image`, the same LeRobot dataset used by the GR00T SFT
example. The launcher is configured for one node with eight GPUs.

## Directory layout

Keep the model, dataset, and training output under the repository-local
`.data/pi05_sft` directory:

```text
.data/pi05_sft/
├── datasets/
│   └── libero_spatial_image/
│       ├── data/
│       ├── meta/
│       ├── videos/
│       └── norm_stats.json
├── models/
│   └── torch_pi05_base/
│       ├── config.json
│       ├── tokenizer.json
│       └── diffusion_pytorch_model-*.safetensors
└── output/
```

Both launch methods use the same repository-relative locations:

```text
model:   .data/pi05_sft/models/torch_pi05_base
dataset: .data/pi05_sft/datasets/libero_spatial_image
output:  .data/pi05_sft/output/pi05_libero_spatial_sft
```

Docker mounts the repository at `/workspace/verl-vla`; native execution uses
the local repository path directly.

## Option 1: Docker

Docker provides the reproducible environment used by this example. The source
repository and `.data` directory remain on the host and are bind-mounted into
the container.

### Build the image

Run from the repository root:

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -f docker/Dockerfile.pi0 \
  -t verl-vla-pi0:dev \
  .
```

The build installs the Pi0.5, verl-vla, LeRobot, and LIBERO dependencies. It
also installs pinned LIBERO assets, verifies their SHA256 checksum, and creates
a LIBERO environment with OSMesa CPU rendering.

### Compute normalization statistics

Run the dataset statistics script inside the image:

```bash
docker run --rm \
  --entrypoint /bin/bash \
  -v "$PWD:/workspace/verl-vla" \
  verl-vla-pi0:dev \
  -lc 'python3 scripts/compute_norm_stats.py \
    --repo-id lerobot/libero_spatial_image \
    --root .data/pi05_sft/datasets/libero_spatial_image \
    --output-path .data/pi05_sft/datasets/libero_spatial_image/norm_stats.json \
    --batch-size 64 \
    --num-workers 8'
```

### Start training

Run the host-side launcher from anywhere in the repository:

```bash
bash examples/pi05_sft/run_docker.sh
```

The launcher verifies the image, checkpoint, dataset, and `norm_stats.json`,
then exposes all eight GPUs and starts training. Because the source repository
is bind-mounted and installed in editable mode, Python changes do not require
rebuilding the image.

## Option 2: Native

Docker is not required. The verified dependency set targets Ubuntu 22.04,
Python 3.10, CUDA 12, and an NVIDIA GPU.

### Install the environment

Install the system packages:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  cmake \
  curl \
  git \
  libgl1 \
  libglib2.0-0 \
  libosmesa6 \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv
```

Using a virtual environment is recommended for a native installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip "setuptools<81" wheel
```

Install the verified LeRobot runtime and verl-vla from the repository root:

```bash
python3 -m pip install --requirement requirements-lerobot.txt
python3 -m pip install --no-deps lerobot==0.4.4
python3 -m pip install -e ".[pi0,libero]"
```

The separate LeRobot installation is intentional. `verl==0.7.1` requires
NumPy below 2, while the `rerun-sdk` range declared by `lerobot==0.4.4`
requires NumPy 2 on Linux. `requirements-lerobot.txt` directly pins the
verified compatible runtime, and `--no-deps` prevents pip from rejecting that
known combination.

The PyPI LIBERO package does not contain its scene and object assets. Install
the assets from the same pinned official revision used by the Dockerfile:

```bash
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_SHA256=05ffcf8349b2e7ef31b038451253d76ca757debbf88c3a0c1de569ca38a80b14
LIBERO_PACKAGE_DIR=$(python3 -c \
  'from importlib.metadata import distribution; print(distribution("libero").locate_file("libero/libero"))')

curl --fail --location \
  --retry 5 --retry-all-errors \
  --output /tmp/libero-source.tar.gz \
  "https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/$LIBERO_COMMIT"
echo "$LIBERO_SHA256  /tmp/libero-source.tar.gz" | sha256sum --check --strict
rm -rf "$LIBERO_PACKAGE_DIR/assets"
tar -xzf /tmp/libero-source.tar.gz \
  --strip-components=3 \
  -C "$LIBERO_PACKAGE_DIR" \
  "LIBERO-$LIBERO_COMMIT/libero/libero/assets"
rm /tmp/libero-source.tar.gz
```

Verify package versions and create a real LIBERO environment with OSMesa CPU
rendering:

```bash
python3 scripts/install_checks/check_libero.py
```

### Compute normalization statistics

Compute dataset-specific state and action statistics once before starting
training:

```bash
python3 scripts/compute_norm_stats.py \
  --repo-id lerobot/libero_spatial_image \
  --root .data/pi05_sft/datasets/libero_spatial_image \
  --output-path .data/pi05_sft/datasets/libero_spatial_image/norm_stats.json \
  --batch-size 64 \
  --num-workers 8
```

The launcher requires this file and passes its path to Pi0.5. The model loads
the state and action statistics from the file when it is initialized instead
of using the normalization values embedded in the pretrained model config.

### Start training

Start native eight-GPU training:

```bash
source .venv/bin/activate
bash examples/pi05_sft/run_pi05_libero_spatial_sft.sh
```

## Training configuration

The training parameters are fixed in the launcher:

| Parameter | Value |
| --- | --- |
| Nodes | 1 |
| GPUs per node | 8 |
| Global batch size | 64 |
| Mini-batch size | 64 |
| Micro-batch size | 1 |
| DataLoader workers | 8 |
| Action horizon | 50 |
| Epochs | 1 |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Warmup ratio | `0.05` |
| FSDP strategy | FSDP2 |
| Model dtype | BF16 |

With 52,970 frames and a global batch size of 64, one epoch contains about 827
full training batches because the launcher drops the incomplete final batch.

Training is running successfully once all eight GPUs have worker processes and
the console reports progress with both loss and gradient metrics:

```text
Training Progress: 1/827 ... grad_pre=... sft_loss=...
```

Use `nvidia-smi` in another terminal to inspect GPU utilization:

```bash
watch -n 1 nvidia-smi
```

Hydra overrides may still be appended explicitly for one-off experiments, but
the normal eight-GPU command requires no environment variables.
