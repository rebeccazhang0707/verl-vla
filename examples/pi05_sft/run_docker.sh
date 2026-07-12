#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME=verl-vla-pi0:dev
DATA_ROOT="${REPO_ROOT}/.data/pi05_sft"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Docker image not found: $IMAGE_NAME" >&2
  echo "Build it with: docker build -f docker/Dockerfile.pi0 -t $IMAGE_NAME ." >&2
  exit 2
fi

if [[ ! -f "${DATA_ROOT}/models/torch_pi05_base/config.json" ]]; then
  echo "Pi0.5 model not found: ${DATA_ROOT}/models/torch_pi05_base" >&2
  exit 2
fi

if [[ ! -f "${DATA_ROOT}/datasets/libero_spatial_image/meta/info.json" ]]; then
  echo "LIBERO Spatial dataset not found: ${DATA_ROOT}/datasets/libero_spatial_image" >&2
  exit 2
fi

if [[ ! -f "${DATA_ROOT}/datasets/libero_spatial_image/norm_stats.json" ]]; then
  echo "Normalization statistics not found: ${DATA_ROOT}/datasets/libero_spatial_image/norm_stats.json" >&2
  echo "Compute them with scripts/compute_norm_stats.py before training." >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}/output"

exec docker run --rm -it \
  --gpus all \
  --ipc=host \
  --entrypoint /bin/bash \
  -v "${REPO_ROOT}:/workspace/verl-vla" \
  "$IMAGE_NAME" \
  -lc 'python3 -m pip install --no-deps -e . && exec bash examples/pi05_sft/run_pi05_libero_spatial_sft.sh'
