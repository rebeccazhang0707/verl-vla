#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME=verl-vla-pi0:dev
DATA_ROOT="${REPO_ROOT}/.data/pi05_sft"
MODEL_PATH=${MODEL_PATH:-Miical/pi05-base}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
MODEL_MOUNT_ARGS=()

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Docker image not found: $IMAGE_NAME" >&2
  echo "Build it with: docker build -f docker/Dockerfile.pi0 -t $IMAGE_NAME ." >&2
  exit 2
fi

if [[ -e "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(realpath "${MODEL_PATH}")"
  MODEL_MOUNT_ARGS+=(-v "${MODEL_PATH}:${MODEL_PATH}:ro")
elif [[ "${MODEL_PATH}" == /* ]]; then
  echo "Pi0.5 model not found: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ -e "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_PATH="$(realpath "${TOKENIZER_PATH}")"
  if [[ "${TOKENIZER_PATH}" != "${MODEL_PATH}" ]]; then
    MODEL_MOUNT_ARGS+=(-v "${TOKENIZER_PATH}:${TOKENIZER_PATH}:ro")
  fi
elif [[ "${TOKENIZER_PATH}" == /* ]]; then
  echo "Pi0.5 tokenizer not found: ${TOKENIZER_PATH}" >&2
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

mkdir -p "${DATA_ROOT}/huggingface" "${DATA_ROOT}/output"

exec docker run --rm -it \
  --gpus all \
  --ipc=host \
  --entrypoint /bin/bash \
  -v "${REPO_ROOT}:/workspace/verl-vla" \
  "${MODEL_MOUNT_ARGS[@]}" \
  -e HF_ENDPOINT \
  -e HF_HOME=/workspace/verl-vla/.data/pi05_sft/huggingface \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e TOKENIZER_PATH="${TOKENIZER_PATH}" \
  "$IMAGE_NAME" \
  -lc 'python3 -m pip install --no-deps -e . && exec bash examples/pi05_sft/run_pi05_libero_spatial_sft.sh'
