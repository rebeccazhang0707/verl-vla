#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME=${IMAGE_NAME:-verl-vla-gr00t:n1.6}
MODEL_REPO_ID=${MODEL_REPO_ID:-nvidia/GR00T-N1.6-3B}
MODEL_REVISION=${MODEL_REVISION:-d0814e7ecb19202e7c8468b46098b0b7ef3a6d61}

DEFAULT_DATA_ROOT="${REPO_ROOT}/.data/gr00t_sft"
DATA_ROOT=${DATA_ROOT:-${DEFAULT_DATA_ROOT}}
DATA_ROOT="$(realpath -m "${DATA_ROOT}")"

MODEL_PATH=${MODEL_PATH:-${DATA_ROOT}/models/gr00t_n1d6_3b}
DATASET_ROOT="${DATA_ROOT}/datasets/libero_spatial_image"
NORM_STATS_PATH=${NORM_STATS_PATH:-${DATASET_ROOT}/norm_stats.json}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/output/gr00t_n1d6_libero_spatial_sft}
MODEL_PATH="$(realpath -m "${MODEL_PATH}")"
MODEL_COMPLETE_MARKER="${MODEL_PATH}/.download_complete_${MODEL_REVISION}"
NORM_STATS_PATH="$(realpath -m "${NORM_STATS_PATH}")"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"

NUM_GPUS=${NUM_GPUS:-8}
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-64}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
SFT_NUM_WORKERS=${SFT_NUM_WORKERS:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-13}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
SAVE_FREQ=${SAVE_FREQ:-500}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image not found: ${IMAGE_NAME}" >&2
  echo "Build it with: docker build -f docker/Dockerfile.gr00t -t ${IMAGE_NAME} ." >&2
  exit 2
fi

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "LIBERO Spatial dataset not found: ${DATASET_ROOT}" >&2
  exit 2
fi

if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Normalization statistics not found: ${NORM_STATS_PATH}" >&2
  echo "Compute them with scripts/compute_norm_stats.py before training." >&2
  exit 2
fi

DATA_MOUNT_ARGS=()
if [[ "${DATA_ROOT}" == "${REPO_ROOT}" ]]; then
  CONTAINER_DATA_ROOT=/workspace/verl-vla
  REPO_MOUNT="${REPO_ROOT}:/workspace/verl-vla"
elif [[ "${DATA_ROOT}" == "${REPO_ROOT}/"* ]]; then
  CONTAINER_DATA_ROOT="/workspace/verl-vla/${DATA_ROOT#"${REPO_ROOT}/"}"
  REPO_MOUNT="${REPO_ROOT}:/workspace/verl-vla"
else
  CONTAINER_DATA_ROOT=/data
  REPO_MOUNT="${REPO_ROOT}:/workspace/verl-vla:ro"
  DATA_MOUNT_ARGS=(-v "${DATA_ROOT}:/data")
fi

host_path_to_container() {
  local host_path=$1
  if [[ "${host_path}" == "${DATA_ROOT}" ]]; then
    printf '%s\n' "${CONTAINER_DATA_ROOT}"
  elif [[ "${host_path}" == "${DATA_ROOT}/"* ]]; then
    printf '%s/%s\n' "${CONTAINER_DATA_ROOT}" "${host_path#"${DATA_ROOT}/"}"
  else
    echo "Path must be inside DATA_ROOT (${DATA_ROOT}): ${host_path}" >&2
    return 2
  fi
}

CONTAINER_NORM_STATS_PATH="$(host_path_to_container "${NORM_STATS_PATH}")"
CONTAINER_OUTPUT_DIR="$(host_path_to_container "${OUTPUT_DIR}")"
CONTAINER_MODEL_PATH="$(host_path_to_container "${MODEL_PATH}")"

mkdir -p \
  "${MODEL_PATH}" \
  "${OUTPUT_DIR}" \
  "${DATA_ROOT}/huggingface" \
  "${DATA_ROOT}/raytmp" \
  "${DATA_ROOT}/tmp"

echo "Image:      ${IMAGE_NAME}"
echo "Data root:  ${DATA_ROOT}"
echo "Model:      ${MODEL_PATH} (${MODEL_REPO_ID}@${MODEL_REVISION})"
echo "GPUs:       ${NUM_GPUS}"
echo "Output:     ${OUTPUT_DIR}"

if [[ ! -f "${MODEL_COMPLETE_MARKER}" ]]; then
  echo "Downloading ${MODEL_REPO_ID}@${MODEL_REVISION} to ${MODEL_PATH}"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="${DATA_ROOT}/huggingface" \
      hf download "${MODEL_REPO_ID}" \
        --revision "${MODEL_REVISION}" \
        --local-dir "${MODEL_PATH}"
  else
    docker run --rm \
      --network host \
      --entrypoint /opt/venv/bin/hf \
      -v "${REPO_MOUNT}" \
      "${DATA_MOUNT_ARGS[@]}" \
      -e HF_HOME="${CONTAINER_DATA_ROOT}/huggingface" \
      -e HF_TOKEN \
      "${IMAGE_NAME}" \
      download "${MODEL_REPO_ID}" \
        --revision "${MODEL_REVISION}" \
        --local-dir "${CONTAINER_MODEL_PATH}"
  fi
  test -f "${MODEL_PATH}/config.json"
  touch "${MODEL_COMPLETE_MARKER}"
fi

exec docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit stack=67108864 \
  --entrypoint /bin/bash \
  -v "${REPO_MOUNT}" \
  "${DATA_MOUNT_ARGS[@]}" \
  -e HF_HOME="${CONTAINER_DATA_ROOT}/huggingface" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -e HF_TOKEN \
  -e PYTHONPATH=/workspace/verl-vla/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e RAY_TMPDIR="${CONTAINER_DATA_ROOT}/raytmp" \
  -e TMPDIR="${CONTAINER_DATA_ROOT}/tmp" \
  -e NO_ALBUMENTATIONS_UPDATE=1 \
  -e MODEL_PATH="${CONTAINER_MODEL_PATH}" \
  -e SFT_REPO_ID=lerobot/libero_spatial_image \
  -e SFT_ROOT="${CONTAINER_DATA_ROOT}/datasets/libero_spatial_image" \
  -e NORM_STATS_PATH="${CONTAINER_NORM_STATS_PATH}" \
  -e OUTPUT_DIR="${CONTAINER_OUTPUT_DIR}" \
  -e NUM_GPUS="${NUM_GPUS}" \
  -e NUM_NODES=1 \
  -e SFT_BATCH_SIZE="${SFT_BATCH_SIZE}" \
  -e MINI_BATCH_SIZE="${MINI_BATCH_SIZE}" \
  -e MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}" \
  -e SFT_NUM_WORKERS="${SFT_NUM_WORKERS}" \
  -e TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
  -e LR="${LR}" \
  -e WEIGHT_DECAY="${WEIGHT_DECAY}" \
  -e SAVE_FREQ="${SAVE_FREQ}" \
  -e MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP}" \
  -e RESUME_DATALOADER_STATE=true \
  "${IMAGE_NAME}" \
  -lc 'exec bash examples/gr00t_sft/run_gr00t_lerobot_sft.sh'
