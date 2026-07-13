#!/usr/bin/env bash
set -e

exec docker run --rm -it \
  --gpus all \
  --ipc=host \
  -p 6006:6006 \
  --entrypoint /bin/bash \
  -v "$PWD:/workspace/project" \
  -w /workspace/project \
  -e HF_ENDPOINT \
  -e HF_HOME=/workspace/project/.data/pi05_sft/huggingface \
  -e PYTHONPATH=/workspace/project/src \
  verl-vla-pi0:dev \
  -c '
    set -e
    python3 -m pip install --no-deps -e .

    mkdir -p .data/pi05_sft/output/pi05_libero_spatial_sft/tensorboard
    tensorboard \
      --logdir .data/pi05_sft/output/pi05_libero_spatial_sft/tensorboard \
      --host 0.0.0.0 \
      --port 6006 \
      >/tmp/tensorboard.log 2>&1 &
    echo "TensorBoard: http://localhost:6006"

    hf download lerobot/libero_spatial_image \
      --repo-type dataset \
      --local-dir .data/pi05_sft/huggingface/lerobot/lerobot/libero_spatial_image

    if [[ ! -f .data/pi05_sft/huggingface/lerobot/lerobot/libero_spatial_image/norm_stats.json ]]; then
      python3 scripts/compute_norm_stats.py \
        --repo-id lerobot/libero_spatial_image \
        --output-path .data/pi05_sft/huggingface/lerobot/lerobot/libero_spatial_image/norm_stats.json \
        --batch-size 64 \
        --num-workers 8
    fi

    exec bash examples/pi05_sft/run_pi05_libero_spatial_sft.sh
  '
