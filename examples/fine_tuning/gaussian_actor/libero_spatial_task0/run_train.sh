#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATASET_ROOT="./.data/gaussian_actor_sft/datasets/libero_spatial_image_task0"

cd "$REPO_ROOT"

hf download Miical/libero_spatial_image_task0 \
  --repo-type dataset \
  --local-dir "$DATASET_ROOT"

vvla-train-sft \
  --config-dir "./examples/fine_tuning/gaussian_actor/libero_spatial_task0" \
  --config-name gaussian_actor_sft \
  cluster.actor_rollout_ref.model.path="./assets/hf_models/gaussian_actor_libero" \
  data.repo_id=Miical/libero_spatial_image_task0 \
  data.root="$DATASET_ROOT" \
  data.batch_size=32 \
  cluster.resource.model.gpus_per_node=1 \
  cluster.actor_rollout_ref.actor.mini_batch_size=32 \
  cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
  'trainer.logger=[console,tensorboard]' \
  trainer.total_epochs=8 \
  "$@"
