#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATASET_ROOT="./.data/act_sft/datasets/libero_spatial_image"

cd "$REPO_ROOT"

hf download lerobot/libero_spatial_image \
  --repo-type dataset \
  --local-dir "$DATASET_ROOT"

vvla-train-sft \
  --config-dir "./examples/fine_tuning/act/official_libero_spatial" \
  --config-name act_sft \
  cluster.actor_rollout_ref.model.path="./assets/hf_models/act_libero" \
  data.repo_id=lerobot/libero_spatial_image \
  data.root="$DATASET_ROOT" \
  data.batch_size=32 \
  cluster.resource.model.gpus_per_node=1 \
  cluster.actor_rollout_ref.actor.mini_batch_size=32 \
  cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
  'trainer.logger=[console,tensorboard]' \
  trainer.total_epochs=5 \
  "$@"
