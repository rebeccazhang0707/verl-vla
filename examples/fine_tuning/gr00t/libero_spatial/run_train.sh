#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "$REPO_ROOT"

python scripts/install_checks/check_gr00t_n1d6.py

vvla-train-sft \
  --config-dir ./examples/fine_tuning/gr00t/libero_spatial \
  --config-name gr00t_sft \
  cluster.actor_rollout_ref.model.path=./.data/gr00t_sft/models/gr00t_n1d6_3b \
  cluster.actor_rollout_ref.model.adapter.norm_stats_path=./.data/gr00t_sft/datasets/libero_spatial_image/norm_stats.json \
  data.repo_id=lerobot/libero_spatial_image \
  data.root=./.data/gr00t_sft/datasets/libero_spatial_image \
  data.batch_size=256 \
  data.num_workers=8 \
  cluster.resource.model.gpus_per_node=8 \
  cluster.actor_rollout_ref.actor.mini_batch_size=256 \
  cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
  cluster.actor_rollout_ref.actor.optim.weight_decay=1e-5 \
  cluster.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  cluster.actor_rollout_ref.actor.optim.total_training_steps=3090 \
  cluster.checkpoint.max_actor_ckpt_to_keep=40 \
  'trainer.logger=[console,tensorboard]' \
  trainer.total_epochs=15 \
  "$@"
