#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="$PWD/outputs/train/pi05-sft/libero-spatial"
NORM_STATS_PATH="$OUTPUT_DIR/norm_stats.json"
export TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard"

if [[ ! -f "$NORM_STATS_PATH" ]]; then
  python scripts/compute_norm_stats.py \
    --repo-id lerobot/libero_spatial_image \
    --output-path "$NORM_STATS_PATH" \
    --batch-size 64 \
    --num-workers 8
fi

python -m verl_vla.entrypoints.train.sft \
  "hydra.run.dir=$OUTPUT_DIR/hydra" \
  cluster.actor_rollout_ref.model.path=Miical/pi05-base \
  +cluster.actor_rollout_ref.model.override_config.n_action_steps=10 \
  cluster.actor_rollout_ref.model.adapter.embodiment=libero \
  "cluster.actor_rollout_ref.model.adapter.norm_stats_path=$NORM_STATS_PATH" \
  cluster.actor_rollout_ref.model.adapter.critic.enabled=False \
  cluster.actor_rollout_ref.actor.mini_batch_size=256 \
  cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
  cluster.actor_rollout_ref.actor.optim.weight_decay=1e-5 \
  cluster.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  cluster.actor_rollout_ref.actor.optim.total_training_steps=5000 \
  cluster.resource.model.gpus_per_node=8 \
  "cluster.checkpoint.default_local_dir=$OUTPUT_DIR" \
  cluster.checkpoint.max_actor_ckpt_to_keep=10 \
  data.repo_id=lerobot/libero_spatial_image \
  data.batch_size=256 \
  data.num_workers=8 \
  data.action_delta_steps=10 \
  trainer.total_epochs=25 \
  trainer.save_freq=500 \
  trainer.save_last=True \
  trainer.project_name=pi05-libero-sft \
  trainer.experiment_name=pi05_libero_spatial_sft \
  "trainer.logger=[console,tensorboard]" \
  "$@"
