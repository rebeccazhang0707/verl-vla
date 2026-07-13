#!/usr/bin/env bash
set -euo pipefail

export TENSORBOARD_DIR=/workspace/project/.data/pi05_sft/output/pi05_libero_spatial_sft/tensorboard

python3 -m verl_vla.entrypoints.train.sft \
  hydra.run.dir=.data/pi05_sft/output/pi05_libero_spatial_sft/hydra \
  cluster.actor_rollout_ref.model.path=Miical/pi05-base \
  cluster.actor_rollout_ref.model.adapter.embodiment=libero \
  cluster.actor_rollout_ref.model.adapter.norm_stats_path=/workspace/project/.data/pi05_sft/huggingface/lerobot/lerobot/libero_spatial_image/norm_stats.json \
  cluster.actor_rollout_ref.model.adapter.critic.enabled=False \
  cluster.actor_rollout_ref.actor.mini_batch_size=64 \
  cluster.actor_rollout_ref.actor.micro_batch_size=8 \
  cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
  cluster.actor_rollout_ref.actor.optim.weight_decay=1e-5 \
  cluster.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  cluster.resource.model.gpus_per_node=8 \
  cluster.checkpoint.default_local_dir=.data/pi05_sft/output/pi05_libero_spatial_sft \
  cluster.checkpoint.max_actor_ckpt_to_keep=3 \
  data.repo_id=lerobot/libero_spatial_image \
  data.batch_size=64 \
  data.num_workers=8 \
  data.action_delta_steps=50 \
  trainer.total_epochs=10 \
  trainer.save_freq=500 \
  trainer.save_last=True \
  trainer.project_name=pi05-libero-sft \
  trainer.experiment_name=pi05_libero_spatial_sft \
  "trainer.logger=[console,tensorboard]" \
  "$@"
