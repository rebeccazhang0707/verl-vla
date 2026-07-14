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

    export TENSORBOARD_DIR=/workspace/project/.data/pi05_sft/output/pi05_libero_spatial_sft/tensorboard

    exec python3 -m verl_vla.entrypoints.train.sft \
      hydra.run.dir=.data/pi05_sft/output/pi05_libero_spatial_sft/hydra \
      cluster.actor_rollout_ref.model.path=Miical/pi05-base \
      +cluster.actor_rollout_ref.model.override_config.n_action_steps=10 \
      cluster.actor_rollout_ref.model.adapter.embodiment=libero \
      cluster.actor_rollout_ref.model.adapter.norm_stats_path=/workspace/project/.data/pi05_sft/huggingface/lerobot/lerobot/libero_spatial_image/norm_stats.json \
      cluster.actor_rollout_ref.model.adapter.critic.enabled=False \
      cluster.actor_rollout_ref.actor.mini_batch_size=256 \
      cluster.actor_rollout_ref.actor.micro_batch_size=16 \
      cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
      cluster.actor_rollout_ref.actor.optim.weight_decay=1e-5 \
      cluster.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
      cluster.actor_rollout_ref.actor.optim.total_training_steps=5000 \
      cluster.resource.model.gpus_per_node=8 \
      cluster.checkpoint.default_local_dir=.data/pi05_sft/output/pi05_libero_spatial_sft \
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
  ' -- "$@"
