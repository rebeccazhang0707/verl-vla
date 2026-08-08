#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="$PWD/outputs/train/pi05-sft/libero-spatial"

STEP="$(<"$CHECKPOINT_DIR/latest_checkpointed_iteration.txt")"
MODEL_PATH="$CHECKPOINT_DIR/global_step_${STEP}/actor/huggingface"
OUTPUT_DIR="$PWD/outputs/eval/pi05-sft/libero-spatial/step-${STEP}"

exec vvla-eval \
  "hydra.run.dir=$OUTPUT_DIR/hydra" \
  "cluster.actor_rollout_ref.model.path=$MODEL_PATH" \
  cluster.actor_rollout_ref.model.adapter.embodiment=libero \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids=null \
  cluster.env.env_worker.simulator.libero.num_trials_per_task=10 \
  cluster.env.env_worker.simulator.libero.max_episode_steps=256 \
  cluster.env.env_loop.max_interactions=8 \
  "cluster.env.env_worker.recorder.video.root=$OUTPUT_DIR/videos" \
  cluster.resource.model.gpus_per_node=2 \
  cluster.resource.env.device=cpu \
  cluster.resource.env.workers_per_node=8 \
  cluster.env.env_worker.num_envs=2 \
  max_episodes=null \
  "output_dir=$OUTPUT_DIR" \
  "$@"
