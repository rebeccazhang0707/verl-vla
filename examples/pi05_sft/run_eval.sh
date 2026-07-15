#!/usr/bin/env bash
set -e

exec docker run --rm -it \
  --gpus all \
  --ipc=host \
  --entrypoint /bin/bash \
  -v "$PWD:/workspace/project" \
  -w /workspace/project \
  -e PYTHONPATH=/workspace/project/src \
  verl-vla-pi0:dev \
  -c '
    set -e
    python3 -m pip install --no-deps -e .

    STEP=$(cat .data/pi05_sft/output/pi05_libero_spatial_sft/latest_checkpointed_iteration.txt)
    MODEL_PATH=/workspace/project/.data/pi05_sft/output/pi05_libero_spatial_sft/global_step_${STEP}/actor/huggingface
    OUTPUT_DIR=/workspace/project/.data/pi05_eval/output/pi05_libero_spatial_step_${STEP}

    exec vvla-eval \
      hydra.run.dir="$OUTPUT_DIR/hydra" \
      cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
      cluster.actor_rollout_ref.model.adapter.embodiment=libero \
      cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
      cluster.env.env_worker.simulator.libero.task_ids=null \
      cluster.env.env_worker.simulator.libero.num_trials_per_task=10 \
      cluster.env.env_worker.simulator.libero.max_episode_steps=256 \
      cluster.env.env_loop.max_interactions=8 \
      cluster.env.env_worker.recorder.video.root="$OUTPUT_DIR/videos" \
      cluster.resource.model.gpus_per_node=2 \
      cluster.resource.env.device=cpu \
      cluster.resource.env.workers_per_node=8 \
      cluster.env.env_worker.num_envs=2 \
      max_episodes=null \
      output_dir="$OUTPUT_DIR"
  '
