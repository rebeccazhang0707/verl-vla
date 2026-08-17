#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STEP="${STEP:-1500}"
MODEL_PATH="./outputs/train/gr00t-sft/libero-spatial/checkpoints/global_step_${STEP}/actor/huggingface"
OUTPUT_DIR="./outputs/eval/gr00t-sft/libero-spatial/step-${STEP}"
LIBERO_CONFIG_PATH="${REPO_ROOT}/.data/gr00t_sft/libero_config"

cd "$REPO_ROOT"

LIBERO_ROOT="$(python -c 'import importlib.util; from pathlib import Path; root = Path(importlib.util.find_spec("libero").origin).parent; print(root / "libero" if (root / "libero" / "bddl_files").is_dir() else root)')"
mkdir -p "$LIBERO_CONFIG_PATH"
printf '%s\n' \
  "benchmark_root: ${LIBERO_ROOT}" \
  "bddl_files: ${LIBERO_ROOT}/bddl_files" \
  "init_states: ${LIBERO_ROOT}/init_files" \
  "datasets: ${REPO_ROOT}/.data/gr00t_sft/datasets" \
  "assets: ${LIBERO_ROOT}/assets" \
  > "${LIBERO_CONFIG_PATH}/config.yaml"

export LIBERO_CONFIG_PATH

vvla-eval \
  model/adapter@cluster.actor_rollout_ref.model.adapter=gr00t \
  model/override@cluster.actor_rollout_ref.model.override_config=gr00t \
  cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
  cluster.actor_rollout_ref.model.load_tokenizer=false \
  cluster.actor_rollout_ref.model.trust_remote_code=true \
  cluster.actor_rollout_ref.model.adapter.policy_type=libero \
  cluster.actor_rollout_ref.model.adapter.embodiment_tag=libero_panda \
  cluster.actor_rollout_ref.model.adapter.action_dim=7 \
  cluster.actor_rollout_ref.model.adapter.embodiment_id=2 \
  cluster.actor_rollout_ref.model.adapter.critic.enabled=false \
  cluster.actor_rollout_ref.model.adapter.override_modality_configs=true \
  cluster.actor_rollout_ref.model.adapter.use_relative_action=true \
  cluster.actor_rollout_ref.model.adapter.num_action_chunks=8 \
  cluster.actor_rollout_ref.rollout.name=hf \
  cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
  cluster.env.env_worker.simulator.libero.task_ids=null \
  cluster.env.env_worker.simulator.libero.num_trials_per_task=10 \
  cluster.env.env_worker.simulator.libero.max_episode_steps=720 \
  cluster.env.env_loop.max_interactions=8 \
  cluster.resource.model.gpus_per_node=2 \
  cluster.resource.env.device=cpu \
  cluster.resource.env.workers_per_node=8 \
  cluster.env.env_worker.num_envs=2 \
  ray_kwargs.ray_init.runtime_env.env_vars.MUJOCO_GL=osmesa \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYOPENGL_PLATFORM=osmesa \
  output_dir="$OUTPUT_DIR" \
  "$@"
