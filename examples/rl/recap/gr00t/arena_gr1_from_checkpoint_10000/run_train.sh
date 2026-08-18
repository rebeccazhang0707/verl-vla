#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

cd "$REPO_ROOT"

OUTPUT_DIR="${RECAP_OUTPUT_DIR:-./outputs/rl/recap/gr00t/arena-gr1-from-checkpoint-10000}"
INITIAL_POLICY_PATH="${RECAP_INITIAL_POLICY_PATH:-$REPO_ROOT/.data/models/checkpoint-10000}"
RAY_ADDRESS="${RECAP_RAY_ADDRESS:-auto}"
ARENA_ROOT="${ARENA_ROOT:-/workspace/arena}"
export ARENA_GR1_JOINT_SPACE_DIR="${ARENA_GR1_JOINT_SPACE_DIR:-$ARENA_ROOT/isaaclab_arena_gr00t/embodiments/gr1}"

ISAAC_PYTHONPATH="$(/isaac-sim/python.sh -c 'import os, sys; print(os.pathsep.join(path for path in sys.path if path and os.path.isdir(path)))')"
export PYTHONPATH="$REPO_ROOT/src:$ARENA_ROOT:$ARENA_ROOT/submodules/Isaac-GR00T:$ISAAC_PYTHONPATH"

runtime_env=(
  "+ray_kwargs.ray_init.address=$RAY_ADDRESS"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$PYTHONPATH'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.ARENA_GR1_JOINT_SPACE_DIR='$ARENA_GR1_JOINT_SPACE_DIR'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED='0'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.GR00T_COMPAT_PATCHES=all"
  "+ray_kwargs.ray_init.runtime_env.env_vars.NO_ALBUMENTATIONS_UPDATE='1'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME='$REPO_ROOT/.data/gr00t_arena/cache/root_cache/huggingface'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_MODULES_CACHE='$REPO_ROOT/.data/gr00t_arena/cache/root_cache/huggingface/modules'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.LW_API_ENDPOINT='https://api.lightwheel.net'"
)

for name in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
  if [[ -n "${!name:-}" ]]; then
    runtime_env+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name='${!name}'")
  fi
done

exec /isaac-sim/python.sh -m verl_vla.entrypoints.train.recap \
  --config-dir "$SCRIPT_DIR" \
  --config-name arena_gr1 \
  "output_dir=$OUTPUT_DIR" \
  "initial_policy_path=$INITIAL_POLICY_PATH" \
  recap.policy_eval.cluster.resource.controller_label=train_rollout \
  recap.policy_eval.cluster.resource.model.resource_label=train_rollout \
  recap.policy_eval.cluster.resource.model.gpus_per_node=8 \
  recap.policy_eval.cluster.resource.env.resource_label=sim \
  recap.policy_eval.cluster.resource.env.gpus_per_node=4 \
  recap.collect_data.cluster.resource.controller_label=train_rollout \
  recap.collect_data.cluster.resource.model.resource_label=train_rollout \
  recap.collect_data.cluster.resource.model.gpus_per_node=8 \
  recap.collect_data.cluster.resource.env.resource_label=sim \
  recap.collect_data.cluster.resource.env.gpus_per_node=4 \
  recap.train_value_model.cluster.resource.controller_label=train_rollout \
  recap.train_value_model.cluster.resource.model.resource_label=train_rollout \
  recap.train_value_model.cluster.resource.model.gpus_per_node=2 \
  recap.train_value_model.data.batch_size=256 \
  recap.train_value_model.data.num_workers=8 \
  recap.train_value_model.data.prefetch_factor=8 \
  recap.train_value_model.cluster.actor_rollout_ref.actor.mini_batch_size=256 \
  recap.train_value_model.cluster.actor_rollout_ref.actor.micro_batch_size=32 \
  recap.value_infer.num_gpus=2 \
  recap.value_infer.data.batch_size=64 \
  recap.value_infer.data.num_workers=8 \
  recap.value_infer.data.prefetch_factor=8 \
  recap.train_policy.cluster.resource.controller_label=train_rollout \
  recap.train_policy.cluster.resource.model.resource_label=train_rollout \
  recap.train_policy.cluster.resource.model.gpus_per_node=8 \
  recap.train_policy.data.batch_size=256 \
  recap.train_policy.data.num_workers=8 \
  recap.train_policy.data.prefetch_factor=8 \
  recap.train_policy.cluster.actor_rollout_ref.actor.mini_batch_size=256 \
  recap.train_policy.cluster.actor_rollout_ref.actor.micro_batch_size=16 \
  "ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard" \
  "${runtime_env[@]}" \
  "$@"
