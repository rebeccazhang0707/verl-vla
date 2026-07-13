#!/usr/bin/env bash
#
# Run GR00T N1.6 policy EVALUATION on Arena LIBERO spatial task 3
# (Franka Abs-IK, new_embodiment) via the RECAP policy_eval stage,
# saving rollout videos + eval metrics to disk. No teleop / no SAC training.
#
# Counterpart of run_gr00t_arena_gr1_eval.sh, but with:
#   * env/simulator@…=arena_libero  (eef_pose, agentview + eye-in-hand)
#   * embodiment_tag=new_embodiment, action_dim=7 (rel_rotvec), embodiment_id=10
#   * libero_task_suite=libero_spatial, libero_task_id=3
#
# ─────────────────────────────────────────────────────────────────────────────
# GR00T DOCKER (required — isaaclab_arena:cuda_gr00t_gn16)
# ─────────────────────────────────────────────────────────────────────────────
# From the host:
#
#   MODELS_HOST=~/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec \
#   GROOT_MODEL_PATH=/models/checkpoint-5000 \
#   EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_libero_spatial_task3_eval.sh \
#     examples/arena_sac/run_docker_gr00t_eval.sh
#
# Mount mapping (via run_docker_gr00t_eval.sh):
#   /models                      <- checkpoint parent (libero_all_suites_rel_rotvec)
#   /eval                        <- this verl-vla repo
#   /workspaces/isaaclab_arena   <- host IsaacLab-Arena checkout
#   /libero_in_lab               <- host libero_in_lab (USD / configs)
#
# Overridable via env vars:
#   GROOT_MODEL_PATH          GR00T export dir
#   GROOT_EMBODIMENT_TAG      default: new_embodiment
#   GROOT_EMBODIMENT_ID       default: 10
#   ACTION_DIM                default: 7  (pos+rotvec+gripper)
#   TASK_SUITE / TASK_ID      default: libero_spatial / 3
#   OUTPUT_ROOT               videos + eval metrics root
#   MAX_EPISODES              episodes to evaluate
#   MAX_INTERACTIONS          env_loop interactions (10 × 16 chunks ≈ 160 env steps)
#   NUM_ACTION_CHUNKS         executed action-chunk length
#   LIBERO_IN_LAB_ROOT        Arena LIBERO assets root inside the container
#
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/isaac-sim/python.sh}"

GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-5000}"
GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-new_embodiment}"
GROOT_EMBODIMENT_ID="${GROOT_EMBODIMENT_ID:-10}"
ACTION_DIM="${ACTION_DIM:-7}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_libero_spatial_task3_eval}"
MAX_EPISODES="${MAX_EPISODES:-10}"
# 10 interactions × 16 action chunks = 160 env steps (matches legacy MAX_EPISODE_STEPS).
MAX_INTERACTIONS="${MAX_INTERACTIONS:-10}"
NUM_ACTION_CHUNKS="${NUM_ACTION_CHUNKS:-16}"
export LIBERO_IN_LAB_ROOT="${LIBERO_IN_LAB_ROOT:-/libero_in_lab}"

mkdir -p "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/eval_metrics" 2>/dev/null || true

export VERL_LOGGING_LEVEL=INFO
export TORCH_CUDNN_SDPA_ENABLED="${TORCH_CUDNN_SDPA_ENABLED:-0}"
export PYTHONPATH="/opt/groot_deps:$REPO_ROOT/src:/workspaces/isaaclab_arena:${PYTHONPATH:-}"

if [[ -d /opt/cuda128-compat ]]; then
  export LD_LIBRARY_PATH="/opt/cuda128-compat:${LD_LIBRARY_PATH:-}"
fi

# Same verl bootstrap as the GR1 eval (image has no verl; pin torch/transformers/numpy).
if ! "$PYTHON" -c "import verl, datasets, torchdata, codetiming" >/dev/null 2>&1; then
  echo "[deps] installing verl==0.7.1 (--no-deps) + missing deps; pin torch/transformers/numpy"
  CONSTRAINTS_FILE="$OUTPUT_ROOT/verl_constraints.txt"
  printf 'torch==%s\ntransformers==4.51.3\nnumpy==%s\n' \
    "$("$PYTHON" -c 'import torch;print(torch.__version__)')" \
    "$("$PYTHON" -c 'import numpy;print(numpy.__version__)')" > "$CONSTRAINTS_FILE"
  pip_install() {
    sudo "$PYTHON" -m pip install -q "$@" || "$PYTHON" -m pip install -q "$@"
  }
  pip_install --no-deps "verl==0.7.1"
  pip_install -c "$CONSTRAINTS_FILE" \
    datasets torchdata codetiming dill pybind11 pylatexenc
fi

if [[ -e /opt/groot_deps/nvidia/nccl/lib/libnccl.so.2 ]]; then
  echo "[nccl] disabling stray cu13 NCCL in /opt/groot_deps (keep torch cu12)"
  mv /opt/groot_deps/nvidia/nccl /opt/groot_deps/nvidia/nccl.cu13-disabled 2>/dev/null \
    || sudo mv /opt/groot_deps/nvidia/nccl /opt/groot_deps/nvidia/nccl.cu13-disabled || true
fi

if [[ ! -d "$LIBERO_IN_LAB_ROOT" ]]; then
  echo "[warn] LIBERO_IN_LAB_ROOT='$LIBERO_IN_LAB_ROOT' missing — Arena LIBERO may fail to resolve USD/configs"
fi

"$PYTHON" -m verl_vla.entrypoints.train.recap \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED="0"' \
  "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_IN_LAB_ROOT=$LIBERO_IN_LAB_ROOT" \
  "recap.policy_eval.enable=true" \
  "recap.collect_data.enable=false" \
  "recap.compute_return.enable=false" \
  "recap.train_value_model.enable=false" \
  "recap.value_infer.enable=false" \
  "recap.train_policy.enable=false" \
  "model/override@recap.policy_eval.cluster.actor_rollout_ref.model.override_config=gr00t" \
  "env/simulator@recap.policy_eval.cluster.env.env_worker.simulator.arena=arena_libero" \
  "recap.policy_eval.model_path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.max_episodes=$MAX_EPISODES" \
  "recap.policy_eval.result_dir=$OUTPUT_ROOT/eval_metrics" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.tokenizer_path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.trust_remote_code=True" \
  "+recap.policy_eval.cluster.actor_rollout_ref.model.load_tokenizer=False" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.embodiment_tag=$GROOT_EMBODIMENT_TAG" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.embodiment_id=$GROOT_EMBODIMENT_ID" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.action_dim=$ACTION_DIM" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.num_action_chunks=$NUM_ACTION_CHUNKS" \
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.name=hf" \
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.output_critic_value=false" \
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "recap.policy_eval.cluster.env.env_loop.max_interactions=$MAX_INTERACTIONS" \
  "recap.policy_eval.cluster.env.env_worker.auto_reset=true" \
  "recap.policy_eval.cluster.env.env_worker.simulator_start_timeout_s=600" \
  "recap.policy_eval.cluster.env.env_worker.simulator.simulator_type=arena" \
  "recap.policy_eval.cluster.env.env_worker.simulator.arena.libero_task_suite=$TASK_SUITE" \
  "recap.policy_eval.cluster.env.env_worker.simulator.arena.libero_task_id=$TASK_ID" \
  "recap.policy_eval.cluster.env.env_worker.modes=[eval]" \
  "recap.policy_eval.cluster.env.env_worker.teleop.enable=false" \
  "recap.policy_eval.cluster.env.env_worker.recorder.enable=true" \
  "recap.policy_eval.cluster.env.env_worker.recorder.recorders=[video]" \
  "recap.policy_eval.cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos" \
  "$@"
