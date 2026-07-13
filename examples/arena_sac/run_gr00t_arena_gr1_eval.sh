#!/usr/bin/env bash
#
# Run GR00T N1.6 policy EVALUATION on the Arena GR1 fridge task
# (put_item_in_fridge_and_close_door) via the RECAP policy_eval stage,
# saving rollout videos + eval metrics to disk. No teleop / no dataset
# recording / no SAC training.
#
# This is the GR00T counterpart of run_pi05_arena_g1_eval.sh: same
# main_recap policy_eval path, but with the gr00t model override and the
# arena_gr1 simulator (gr1_joint, 26-DOF).
#
# ─────────────────────────────────────────────────────────────────────────────
# GR00T DOCKER (required — isaaclab_arena:cuda_gr00t_gn16, NOT verl-vla-arena)
# ─────────────────────────────────────────────────────────────────────────────
# From the host (preferred helper):
#
#   MODELS_HOST=~/Projects/libero_rl_example/checkpoints/gr1_ranch_bottle_into_fridge \
#     examples/arena_sac/run_docker_gr00t_eval.sh
#
# Or start via IsaacLab-Arena (see _libero_rl_example/README.md §1), then exec:
#
#   cd ~/Projects/libero_rl_example/IsaacLab-Arena
#   bash docker/run_docker.sh -g \
#     -m ~/Projects/libero_rl_example/checkpoints/gr1_ranch_bottle_into_fridge \
#     -e ~/Projects/verl-vla
#   GROOT_MODEL_PATH=/models/checkpoint-5000-export \
#     bash /eval/examples/arena_sac/run_gr00t_arena_gr1_eval.sh
#
# Mount mapping inside the container (via run_docker_gr00t_eval.sh):
#   /models                      <- checkpoint parent
#   /eval                        <- this verl-vla repo
#   /workspaces/isaaclab_arena   <- host IsaacLab-Arena checkout (ARENA_HOST)
#
# Overridable via env vars:
#   GROOT_MODEL_PATH          GR00T export dir (HF-format: config.json + weights + …)
#   GROOT_EMBODIMENT_TAG      embodiment tag (default: gr1)
#   ARENA_GR1_JOINT_SPACE_DIR dir with gr00t_26dof / 36dof / 54dof joint-space YAMLs
#   OUTPUT_ROOT               videos + eval metrics root
#   MAX_EPISODES              episodes to evaluate (Arena GR1 benchmark size is 1;
#                             set explicitly to eval more than 1)
#   MAX_INTERACTIONS          env_loop interactions per rollout
#                             (32 × num_action_chunks=16 ≈ 512 env steps ≈ 10 s @ 50 Hz)
#   NUM_ACTION_CHUNKS         executed action-chunk length (must match training)
#
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── Python: Isaac Sim's wrapped interpreter inside the GR00T docker ──────────────
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

# ── Paths (override to match your mounts) ───────────────────────────────────────
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-5000}"
GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-gr1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_gr1_eval}"
MAX_EPISODES="${MAX_EPISODES:-10}"
# 32 interactions × 16 action chunks = 512 env steps (matches task episode_length_s≈10).
MAX_INTERACTIONS="${MAX_INTERACTIONS:-32}"
NUM_ACTION_CHUNKS="${NUM_ACTION_CHUNKS:-16}"

# Joint-space YAMLs live in the Arena GR00T package (baked into the image at
# /workspaces/isaaclab_arena). Override if your layout differs.
export ARENA_GR1_JOINT_SPACE_DIR="${ARENA_GR1_JOINT_SPACE_DIR:-/workspaces/isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1}"

mkdir -p "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/eval_metrics" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# GR00T docker runtime env (mirrors run_gr00t_arena_gr1_sac_smoke.sh).
# ─────────────────────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL=INFO
export TORCH_CUDNN_SDPA_ENABLED="${TORCH_CUDNN_SDPA_ENABLED:-0}"

# transformers 4.51.3 (Eagle) lives in /opt/groot_deps -> must be PREPENDED so it
# wins over Isaac Sim's newer bundled transformers.
export PYTHONPATH="/opt/groot_deps:$REPO_ROOT/src:/workspaces/isaaclab_arena:${PYTHONPATH:-}"

if [[ -d /opt/cuda128-compat ]]; then
  export LD_LIBRARY_PATH="/opt/cuda128-compat:${LD_LIBRARY_PATH:-}"
fi

# The Arena GR00T image has no verl / a few verl-only deps (unlike verl-vla-arena).
# verl==0.7.1 declares numpy<2.0, but this image ships numpy 2.x for GR00T — so
# install verl with --no-deps and pull only the missing lightweight deps. Pin
# torch/transformers/numpy so pip cannot upgrade Eagle. Constraints file lives
# under OUTPUT_ROOT (host /tmp bind-mounts often have unwritable leftovers).
# lerobot is NOT required for eval (teleop.devices imports it lazily).
if ! "$PYTHON" -c "import verl, datasets, torchdata, codetiming" >/dev/null 2>&1; then
  echo "[deps] installing verl==0.7.1 (--no-deps) + missing deps; pin torch/transformers/numpy"
  CONSTRAINTS_FILE="$OUTPUT_ROOT/verl_constraints.txt"
  printf 'torch==%s\ntransformers==4.51.3\nnumpy==%s\n' \
    "$("$PYTHON" -c 'import torch;print(torch.__version__)')" \
    "$("$PYTHON" -c 'import numpy;print(numpy.__version__)')" > "$CONSTRAINTS_FILE"
  pip_install() {
    sudo "$PYTHON" -m pip install -q "$@" || "$PYTHON" -m pip install -q "$@"
  }
  # Core package only — skip its numpy<2 resolver conflict with the image numpy.
  pip_install --no-deps "verl==0.7.1"
  # Missing runtime deps (already present in-image: accelerate/hydra/ray/…).
  pip_install -c "$CONSTRAINTS_FILE" \
    datasets torchdata codetiming dill pybind11 pylatexenc
fi

# Drop the stray CUDA-13 NCCL bundled in /opt/groot_deps so `import nvidia.nccl`
# falls through to Isaac Sim's cu12 NCCL.
if [[ -e /opt/groot_deps/nvidia/nccl/lib/libnccl.so.2 ]]; then
  echo "[nccl] disabling stray cu13 NCCL in /opt/groot_deps (keep torch cu12)"
  mv /opt/groot_deps/nvidia/nccl /opt/groot_deps/nvidia/nccl.cu13-disabled 2>/dev/null \
    || sudo mv /opt/groot_deps/nvidia/nccl /opt/groot_deps/nvidia/nccl.cu13-disabled || true
fi

# ─────────────────────────────────────────────────────────────────────────────
# main_recap policy_eval launch.
#
# Hydra *group* overrides (same pattern as the SAC smoke, but nested under
# recap.policy_eval.cluster):
#   * model/override@…=gr00t  -> GR00T SAC override (policy_type=arena, action_dim=26, …)
#   * env/simulator@…=arena_gr1 -> GR1 fridge sim (gr1_joint, cameras, joint-space YAMLs)
#
# TrainCluster.eval() calls generate_sequences(..., eval=True), which sets
# Flow-SDE noise_scale=0 for deterministic ODE sampling.
# ─────────────────────────────────────────────────────────────────────────────
"$PYTHON" -m verl_vla.entrypoints.train.recap \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED="0"' \
  "+ray_kwargs.ray_init.runtime_env.env_vars.ARENA_GR1_JOINT_SPACE_DIR=$ARENA_GR1_JOINT_SPACE_DIR" \
  "recap.policy_eval.enable=true" \
  "recap.collect_data.enable=false" \
  "recap.compute_return.enable=false" \
  "recap.train_value_model.enable=false" \
  "recap.value_infer.enable=false" \
  "recap.train_policy.enable=false" \
  "model/override@recap.policy_eval.cluster.actor_rollout_ref.model.override_config=gr00t" \
  "env/simulator@recap.policy_eval.cluster.env.env_worker.simulator.arena=arena_gr1" \
  "recap.policy_eval.model_path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.max_episodes=$MAX_EPISODES" \
  "recap.policy_eval.result_dir=$OUTPUT_ROOT/eval_metrics" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.tokenizer_path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.trust_remote_code=True" \
  "+recap.policy_eval.cluster.actor_rollout_ref.model.load_tokenizer=False" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.embodiment_tag=$GROOT_EMBODIMENT_TAG" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.num_action_chunks=$NUM_ACTION_CHUNKS" \
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.name=hf" \
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.output_critic_value=false" \
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "recap.policy_eval.cluster.env.env_loop.max_interactions=$MAX_INTERACTIONS" \
  "recap.policy_eval.cluster.env.env_worker.auto_reset=true" \
  "recap.policy_eval.cluster.env.env_worker.simulator_start_timeout_s=600" \
  "recap.policy_eval.cluster.env.env_worker.simulator.simulator_type=arena" \
  "recap.policy_eval.cluster.env.env_worker.modes=[eval]" \
  "recap.policy_eval.cluster.env.env_worker.teleop.enable=false" \
  "recap.policy_eval.cluster.env.env_worker.recorder.enable=true" \
  "recap.policy_eval.cluster.env.env_worker.recorder.recorders=[video]" \
  "recap.policy_eval.cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos" \
  "$@"
