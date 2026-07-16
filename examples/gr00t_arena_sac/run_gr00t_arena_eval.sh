#!/usr/bin/env bash
#
# Run GR00T N1.6 policy EVALUATION on an Arena task via the RECAP policy_eval
# stage, saving rollout videos + eval metrics to disk. No teleop / no dataset
# recording / no SAC training.
#
# GR00T counterpart of run_pi05_arena_g1_eval.sh: same main_recap policy_eval
# path, but with the gr00t model override and an Arena simulator. Pick the task
# with ARENA_TASK:
#
#   ARENA_TASK=gr1     (default)  GR1 fridge (put_item_in_fridge_and_close_door),
#                                 gr1_joint 26-DOF, embodiment_tag=gr1.
#   ARENA_TASK=libero             Franka Abs-IK LIBERO, eef_pose 7-DOF (rel_rotvec),
#                                 embodiment_tag=new_embodiment; task via TASK_SUITE/TASK_ID.
#
# Must run inside the GR00T docker (isaaclab_arena:cuda_gr00t_gn16), NOT
# verl-vla-arena. Launch it from the host with:
#
#   ARENA_TASK=gr1 INNER_SCRIPT=examples/gr00t_arena_sac/run_gr00t_arena_eval.sh \
#     examples/gr00t_arena_sac/run_docker.sh
#
# See README.md for the full path / variable reference.
#
# ─────────────────────────────────────────────────────────────────────────────
# Overridable via env vars
# ─────────────────────────────────────────────────────────────────────────────
#   ARENA_TASK                gr1 | libero                       (default: gr1)
#   GROOT_MODEL_PATH          GR00T export dir (HF-format: config.json + weights)
#   GROOT_EMBODIMENT_TAG      embodiment tag                     (task default)
#   GROOT_EMBODIMENT_ID       projector index                   (task default)
#   ACTION_DIM                real (unpadded) env action width   (task default)
#   TASK_SUITE / TASK_ID      LIBERO suite / task id             (libero only)
#   OUTPUT_ROOT               videos + eval metrics root
#   MAX_EPISODES              episodes to evaluate (Arena GR1 benchmark size is 1)
#   MAX_INTERACTIONS          env_loop interactions per rollout  (task default)
#   NUM_ACTION_CHUNKS         executed action-chunk length (must match training)
#   ARENA_GR1_JOINT_SPACE_DIR gr00t 26/36/54-DOF joint-space YAML dir  (gr1 only)
#   LIBERO_IN_LAB_ROOT        Arena LIBERO assets root inside the container (libero only)
#
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Isaac Sim's wrapped interpreter inside the GR00T docker.
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

ARENA_TASK="${ARENA_TASK:-gr1}"

# ── Common paths / knobs ─────────────────────────────────────────────────────
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-10000}"
MAX_EPISODES="${MAX_EPISODES:-10}"
NUM_ACTION_CHUNKS="${NUM_ACTION_CHUNKS:-16}"

# ── Task-specific defaults ───────────────────────────────────────────────────
# EXTRA_OVERRIDES holds hydra overrides that differ per task.
EXTRA_OVERRIDES=()
case "$ARENA_TASK" in
  gr1)
    GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-gr1}"
    GROOT_EMBODIMENT_ID="${GROOT_EMBODIMENT_ID:-20}"
    ACTION_DIM="${ACTION_DIM:-26}"
    ARENA_SIM="arena_gr1"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_gr1_eval}"
    # 32 interactions × 16 action chunks = 512 env steps (episode_length_s≈10 @ 50 Hz).
    MAX_INTERACTIONS="${MAX_INTERACTIONS:-32}"
    # Joint-space YAMLs live in the Arena GR00T package (mounted at /workspaces/isaaclab_arena).
    export ARENA_GR1_JOINT_SPACE_DIR="${ARENA_GR1_JOINT_SPACE_DIR:-/workspaces/isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1}"
    EXTRA_OVERRIDES+=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.ARENA_GR1_JOINT_SPACE_DIR=$ARENA_GR1_JOINT_SPACE_DIR"
    )
    ;;
  libero)
    GROOT_EMBODIMENT_TAG="${GROOT_EMBODIMENT_TAG:-new_embodiment}"
    GROOT_EMBODIMENT_ID="${GROOT_EMBODIMENT_ID:-10}"
    ACTION_DIM="${ACTION_DIM:-7}"
    ARENA_SIM="arena_libero"
    TASK_SUITE="${TASK_SUITE:-libero_spatial}"
    TASK_ID="${TASK_ID:-3}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_${TASK_SUITE}_task${TASK_ID}_eval}"
    # 10 interactions × 16 action chunks = 160 env steps (matches legacy MAX_EPISODE_STEPS).
    MAX_INTERACTIONS="${MAX_INTERACTIONS:-10}"
    export LIBERO_IN_LAB_ROOT="${LIBERO_IN_LAB_ROOT:-/libero_in_lab}"
    if [[ ! -d "$LIBERO_IN_LAB_ROOT" ]]; then
      echo "[warn] LIBERO_IN_LAB_ROOT='$LIBERO_IN_LAB_ROOT' missing — Arena LIBERO may fail to resolve USD/configs"
    fi
    EXTRA_OVERRIDES+=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_IN_LAB_ROOT=$LIBERO_IN_LAB_ROOT"
      "recap.policy_eval.cluster.env.env_worker.simulator.arena.libero_task_suite=$TASK_SUITE"
      "recap.policy_eval.cluster.env.env_worker.simulator.arena.libero_task_id=$TASK_ID"
    )
    ;;
  *)
    echo "Unknown ARENA_TASK='$ARENA_TASK' (expected: gr1 | libero)" >&2
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/eval_metrics" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# GR00T docker runtime env.
# ─────────────────────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL=INFO
export TORCH_CUDNN_SDPA_ENABLED="${TORCH_CUDNN_SDPA_ENABLED:-0}"
# transformers 4.51.3 (Eagle) lives in /opt/groot_deps -> must be PREPENDED so it
# wins over Isaac Sim's newer bundled transformers.
export PYTHONPATH="/opt/groot_deps:$REPO_ROOT/src:/workspaces/isaaclab_arena:${PYTHONPATH:-}"

# verl / lerobot deps and the cu13 NCCL fix are baked into the image at build
# time (Dockerfile.isaaclab_arena with INSTALL_GROOT=true) — no runtime installs.

# ─────────────────────────────────────────────────────────────────────────────
# main_recap policy_eval launch.
#
# Hydra *group* overrides (nested under recap.policy_eval.cluster):
#   * model/adapter@…=gr00t    -> GR00T Arena adapter (policy_type=arena, …)
#   * model/override@…=gr00t   -> FSDP / processor compatibility fields
#   * env/simulator@…=$ARENA_SIM -> arena_gr1 (GR1 fridge) or arena_libero (Franka)
#
# TrainCluster.eval() calls generate_sequences(..., eval=True), which sets
# Flow-SDE noise_scale=0 for deterministic ODE sampling.
# ─────────────────────────────────────────────────────────────────────────────
"$PYTHON" -m verl_vla.entrypoints.train.recap \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED="0"' \
  "${EXTRA_OVERRIDES[@]}" \
  "recap.policy_eval.enable=true" \
  "recap.collect_data.enable=false" \
  "recap.compute_return.enable=false" \
  "recap.train_value_model.enable=false" \
  "recap.value_infer.enable=false" \
  "recap.train_policy.enable=false" \
  "model/adapter@recap.policy_eval.cluster.actor_rollout_ref.model.adapter=gr00t" \
  "+model/override@recap.policy_eval.cluster.actor_rollout_ref.model.override_config=gr00t" \
  "env/simulator@recap.policy_eval.cluster.env.env_worker.simulator.arena=$ARENA_SIM" \
  "recap.policy_eval.model_path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.max_episodes=$MAX_EPISODES" \
  "recap.policy_eval.result_dir=$OUTPUT_ROOT/eval_metrics" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.tokenizer_path=$GROOT_MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.trust_remote_code=True" \
  "+recap.policy_eval.cluster.actor_rollout_ref.model.load_tokenizer=False" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.adapter.embodiment_tag=$GROOT_EMBODIMENT_TAG" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.adapter.embodiment_id=$GROOT_EMBODIMENT_ID" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.adapter.action_dim=$ACTION_DIM" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.adapter.num_action_chunks=$NUM_ACTION_CHUNKS" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.adapter.critic.enabled=False" \
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
