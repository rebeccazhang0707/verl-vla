#!/usr/bin/env bash
#
# Run GR00T N1.6 SAC training on an Arena task via verl_vla.entrypoints.train.sac.
#
# Counterpart of run_gr00t_arena_eval.sh (same docker / Hydra group overrides),
# but launches the SAC trainer instead of RECAP policy_eval. Pick the task with
# ARENA_TASK:
#
#   ARENA_TASK=gr1     (default)  GR1 fridge (put_item_in_fridge_and_close_door),
#                                 gr1_joint 26-DOF, embodiment_tag=gr1.
#   ARENA_TASK=libero             Franka Abs-IK LIBERO, eef_pose 7-DOF (rel_rotvec),
#                                 embodiment_tag=new_embodiment; task via TASK_SUITE/TASK_ID.
#
# Minimal SAC launch: uses gr00t.yaml defaults for critic / Flow-SDE / freeze
# knobs (single shared cross_attn critic — Arena launchers pin one task id, so
# multitask multi_cross_attn is not required). Extra Hydra overrides via "$@".
#
# Must run inside the GR00T docker (isaaclab_arena:cuda_gr00t_gn16). Launch from
# the host with:
#
#   ARENA_TASK=gr1 INNER_SCRIPT=examples/gr00t_arena_sac/run_gr00t_arena_sac.sh \
#     OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_sac \
#     examples/gr00t_arena_sac/run_docker.sh
#
# See README.md for the full path / variable reference.
#
# ─────────────────────────────────────────────────────────────────────────────
# Overridable via env vars (see knobs below). Extra Hydra overrides: "$@"
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/isaac-sim/python.sh}"

ARENA_TASK="${ARENA_TASK:-gr1}"

# ── Common paths ─────────────────────────────────────────────────────────────
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-10000}"
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
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_gr1_sac}"
    PROJECT_NAME="${PROJECT_NAME:-gr00t-arena-gr1-sac}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-arena_gr00t_gr1_fridge}"
    # 32 interactions × 16 action chunks = 512 env steps (episode_length_s≈10 @ 50 Hz).
    MAX_INTERACTIONS="${MAX_INTERACTIONS:-32}"
    export ARENA_GR1_JOINT_SPACE_DIR="${ARENA_GR1_JOINT_SPACE_DIR:-/workspaces/isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1}"
    EXTRA_RAY_ENV=(
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
    OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_gr00t_libero_sac}"
    PROJECT_NAME="${PROJECT_NAME:-gr00t-arena-libero-sac}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-arena_gr00t_libero_${TASK_SUITE}_task${TASK_ID}}"
    # 10 interactions × 16 action chunks = 160 env steps (matches LIBERO eval default).
    MAX_INTERACTIONS="${MAX_INTERACTIONS:-10}"
    # Episodes run up to max_episode_steps=512 env steps but a rollout window is only
    # 160, so episodes span 3-4 windows. Collect them episodically so the early/middle
    # transitions are not dropped (docs/reinforcement-learning/episodic-replay.md).
    EPISODIC_REPLAY="${EPISODIC_REPLAY:-True}"
    # LIBERO data resolution is split: the task-suite JSON configs come from Arena's
    # colocated copy (external_environments/libero/data/config, kept in sync with the env
    # code), while the heavy USD assets + assembled HDF5 demos come from the
    # LIBERO_IN_LAB_ROOT mount. We pre-set LIBERO_CONFIG_DIR so Arena's
    # _configure_libero_env_vars() setdefault does not override it with
    # $LIBERO_IN_LAB_ROOT/.../config (USD/hdf5 still resolve under LIBERO_IN_LAB_ROOT).
    export LIBERO_IN_LAB_ROOT="${LIBERO_IN_LAB_ROOT:-/libero_in_lab}"
    if [[ ! -d "$LIBERO_IN_LAB_ROOT" ]]; then
      echo "[warn] LIBERO_IN_LAB_ROOT='$LIBERO_IN_LAB_ROOT' missing — Arena LIBERO may fail to resolve USD/hdf5"
    fi
    ARENA_LIBERO_DATA_DIR="${ARENA_LIBERO_DATA_DIR:-/workspaces/isaaclab_arena/isaaclab_arena_examples/external_environments/libero/data}"
    export LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-$ARENA_LIBERO_DATA_DIR/config}"
    EXTRA_RAY_ENV=(
      "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_IN_LAB_ROOT=$LIBERO_IN_LAB_ROOT"
      "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_CONFIG_DIR=$LIBERO_CONFIG_DIR"
    )
    EXTRA_OVERRIDES+=(
      "cluster.env.env_worker.simulator.arena.libero_task_suite=$TASK_SUITE"
      "cluster.env.env_worker.simulator.arena.libero_task_id=$TASK_ID"
    )
    ;;
  *)
    echo "Unknown ARENA_TASK='$ARENA_TASK' (expected: gr1 | libero)" >&2
    exit 1
    ;;
esac

# ── Experiment identity ──────────────────────────────────────────────────────
REPLAY_POOL_DIR="${REPLAY_POOL_DIR:-$OUTPUT_ROOT/replay_pools}"

# ── Topology (Ray resource pools under cluster.resource) ─────────────────────
# Default: co-located 1 env worker + 1 model worker, 8 Isaac envs per env GPU.
# Scale workers with NUM_ENV_GPUS / NUM_MODEL_GPUS; override NUM_ENV for denser sims.
NUM_NODES="${NUM_NODES:-1}"
NUM_ENV_GPUS="${NUM_ENV_GPUS:-4}"
NUM_MODEL_GPUS="${NUM_MODEL_GPUS:-4}"
NUM_ENV="${NUM_ENV:-8}"
NUM_STAGE="${NUM_STAGE:-2}"

# ── SAC batch / schedule ─────────────────────────────────────────────────────
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
ROLLOUT_INTERVAL="${ROLLOUT_INTERVAL:-20}"
WARM_ROLLOUT_STEPS="${WARM_ROLLOUT_STEPS:-5}"
CRITIC_WARMUP_STEPS="${CRITIC_WARMUP_STEPS:-200}"
ACTOR_UPDATE_INTERVAL="${ACTOR_UPDATE_INTERVAL:-1}"
SAVE_FREQ="${SAVE_FREQ:-500}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

# ── Episodic replay collection (requires auto_reset=true, which this script sets) ──
# Task branches may set a task-specific default above (libero: True); fall back off.
EPISODIC_REPLAY="${EPISODIC_REPLAY:-False}"
EPISODIC_MAX_OPEN_LEN="${EPISODIC_MAX_OPEN_LEN:-128}"

# ── SAC entropy / critic tau ─────────────────────────────────────────────────
INITIAL_ALPHA="${INITIAL_ALPHA:-0.01}"
ALPHA_TYPE="${ALPHA_TYPE:-softplus}"
AUTO_ENTROPY="${AUTO_ENTROPY:-False}"
CRITIC_TAU="${CRITIC_TAU:-0.01}"

# ── Logging ──────────────────────────────────────────────────────────────────
TRAINER_LOGGER="${TRAINER_LOGGER:-[console]}"

mkdir -p "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/checkpoints" "$REPLAY_POOL_DIR" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# GR00T docker runtime env.
# ─────────────────────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL=INFO
export TORCH_CUDNN_SDPA_ENABLED="${TORCH_CUDNN_SDPA_ENABLED:-0}"
export PYTHONPATH="/opt/groot_deps:$REPO_ROOT/src:/workspaces/isaaclab_arena:${PYTHONPATH:-}"
# Do NOT set RAY_ADDRESS=auto here: that makes Ray try to attach to an existing
# cluster. Collocated single-node runs should let ensure_ray_initialized() start
# a local cluster via ray.init().

# Same verl bootstrap as the eval script (image has no verl; pin torch/transformers/numpy).
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

# ─────────────────────────────────────────────────────────────────────────────
# main_sac launch.
#
# Hydra *group* overrides:
#   * model/adapter@…=gr00t      -> GR00T SAC adapter (critic / Flow-SDE / policy_type)
#   * model/override@…=gr00t     -> FSDP / processor compatibility fields
#   * env/simulator@…=$ARENA_SIM -> arena_gr1 (GR1 fridge) or arena_libero (Franka)
# ─────────────────────────────────────────────────────────────────────────────
"$PYTHON" -m verl_vla.entrypoints.train.sac \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED="0"' \
  "${EXTRA_RAY_ENV[@]}" \
  "model/adapter@cluster.actor_rollout_ref.model.adapter=gr00t" \
  "+model/override@cluster.actor_rollout_ref.model.override_config=gr00t" \
  "env/simulator@cluster.env.env_worker.simulator.arena=$ARENA_SIM" \
  "cluster.actor_rollout_ref.model.path=$GROOT_MODEL_PATH" \
  "cluster.actor_rollout_ref.model.tokenizer_path=$GROOT_MODEL_PATH" \
  "cluster.actor_rollout_ref.model.trust_remote_code=True" \
  "+cluster.actor_rollout_ref.model.load_tokenizer=False" \
  "cluster.actor_rollout_ref.model.use_remove_padding=False" \
  "cluster.actor_rollout_ref.model.adapter.embodiment_tag=$GROOT_EMBODIMENT_TAG" \
  "cluster.actor_rollout_ref.model.adapter.embodiment_id=$GROOT_EMBODIMENT_ID" \
  "cluster.actor_rollout_ref.model.adapter.action_dim=$ACTION_DIM" \
  "cluster.actor_rollout_ref.model.adapter.num_action_chunks=$NUM_ACTION_CHUNKS" \
  "cluster.actor_rollout_ref.model.adapter.critic.action_horizon=$NUM_ACTION_CHUNKS" \
  "cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" \
  "cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,Siglip2EncoderLayer,BasicTransformerBlock,MultiEmbodimentActionEncoder,CategorySpecificMLP]" \
  "cluster.actor_rollout_ref.actor.optim.lr=5e-6" \
  "cluster.actor_rollout_ref.actor.optim.warmup_style=constant" \
  "cluster.actor_rollout_ref.actor.mini_batch_size=$MINI_BATCH_SIZE" \
  "cluster.actor_rollout_ref.actor.micro_batch_size=$MICRO_BATCH_SIZE" \
  "cluster.actor_rollout_ref.actor.actor_update_interval=$ACTOR_UPDATE_INTERVAL" \
  "cluster.actor_rollout_ref.actor.sac.auto_entropy=$AUTO_ENTROPY" \
  "cluster.actor_rollout_ref.actor.sac.initial_alpha=$INITIAL_ALPHA" \
  "cluster.actor_rollout_ref.actor.sac.alpha_type=$ALPHA_TYPE" \
  "cluster.actor_rollout_ref.actor.critic.tau=$CRITIC_TAU" \
  "cluster.actor_rollout_ref.actor.critic.warmup_steps=$CRITIC_WARMUP_STEPS" \
  "cluster.actor_rollout_ref.actor.replay.save_dir=$REPLAY_POOL_DIR" \
  "cluster.actor_rollout_ref.actor.replay.online_single_size=2000" \
  "cluster.actor_rollout_ref.rollout.name=hf" \
  "cluster.actor_rollout_ref.rollout.output_critic_value=false" \
  "cluster.actor_rollout_ref.rollout.tensor_model_parallel_size=1" \
  "cluster.env.env_loop.pipeline_stage_num=$NUM_STAGE" \
  "cluster.env.env_loop.max_interactions=$MAX_INTERACTIONS" \
  "cluster.env.env_worker.auto_reset=true" \
  "cluster.env.env_worker.num_envs=$NUM_ENV" \
  "cluster.env.env_worker.simulator_start_timeout_s=600" \
  "cluster.env.env_worker.simulator.simulator_type=arena" \
  "${EXTRA_OVERRIDES[@]}" \
  "cluster.env.env_worker.modes=[train]" \
  "cluster.env.env_worker.teleop.enable=false" \
  "cluster.env.env_worker.recorder.enable=true" \
  "cluster.env.env_worker.recorder.recorders=[video]" \
  "cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos" \
  "cluster.resource.env.nnodes=$NUM_NODES" \
  "cluster.resource.env.gpus_per_node=$NUM_ENV_GPUS" \
  "cluster.resource.env.workers_per_node=$NUM_ENV_GPUS" \
  "cluster.resource.model.nnodes=$NUM_NODES" \
  "cluster.resource.model.gpus_per_node=$NUM_MODEL_GPUS" \
  "cluster.resource.model.workers_per_node=$NUM_MODEL_GPUS" \
  "cluster.checkpoint.resume_mode=${RESUME_MODE:-disable}" \
  "cluster.checkpoint.resume_from_path=${RESUME_FROM_PATH:-null}" \
  "cluster.checkpoint.default_local_dir=$OUTPUT_ROOT/checkpoints" \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.experiment_name=$EXPERIMENT_NAME" \
  "trainer.logger=$TRAINER_LOGGER" \
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
  "trainer.rollout_interval=$ROLLOUT_INTERVAL" \
  "trainer.warm_rollout_steps=$WARM_ROLLOUT_STEPS" \
  "trainer.save_freq=$SAVE_FREQ" \
  "trainer.test_freq=$TEST_FREQ" \
  "trainer.val_before_train=$VAL_BEFORE_TRAIN" \
  "trainer.val_only=False" \
  "trainer.episodic_replay=$EPISODIC_REPLAY" \
  "trainer.episodic_max_open_len=$EPISODIC_MAX_OPEN_LEN" \
  "$@"
