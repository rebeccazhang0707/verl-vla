#!/usr/bin/env bash
#
# DSRL (arXiv:2506.15799) for GR00T N1.6 on Isaac Lab Arena LIBERO.
#
# The whole GR00T policy stays frozen. Only a Transformer noise actor over the
# flow-matching initial noise x0 and its SAC critic are trained: the steering
# noise IS the SAC action, and the frozen flow head decodes it into the env
# action with a deterministic Euler ODE.
#
# Defaults reproduce the published LIBERO-Spatial runs; see
# docs/reinforcement-learning/dsrl/gr00t/arena-libero-spatial.md.
#
# Run inside the Arena GR00T container (docker/Dockerfile.isaaclab_arena):
#
#   TASK_ID=7 TOTAL_TRAINING_STEPS=8000 \
#     bash examples/rl/dsrl/gr00t/run_gr00t_arena_libero_dsrl.sh
#
# Any extra argument is forwarded to Hydra verbatim.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

# ── Task and checkpoint ──────────────────────────────────────────────────────
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK_ID="${TASK_ID:-3}"
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-$REPO_ROOT/.data/models/checkpoint-10000}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/rl/dsrl/gr00t/${TASK_SUITE}-task${TASK_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dsrl_${TASK_SUITE}_task${TASK_ID}}"

# Replay shards are large (~90 GB per run) and regenerable. Point this at
# node-local scratch when the output root lives on a quota-limited filesystem.
REPLAY_POOL_DIR="${REPLAY_POOL_DIR:-$OUTPUT_ROOT/replay_pools}"

# ── Topology: 4 simulation GPUs + 4 model GPUs on one 8-GPU node ─────────────
# The two Ray pools are disjoint, so this needs 8 GPUs. num_envs is per env
# worker, giving 4 x 64 = 256 parallel auto-reset environments.
NUM_ENV_GPUS="${NUM_ENV_GPUS:-4}"
NUM_MODEL_GPUS="${NUM_MODEL_GPUS:-4}"
NUM_ENV="${NUM_ENV:-64}"

# ── Schedule ─────────────────────────────────────────────────────────────────
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-8000}"
ROLLOUT_INTERVAL="${ROLLOUT_INTERVAL:-160}"
SAVE_FREQ="${SAVE_FREQ:-500}"
TEST_FREQ="${TEST_FREQ:-500}"
EVAL_EPISODES="${EVAL_EPISODES:-32}"

# ── Environment paths ────────────────────────────────────────────────────────
# Arena ships inside the image, but the LIBERO scene USDs and demonstration
# HDF5 files do not. Arena resolves each directory from its own environment
# variable before falling back to a fixed `benchmarks/datasets/libero/...`
# layout, so setting them directly keeps the download path flat.
ARENA_ROOT="${ARENA_ROOT:-/workspace/arena}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$REPO_ROOT/.data/libero}"
LIBERO_ASSETS_DATA_DIR="${LIBERO_ASSETS_DATA_DIR:-$LIBERO_DATA_ROOT/USD}"
LIBERO_ASSEMBLED_DATASET_DIR="${LIBERO_ASSEMBLED_DATASET_DIR:-$LIBERO_DATA_ROOT/assembled_hdf5}"
# The task-config JSONs ship with Arena, so nothing needs to be copied.
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-$ARENA_ROOT/isaaclab_arena_examples/external_environments/libero/data/config}"

for dir in "$LIBERO_ASSETS_DATA_DIR" "$LIBERO_ASSEMBLED_DATASET_DIR"; do
  [[ -d "$dir" ]] || echo "[warn] missing LIBERO asset directory: $dir" >&2
done

TENSORBOARD_DIR="${TENSORBOARD_DIR:-$OUTPUT_ROOT/tensorboard}"
mkdir -p "$OUTPUT_ROOT/videos" "$OUTPUT_ROOT/checkpoints" "$TENSORBOARD_DIR" "$REPLAY_POOL_DIR"

# The image symlinks /root/.cache here, so these must exist before Isaac Sim
# starts. On a bind-mounted repo the image's own mkdir is shadowed.
mkdir -p "$REPO_ROOT/.data/gr00t_arena/cache/root_cache"/{nv,triton/autotune,huggingface}

ISAAC_PYTHONPATH="$(/isaac-sim/python.sh -c 'import os, sys; print(os.pathsep.join(p for p in sys.path if p and os.path.isdir(p)))')"
export PYTHONPATH="$REPO_ROOT/src:$ARENA_ROOT:$ARENA_ROOT/submodules/Isaac-GR00T:$ISAAC_PYTHONPATH"

# Ray workers do not inherit the shell environment, so everything the env and
# model workers need has to be declared here.
runtime_env=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='$PYTHONPATH'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_ASSETS_DATA_DIR='$LIBERO_ASSETS_DATA_DIR'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_ASSEMBLED_DATASET_DIR='$LIBERO_ASSEMBLED_DATASET_DIR'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_CONFIG_DIR='$LIBERO_CONFIG_DIR'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR='$TENSORBOARD_DIR'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED='0'"
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO"
)

overrides=(
  "${runtime_env[@]}"
  "model/adapter@cluster.actor_rollout_ref.model.adapter=gr00t"
  "model/override@cluster.actor_rollout_ref.model.override_config=gr00t"
  # ─ Frozen GR00T policy, LIBERO end-effector embodiment ─
  "cluster.actor_rollout_ref.model.path=$GROOT_MODEL_PATH"
  "cluster.actor_rollout_ref.model.tokenizer_path=$GROOT_MODEL_PATH"
  "cluster.actor_rollout_ref.model.trust_remote_code=True"
  "cluster.actor_rollout_ref.model.load_tokenizer=False"
  "cluster.actor_rollout_ref.model.use_remove_padding=False"
  # policy_type stays at the gr00t.yaml default `arena`: that adapter reads the
  # generic Arena obs keys and is not GR1-specific. The `libero` adapter is for
  # the MuJoCo LIBERO harness, not Arena.
  "cluster.actor_rollout_ref.model.adapter.embodiment_tag=new_embodiment"
  "cluster.actor_rollout_ref.model.adapter.embodiment_id=10"
  "cluster.actor_rollout_ref.model.adapter.action_dim=7"
  "cluster.actor_rollout_ref.model.adapter.num_action_chunks=16"
  # ─ DSRL noise actor: Transformer, one latent per flow step ─
  "cluster.actor_rollout_ref.model.adapter.dsrl.enabled=true"
  "cluster.actor_rollout_ref.model.adapter.dsrl.actor_type=transformer"
  "cluster.actor_rollout_ref.model.adapter.dsrl.noise_per_step=true"
  # ─ Transformer sequence critic over [obs_token, noise_1..K] ─
  "cluster.actor_rollout_ref.model.adapter.critic.type=transformer"
  "cluster.actor_rollout_ref.model.adapter.critic.head_num=4"
  "cluster.actor_rollout_ref.model.adapter.critic.layernorm=True"
  # ─ Optimisation: noise-actor lr, not a VLA lr ─
  "cluster.actor_rollout_ref.actor.optim.lr=5e-5"
  "cluster.actor_rollout_ref.actor.optim.warmup_style=constant"
  "cluster.actor_rollout_ref.actor.critic.lr=1e-4"
  "cluster.actor_rollout_ref.actor.critic.tau=0.005"
  "cluster.actor_rollout_ref.actor.critic.warmup_steps=100"
  "cluster.actor_rollout_ref.actor.actor_update_interval=1"
  "cluster.actor_rollout_ref.actor.mini_batch_size=128"
  "cluster.actor_rollout_ref.actor.micro_batch_size=32"
  "cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
  # The shared default wraps pi0 layers; GR00T needs its own module names.
  "cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,Siglip2EncoderLayer,BasicTransformerBlock,MultiEmbodimentActionEncoder,CategorySpecificMLP]"
  # ─ Entropy: target_entropy=0 anneals alpha toward zero by construction ─
  "cluster.actor_rollout_ref.actor.sac.auto_entropy=True"
  "cluster.actor_rollout_ref.actor.sac.alpha_type=softplus"
  "cluster.actor_rollout_ref.actor.sac.initial_alpha=1.0"
  "cluster.actor_rollout_ref.actor.sac.target_entropy=0.0"
  # ─ Keep -alpha*log_pi out of the TD target: the summed noise
  #    log-prob would otherwise dominate the bootstrap ─
  "cluster.actor_rollout_ref.actor.sac.backup_entropy=False"
  "cluster.actor_rollout_ref.actor.replay.actor_positive_sample_ratio=0.6"
  "cluster.actor_rollout_ref.actor.replay.online_single_size=20000"
  "cluster.actor_rollout_ref.actor.replay.save_dir=$REPLAY_POOL_DIR"
  "cluster.actor_rollout_ref.rollout.name=hf"
  "cluster.actor_rollout_ref.rollout.output_critic_value=false"
  # ─ Arena LIBERO: 20 interactions x 16 chunks = 320 env steps ─
  "cluster.env.env_loop.max_interactions=20"
  "cluster.env.env_worker.auto_reset=true"
  "cluster.env.env_worker.num_envs=$NUM_ENV"
  # Isaac Sim needs well over the 180 s default to come up with 64 envs. The
  # first run on a fresh image is slower still: it downloads the Isaac and
  # lightwheel assets and compiles warp kernels and shaders into
  # .data/gr00t_arena/cache. Later runs reuse that cache.
  "cluster.env.env_worker.simulator_start_timeout_s=${SIM_START_TIMEOUT_S:-2400}"
  "cluster.env.env_worker.simulator.simulator_type=arena"
  "cluster.env.env_worker.simulator.arena.environment=libero"
  "cluster.env.env_worker.simulator.arena.libero.libero_task_suite=$TASK_SUITE"
  "cluster.env.env_worker.simulator.arena.libero.libero_task_id=$TASK_ID"
  "cluster.env.env_worker.simulator.arena.sim_dt=0.016666666666666666"
  "cluster.env.env_worker.simulator.arena.decimation=3"
  # Recording is off by default, and the default recorder set also writes a
  # LeRobot dataset this recipe does not need.
  "cluster.env.env_worker.recorder.enable=true"
  "cluster.env.env_worker.recorder.recorders=[video]"
  "cluster.env.env_worker.recorder.video.fps=20"
  "cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos"
  # ─ Two disjoint Ray pools: simulation and model ─
  "cluster.resource.env.gpus_per_node=$NUM_ENV_GPUS"
  "cluster.resource.env.workers_per_node=$NUM_ENV_GPUS"
  "cluster.resource.model.gpus_per_node=$NUM_MODEL_GPUS"
  "cluster.resource.model.workers_per_node=$NUM_MODEL_GPUS"
  # Checkpoint resume defaults to `auto`, which would silently continue a run
  # found in the output directory. Start fresh unless explicitly resumed.
  "cluster.checkpoint.resume_mode=${RESUME_MODE:-disable}"
  "cluster.checkpoint.default_local_dir=$OUTPUT_ROOT/checkpoints"
  "trainer.project_name=gr00t-arena-libero-dsrl"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.logger=[console,tensorboard]"
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
  "trainer.rollout_interval=$ROLLOUT_INTERVAL"
  "trainer.warm_rollout_steps=5"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.eval_episodes=$EVAL_EPISODES"
  "trainer.val_before_train=False"
)

exec /isaac-sim/python.sh -m verl_vla.entrypoints.train.sac "${overrides[@]}" "$@"
