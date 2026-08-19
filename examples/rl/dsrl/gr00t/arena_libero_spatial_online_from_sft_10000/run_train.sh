#!/usr/bin/env bash
#
# DSRL for GR00T N1.6 on Isaac Lab Arena LIBERO Spatial.
#
# Run inside the Arena GR00T container; see
# docs/reinforcement-learning/dsrl/gr00t/arena-libero-spatial.md.
#
#   TASK_ID=7 bash examples/rl/dsrl/gr00t/arena_libero_spatial_online_from_sft_10000/run_train.sh
#
# Extra arguments are forwarded to Hydra.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "$REPO_ROOT"

# LIBERO Spatial task, zero-based.
TASK_ID="${TASK_ID:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/rl/dsrl/gr00t/arena-libero-spatial-task${TASK_ID}}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.data/models/checkpoint-10000}"

# Two disjoint Ray pools, so this needs 8 GPUs. num_envs is per env worker,
# giving 4 x 64 = 256 parallel auto-reset environments.
NUM_ENV_GPUS="${NUM_ENV_GPUS:-4}"
NUM_MODEL_GPUS="${NUM_MODEL_GPUS:-4}"
NUM_ENV="${NUM_ENV:-64}"

# Arena ships inside the image; the LIBERO scene USDs and demonstration HDF5
# files do not. Arena reads each directory from its own environment variable.
ARENA_ROOT="${ARENA_ROOT:-/workspace/arena}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${REPO_ROOT}/.data/libero}"
LIBERO_ASSETS_DATA_DIR="${LIBERO_ASSETS_DATA_DIR:-${LIBERO_DATA_ROOT}/USD}"
LIBERO_ASSEMBLED_DATASET_DIR="${LIBERO_ASSEMBLED_DATASET_DIR:-${LIBERO_DATA_ROOT}/assembled_hdf5}"
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-${ARENA_ROOT}/isaaclab_arena_examples/external_environments/libero/data/config}"

for dir in "$LIBERO_ASSETS_DATA_DIR" "$LIBERO_ASSEMBLED_DATASET_DIR"; do
  [[ -d "$dir" ]] || echo "[warn] missing LIBERO asset directory: $dir" >&2
done

# The image symlinks /root/.cache here, so it must exist before Isaac Sim
# starts. A bind-mounted repository shadows the directory the image created.
mkdir -p "${REPO_ROOT}/.data/gr00t_arena/cache/root_cache"/{nv,triton/autotune,huggingface}

ISAAC_PYTHONPATH="$(/isaac-sim/python.sh -c 'import os, sys; print(os.pathsep.join(p for p in sys.path if p and os.path.isdir(p)))')"
export PYTHONPATH="${REPO_ROOT}/src:${ARENA_ROOT}:${ARENA_ROOT}/submodules/Isaac-GR00T:${ISAAC_PYTHONPATH}"

# Isaac Sim needs the environment /isaac-sim/python.sh exports (EXP_PATH and
# friends); the vvla-train-sac console script has the right interpreter but not
# that environment, and the simulator fails to initialize without it.
exec /isaac-sim/python.sh -m verl_vla.entrypoints.train.sac \
  --config-dir "$SCRIPT_DIR" \
  --config-name dsrl \
  "output_dir=$OUTPUT_DIR" \
  "cluster.env.env_worker.simulator.arena.libero.libero_task_id=$TASK_ID" \
  "cluster.actor_rollout_ref.model.path=$MODEL_PATH" \
  "cluster.actor_rollout_ref.model.tokenizer_path=$MODEL_PATH" \
  "cluster.actor_rollout_ref.actor.mini_batch_size=128" \
  "cluster.actor_rollout_ref.actor.micro_batch_size=32" \
  "cluster.env.env_worker.num_envs=$NUM_ENV" \
  "cluster.resource.env.gpus_per_node=$NUM_ENV_GPUS" \
  "cluster.resource.env.workers_per_node=$NUM_ENV_GPUS" \
  "cluster.resource.model.gpus_per_node=$NUM_MODEL_GPUS" \
  "cluster.resource.model.workers_per_node=$NUM_MODEL_GPUS" \
  "trainer.experiment_name=gr00t-arena-libero-spatial-task${TASK_ID}-dsrl" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard" \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDNN_SDPA_ENABLED='0'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_ASSETS_DATA_DIR=$LIBERO_ASSETS_DATA_DIR" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_ASSEMBLED_DATASET_DIR=$LIBERO_ASSEMBLED_DATASET_DIR" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.LIBERO_CONFIG_DIR=$LIBERO_CONFIG_DIR" \
  "$@"
