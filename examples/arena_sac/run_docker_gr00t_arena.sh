#!/usr/bin/env bash
#
# Launch the IsaacLab-Arena GR00T image (isaaclab_arena:cuda_gr00t_gn16) and run
# a GR00T N1.6 policy EVALUATION script (default: Arena GR1 fridge).
#
# The plain verl-vla-arena image has NO gr00t / Eagle / /opt/groot_deps stack —
# use this helper (or IsaacLab-Arena's docker/run_docker.sh -g) instead of
# run_docker_eval.sh for GR00T.
#
# Mount mapping (matches IsaacLab-Arena docker/run_docker.sh -g -e):
#   host verl-vla repo  -> /eval
#   host checkpoint dir -> /models
#   host Arena repo     -> /workspaces/isaaclab_arena  (overrides the baked-in tree)
#   host libero_in_lab  -> /libero_in_lab              (LIBERO USD/configs; optional)
#
# Usage:
#   examples/arena_sac/run_docker_gr00t_eval.sh              # GR1 fridge eval (default)
#   EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_libero_spatial_task3_eval.sh \
#     MODELS_HOST=~/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec \
#     GROOT_MODEL_PATH=/models/checkpoint-5000 \
#     examples/arena_sac/run_docker_gr00t_eval.sh            # LIBERO spatial task 3
#   EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_gr1_sac.sh \
#     MODELS_HOST=~/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out \
#     OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_sac \
#     examples/arena_sac/run_docker_gr00t_arena.sh            # GR1 fridge SAC train
#   EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_libero_sac.sh \
#     MODELS_HOST=~/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec \
#     OUTPUT_ROOT=/eval/outputs/arena_gr00t_libero_sac \
#     examples/arena_sac/run_docker_gr00t_arena.sh            # LIBERO SAC train
#   examples/arena_sac/run_docker_gr00t_arena.sh --shell
#   examples/arena_sac/run_docker_gr00t_arena.sh --no-run
#
# Common overrides (env vars):
#   IMAGE              docker image (default: isaaclab_arena:cuda_gr00t_gn16)
#   CONTAINER_NAME     container name
#   RECREATE=1         force remove + recreate
#   ARENA_HOST         host IsaacLab-Arena checkout -> /workspaces/isaaclab_arena
#   LIBERO_IN_LAB_HOST host libero_in_lab checkout -> /libero_in_lab
#   MODELS_HOST        host dir mounted at /models (checkpoint parent)
#   GROOT_MODEL_PATH   checkpoint path *inside* the container
#   MAX_EPISODES       episodes to evaluate (default 10; ignored by train scripts)
#   OUTPUT_ROOT        eval/train output root *inside* the container
#   EVAL_SCRIPT        script path relative to the mounted repo (eval or train)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-isaaclab_arena:cuda_gr00t_gn16}"
CONTAINER_NAME="${CONTAINER_NAME:-isaaclab_arena-cuda_gr00t_gn16}"
# Repo mount inside the container (README / run_docker.sh -e convention).
WORKDIR="${WORKDIR:-/eval}"
EVAL_SCRIPT="${EVAL_SCRIPT:-examples/arena_sac/run_gr00t_arena_gr1_eval.sh}"
RECREATE="${RECREATE:-0}"

# Checkpoint parent on the host -> /models. Override to your export's parent dir.
# Default matches the ranch GR1 fridge finetune used by the README runbook.
MODELS_HOST="${MODELS_HOST:-$HOME/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out}"
# Local IsaacLab-Arena checkout -> /workspaces/isaaclab_arena (same as run_docker.sh -v .:$WORKDIR).
# Required so wrist-cam / env patches on the host branch are visible inside the container;
# the image's baked copy is often stale (no .git, older than host reb/arena-verl).
ARENA_HOST="${ARENA_HOST:-$HOME/Projects/libero_rl_example/IsaacLab-Arena}"
ARENA_WORKDIR="${ARENA_WORKDIR:-/workspaces/isaaclab_arena}"
# LIBERO-in-Lab assets (required for arena_libero evals; harmless for GR1).
LIBERO_IN_LAB_HOST="${LIBERO_IN_LAB_HOST:-$HOME/Projects/libero_rl_example/libero_in_lab}"
LIBERO_IN_LAB_WORKDIR="${LIBERO_IN_LAB_WORKDIR:-/libero_in_lab}"
# Path seen by the eval script inside the container.
GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-5000}"
MAX_EPISODES="${MAX_EPISODES:-10}"
# Default output root matches the GR1 eval script; override for LIBERO / other tasks.
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/outputs/arena_gr00t_gr1_eval}"

MODE="run"
case "${1:-}" in
  --shell)  MODE="shell" ;;
  --no-run) MODE="none" ;;
  "")       MODE="run" ;;
  *) echo "Unknown option: $1" >&2; exit 1 ;;
esac

log() { echo -e "\033[1;35m[run_docker_gr00t_arena]\033[0m $*"; }

# ---------------------------------------------------------------------------
# 1. Host-side dirs the container must write into / read from.
# ---------------------------------------------------------------------------
mkdir -p "$HOST_REPO/outputs"
chmod 777 "$HOST_REPO/outputs" 2>/dev/null || true

DOCKER_MOUNT_ARGS=(
  -v "$HOST_REPO:$WORKDIR"
)
DOCKER_ENV_ARGS=(
  -e PYTHONDONTWRITEBYTECODE=1
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
)

if [[ -d "$ARENA_HOST" ]]; then
  log "Mounting Arena '$ARENA_HOST' -> $ARENA_WORKDIR"
  DOCKER_MOUNT_ARGS+=(-v "$ARENA_HOST:$ARENA_WORKDIR")
else
  log "ERROR: ARENA_HOST='$ARENA_HOST' does not exist."
  log "  Set ARENA_HOST to your local IsaacLab-Arena checkout (needed for wrist cam / env code)."
  exit 1
fi

if [[ -d "$MODELS_HOST" ]]; then
  log "Mounting models '$MODELS_HOST' -> /models"
  DOCKER_MOUNT_ARGS+=(-v "$MODELS_HOST:/models")
else
  log "WARNING: MODELS_HOST='$MODELS_HOST' does not exist; /models will be empty."
  log "  Set MODELS_HOST to the host directory that contains checkpoint-*-export."
fi

if [[ -d "$LIBERO_IN_LAB_HOST" ]]; then
  log "Mounting libero_in_lab '$LIBERO_IN_LAB_HOST' -> $LIBERO_IN_LAB_WORKDIR"
  DOCKER_MOUNT_ARGS+=(-v "$LIBERO_IN_LAB_HOST:$LIBERO_IN_LAB_WORKDIR")
else
  log "WARNING: LIBERO_IN_LAB_HOST='$LIBERO_IN_LAB_HOST' missing; Arena LIBERO evals need it."
fi

# CUDA 12.8 forward-compat (same as IsaacLab-Arena run_docker.sh); optional.
if [[ -d "$HOME/cuda128-compat" ]]; then
  DOCKER_MOUNT_ARGS+=(-v "$HOME/cuda128-compat:/opt/cuda128-compat:ro")
fi

# Do NOT bind-mount host /tmp -> /tmp: host files (e.g. verl_constraints.txt owned
# by another uid) cause Permission denied inside the container. Use the image's
# own /tmp; Hydra/Ray scratch stays local to the container.

# ---------------------------------------------------------------------------
# 2. (Re)create the container with all GPUs.
#    Bypass the image entrypoint (host-user bootstrap) and keep a long-lived
#    bash so we can docker exec repeatedly — same pattern as run_docker_eval.sh.
# ---------------------------------------------------------------------------
if [[ "$RECREATE" == "1" ]]; then
  log "Removing existing container '$CONTAINER_NAME' (RECREATE=1)"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

need_recreate=0
if [[ "$RECREATE" != "1" ]] && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  mounts="$(docker inspect -f '{{range .Mounts}}{{println .Source .Destination}}{{end}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  if ! grep -qx "$HOST_REPO $WORKDIR" <<<"$mounts"; then
    log "Running container is missing the repo mount ($HOST_REPO -> $WORKDIR); recreating"
    need_recreate=1
  elif ! grep -qx "$ARENA_HOST $ARENA_WORKDIR" <<<"$mounts"; then
    log "Running container is missing the Arena mount ($ARENA_HOST -> $ARENA_WORKDIR); recreating"
    need_recreate=1
  elif [[ -d "$MODELS_HOST" ]] && ! grep -qx "$MODELS_HOST /models" <<<"$mounts"; then
    log "Running container is missing the models mount ($MODELS_HOST -> /models); recreating"
    need_recreate=1
  elif [[ -d "$LIBERO_IN_LAB_HOST" ]] && ! grep -qx "$LIBERO_IN_LAB_HOST $LIBERO_IN_LAB_WORKDIR" <<<"$mounts"; then
    log "Running container is missing libero_in_lab mount ($LIBERO_IN_LAB_HOST -> $LIBERO_IN_LAB_WORKDIR); recreating"
    need_recreate=1
  fi
  if [[ "$need_recreate" == "1" ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  log "Starting container '$CONTAINER_NAME' from image '$IMAGE' (--gpus all)"
  docker run -d --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host --network=host \
    --ulimit memlock=-1 --ulimit stack=-1 \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_MOUNT_ARGS[@]}" \
    --entrypoint bash \
    "$IMAGE" \
    -c 'sleep infinity' >/dev/null
else
  log "Container '$CONTAINER_NAME' already running, reusing it"
fi

# ---------------------------------------------------------------------------
# 3. Sanity checks.
# ---------------------------------------------------------------------------
log "Container user: $(docker exec "$CONTAINER_NAME" whoami)"
log "GPUs visible inside container:"
docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=index,name,memory.total --format=csv
if docker exec "$CONTAINER_NAME" test -d /opt/groot_deps; then
  log "Found /opt/groot_deps (GR00T deps OK)"
else
  log "WARNING: /opt/groot_deps missing — is IMAGE really the cuda_gr00t_gn16 build?"
fi
# Confirm the bind-mounted Arena (not the baked image copy) is what PYTHONPATH sees.
if docker exec "$CONTAINER_NAME" test -f "$ARENA_WORKDIR/isaaclab_arena_environments/gr1_put_and_close_door_environment.py"; then
  if docker exec "$CONTAINER_NAME" grep -q 'GR1T2WristCameraCfg' \
      "$ARENA_WORKDIR/isaaclab_arena_environments/gr1_put_and_close_door_environment.py"; then
    log "Arena mount OK: wrist-cam patch present at $ARENA_WORKDIR"
  else
    log "WARNING: Arena at $ARENA_WORKDIR has no GR1T2WristCameraCfg — check ARENA_HOST branch"
  fi
else
  log "WARNING: Arena env file missing under $ARENA_WORKDIR"
fi

# ---------------------------------------------------------------------------
# 4. Run eval / shell / nothing.
# ---------------------------------------------------------------------------
# Allocate a TTY only when stdin is a terminal (avoids "the input device is not a TTY"
# when this helper is driven by CI / nohup / pipes).
DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS+=(-t)
fi

case "$MODE" in
  none)
    log "Container ready. Attach with: docker exec -ti $CONTAINER_NAME bash"
    ;;
  shell)
    log "Dropping into an interactive shell (workdir=$WORKDIR)"
    docker exec -ti -w "$WORKDIR" "$CONTAINER_NAME" bash
    ;;
  run)
    # Ensure host-visible output dirs are writable (prior root runs leave root-owned dirs).
    docker exec "$CONTAINER_NAME" bash -lc \
      "mkdir -p '$OUTPUT_ROOT'/{videos,eval_metrics} && chmod -R a+rwX '$OUTPUT_ROOT'" \
      >/dev/null 2>&1 || true
    log "Running $EVAL_SCRIPT (GROOT_MODEL_PATH=$GROOT_MODEL_PATH, MAX_EPISODES=$MAX_EPISODES, OUTPUT_ROOT=$OUTPUT_ROOT)"
    docker exec -i "${DOCKER_TTY_ARGS[@]}" -w "$WORKDIR" \
      -e GROOT_MODEL_PATH="$GROOT_MODEL_PATH" \
      -e MAX_EPISODES="$MAX_EPISODES" \
      -e OUTPUT_ROOT="$OUTPUT_ROOT" \
      -e LIBERO_IN_LAB_ROOT="$LIBERO_IN_LAB_WORKDIR" \
      -e TASK_SUITE="${TASK_SUITE:-}" \
      -e TASK_ID="${TASK_ID:-}" \
      "$CONTAINER_NAME" \
      bash "$EVAL_SCRIPT"
    # Map container OUTPUT_ROOT back to the host when it lives under the repo mount.
    HOST_OUTPUT="${OUTPUT_ROOT/#$WORKDIR/$HOST_REPO}"
    log "Eval videos on host:  $HOST_OUTPUT/videos"
    log "Eval metrics on host: $HOST_OUTPUT/eval_metrics"
    ;;
esac
