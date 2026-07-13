#!/usr/bin/env bash
#
# Launch the verl-vla arena container (non-root, all GPUs) and run PI0.5 policy
# EVALUATION on the Arena G1 task, saving rollout videos back to the host repo.
#
# By default the host verl-vla repo is bind-mounted into the container at
# /workspaces/verl-vla, so eval videos land in ./outputs/arena_g1_eval on the
# host (visible in the IDE).
#
# GR00T eval: do NOT use this script — verl-vla-arena has no gr00t deps.
# Use examples/arena_sac/run_docker_gr00t_eval.sh (isaaclab_arena:cuda_gr00t_gn16).
#
# Usage:
#   examples/arena_sac/run_docker_eval.sh              # start container + run eval
#   examples/arena_sac/run_docker_eval.sh --shell      # start container + drop into a shell
#   examples/arena_sac/run_docker_eval.sh --no-run     # only (re)start the container
#
# Common overrides (env vars):
#   IMAGE            docker image to use
#   CONTAINER_NAME   container name
#   RECREATE=1       force remove + recreate the container
#   MOUNT_REPO=0     do NOT bind-mount the host repo (use the image's baked copy)
#   MODEL_PATH       policy checkpoint passed through to the eval script
#   MAX_EPISODES     number of episodes to evaluate (default 10)
#   EVAL_SCRIPT      eval script path inside the container (default: PI0.5 G1)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-vemlp-demo-cn-beijing.cr.volces.com/verl/verl-vla-arena:v0.2}"
CONTAINER_NAME="${CONTAINER_NAME:-verl-vla-arena}"
# Non-root user baked into the Isaac Sim image (uid 1234).
CONTAINER_USER="${CONTAINER_USER:-isaac-sim}"
# Path of the repo inside the container.
WORKDIR="${WORKDIR:-/workspaces/verl-vla}"
EVAL_SCRIPT="${EVAL_SCRIPT:-examples/arena_sac/run_pi05_arena_g1_eval.sh}"
RECREATE="${RECREATE:-0}"
# Bind-mount the host repo into the container (so videos/metrics persist on the host).
MOUNT_REPO="${MOUNT_REPO:-1}"

# Eval knobs forwarded into the container's environment.
MODEL_PATH="${MODEL_PATH:-/workspaces/models/torch_pi05_base}"
MAX_EPISODES="${MAX_EPISODES:-10}"

MODE="run"
case "${1:-}" in
  --shell)  MODE="shell" ;;
  --no-run) MODE="none" ;;
  "")       MODE="run" ;;
  *) echo "Unknown option: $1" >&2; exit 1 ;;
esac

log() { echo -e "\033[1;35m[run_docker_eval]\033[0m $*"; }

# ---------------------------------------------------------------------------
# 1. Prepare host-side output dir the non-root container user must write into.
#    The container user (uid 1234) differs from the host user, so make the
#    write target world-writable. billyw owns the repo, so no sudo needed.
# ---------------------------------------------------------------------------
DOCKER_MOUNT_ARGS=()
DOCKER_ENV_ARGS=()
if [[ "$MOUNT_REPO" == "1" ]]; then
  log "Mounting host repo '$HOST_REPO' -> '$WORKDIR'"
  mkdir -p "$HOST_REPO/outputs"
  chmod 777 "$HOST_REPO/outputs"
  DOCKER_MOUNT_ARGS=(-v "$HOST_REPO:$WORKDIR")
  # Avoid failing on __pycache__ writes into the (host-owned) src tree.
  DOCKER_ENV_ARGS=(-e PYTHONDONTWRITEBYTECODE=1)
fi

# ---------------------------------------------------------------------------
# 2. (Re)create the container with all GPUs, non-root default user.
# ---------------------------------------------------------------------------
if [[ "$RECREATE" == "1" ]]; then
  log "Removing existing container '$CONTAINER_NAME' (RECREATE=1)"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# If a container is already running but was created with an incompatible mount
# config (e.g. the repo isn't mounted while MOUNT_REPO=1, or vice versa),
# recreate it so the requested layout actually takes effect.
if [[ "$RECREATE" != "1" ]] && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  HAS_MOUNT=0
  if docker inspect -f '{{range .Mounts}}{{println .Source .Destination}}{{end}}' "$CONTAINER_NAME" 2>/dev/null \
      | grep -qx "$HOST_REPO $WORKDIR"; then
    HAS_MOUNT=1
  fi
  if [[ "$MOUNT_REPO" == "1" && "$HAS_MOUNT" != "1" ]]; then
    log "Running container is missing the repo mount ($HOST_REPO -> $WORKDIR); recreating it"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  elif [[ "$MOUNT_REPO" != "1" && "$HAS_MOUNT" == "1" ]]; then
    log "Running container has the repo mounted but MOUNT_REPO=0; recreating it"
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  # If a stopped container with the same name exists, remove it first.
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  log "Starting container '$CONTAINER_NAME' from image '$IMAGE' (--gpus all)"
  docker run -d --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host --network=host \
    --ulimit memlock=-1 --ulimit stack=-1 \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    "${DOCKER_ENV_ARGS[@]}" \
    "${DOCKER_MOUNT_ARGS[@]}" \
    --entrypoint bash \
    "$IMAGE" \
    -c 'sleep infinity' >/dev/null
else
  log "Container '$CONTAINER_NAME' already running, reusing it"
fi

# ---------------------------------------------------------------------------
# 3. Sanity checks: GPU visibility + default user.
# ---------------------------------------------------------------------------
log "Container user: $(docker exec "$CONTAINER_NAME" whoami)"
log "GPUs visible inside container:"
docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---------------------------------------------------------------------------
# 4. If NOT mounting the host repo, fix ownership of the image's baked copy so
#    the non-root user can write outputs.
# ---------------------------------------------------------------------------
if [[ "$MOUNT_REPO" != "1" ]]; then
  log "Chowning $WORKDIR to $CONTAINER_USER (baked copy)"
  docker exec -u root "$CONTAINER_NAME" chown -R "$CONTAINER_USER:$CONTAINER_USER" "$WORKDIR"
fi

# ---------------------------------------------------------------------------
# 5. Run the eval flow (or drop into a shell / do nothing).
# ---------------------------------------------------------------------------
case "$MODE" in
  none)
    log "Container ready. Attach with: docker exec -ti $CONTAINER_NAME bash"
    ;;
  shell)
    log "Dropping into an interactive shell as $CONTAINER_USER"
    docker exec -ti -u "$CONTAINER_USER" -w "$WORKDIR" "$CONTAINER_NAME" bash
    ;;
  run)
    log "Running eval script: $EVAL_SCRIPT (MODEL_PATH=$MODEL_PATH, MAX_EPISODES=$MAX_EPISODES)"
    docker exec -ti -u "$CONTAINER_USER" -w "$WORKDIR" \
      -e MODEL_PATH="$MODEL_PATH" \
      -e MAX_EPISODES="$MAX_EPISODES" \
      "$CONTAINER_NAME" \
      bash "$EVAL_SCRIPT"
    if [[ "$MOUNT_REPO" == "1" ]]; then
      log "Eval videos saved on host at:   $HOST_REPO/outputs/arena_g1_eval/videos"
      log "Eval metrics saved on host at:  $HOST_REPO/outputs/arena_g1_eval/eval_metrics"
    fi
    ;;
esac
