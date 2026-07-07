#!/usr/bin/env bash
#
# Launch the verl-vla arena container (non-root, all GPUs) and run the G1 SAC
# smoke data-collection flow inside it.
#
# By default the host verl-vla repo is bind-mounted into the container at
# /workspaces/verl-vla, so:
#   * the container runs THIS repo's code, and
#   * all outputs are written back into ./outputs/ on the host (visible in the IDE).
#
# Usage:
#   examples/arena_sac/run_docker_smoke.sh              # start container + run smoke
#   examples/arena_sac/run_docker_smoke.sh --shell      # start container + drop into a shell
#   examples/arena_sac/run_docker_smoke.sh --no-run     # only (re)start the container
#
# Common overrides (env vars):
#   IMAGE            docker image to use
#   CONTAINER_NAME   container name
#   RECREATE=1       force remove + recreate the container
#   MOUNT_REPO=0     do NOT bind-mount the host repo (use the image's baked copy)
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
SMOKE_SCRIPT="${SMOKE_SCRIPT:-examples/arena_sac/run_pi05_arena_g1_sac_smoke.sh}"
RECREATE="${RECREATE:-0}"
# Bind-mount the host repo into the container (so outputs persist on the host).
MOUNT_REPO="${MOUNT_REPO:-1}"

MODE="run"
case "${1:-}" in
  --shell)  MODE="shell" ;;
  --no-run) MODE="none" ;;
  "")       MODE="run" ;;
  *) echo "Unknown option: $1" >&2; exit 1 ;;
esac

log() { echo -e "\033[1;36m[run_docker_smoke]\033[0m $*"; }

# ---------------------------------------------------------------------------
# 1. Prepare host-side dirs the non-root container user must write into.
#    The container user (uid 1234) differs from the host user, so make the
#    write targets world-writable. billyw owns the repo, so no sudo needed.
# ---------------------------------------------------------------------------
DOCKER_MOUNT_ARGS=()
DOCKER_ENV_ARGS=()
if [[ "$MOUNT_REPO" == "1" ]]; then
  log "Mounting host repo '$HOST_REPO' -> '$WORKDIR'"
  mkdir -p "$HOST_REPO/outputs" "$HOST_REPO/certs"
  chmod 777 "$HOST_REPO/outputs" "$HOST_REPO/certs"
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
#    the non-root user can write outputs/certs/etc.
# ---------------------------------------------------------------------------
if [[ "$MOUNT_REPO" != "1" ]]; then
  log "Chowning $WORKDIR to $CONTAINER_USER (baked copy)"
  docker exec -u root "$CONTAINER_NAME" chown -R "$CONTAINER_USER:$CONTAINER_USER" "$WORKDIR"
fi

# ---------------------------------------------------------------------------
# 5. Run the smoke flow (or drop into a shell / do nothing).
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
    log "Running smoke script: $SMOKE_SCRIPT"
    docker exec -ti -u "$CONTAINER_USER" -w "$WORKDIR" "$CONTAINER_NAME" \
      bash "$SMOKE_SCRIPT"
    if [[ "$MOUNT_REPO" == "1" ]]; then
      log "Outputs saved on host at: $HOST_REPO/outputs/arena_g1_smoke"
    fi
    ;;
esac
