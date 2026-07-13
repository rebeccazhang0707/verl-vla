#!/usr/bin/env bash
#
# Unified Arena container launcher for verl-vla.
#
# One wrapper for BOTH backends — it (re)creates the right container, mounts the
# right host dirs, then runs an inner eval / sac / smoke script inside it. The
# inner script is selected with EVAL_SCRIPT (relative to the repo root as seen
# inside the container).
#
#   BACKEND=gr00t (default)  isaaclab_arena:cuda_gr00t_gn16  (root user)
#       Has the GR00T / Eagle / /opt/groot_deps stack. Mounts:
#         host verl-vla repo  -> /eval
#         host checkpoint dir -> /models            (MODELS_HOST)
#         host IsaacLab-Arena -> /workspaces/isaaclab_arena   (ARENA_HOST)
#         host libero_in_lab  -> /libero_in_lab      (LIBERO_IN_LAB_HOST, optional)
#
#   BACKEND=pi05             verl-vla-arena:v0.2     (non-root isaac-sim, uid 1234)
#       No GR00T deps. Mounts only the host verl-vla repo -> /workspaces/verl-vla.
#
# See README.md in this folder for the full path / variable reference and
# copy-paste command recipes.
#
# ─────────────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   # GR00T GR1 fridge eval (defaults):
#   examples/arena_sac/run_docker.sh
#
#   # GR00T LIBERO spatial task 3 eval:
#   EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_eval.sh ARENA_TASK=libero \
#     MODELS_HOST=~/iDataset/VLA/gr00t/libero_all_suites_rel_rotvec \
#     examples/arena_sac/run_docker.sh
#
#   # GR00T GR1 SAC train:
#   EVAL_SCRIPT=examples/arena_sac/run_gr00t_arena_sac.sh ARENA_TASK=gr1 \
#     MODELS_HOST=~/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out \
#     OUTPUT_ROOT=/eval/outputs/arena_gr00t_gr1_sac \
#     examples/arena_sac/run_docker.sh
#
#   # PI0.5 G1 eval:
#   BACKEND=pi05 examples/arena_sac/run_docker.sh
#
#   # PI0.5 G1 teleop smoke:
#   BACKEND=pi05 EVAL_SCRIPT=examples/arena_sac/run_pi05_arena_g1_sac_smoke.sh \
#     examples/arena_sac/run_docker.sh
#
#   # Just (re)start the container / drop into a shell:
#   examples/arena_sac/run_docker.sh --shell
#   examples/arena_sac/run_docker.sh --no-run
#
# ─────────────────────────────────────────────────────────────────────────────
# Common overrides (env vars) — see README.md for the full table
# ─────────────────────────────────────────────────────────────────────────────
#   BACKEND            gr00t | pi05                (default: gr00t)
#   IMAGE              docker image                (backend default)
#   CONTAINER_NAME     container name              (backend default)
#   RECREATE=1         force remove + recreate the container
#   EVAL_SCRIPT        inner script (relative to the repo inside the container)
#   MAX_EPISODES       episodes to evaluate        (default 10; ignored by train)
#   ARENA_TASK         gr1 | libero                (forwarded to gr00t inner scripts)
#   OUTPUT_ROOT        eval/train output root inside the container
#   -- gr00t only --
#   MODELS_HOST        host checkpoint parent      -> /models
#   GROOT_MODEL_PATH   checkpoint path inside the container
#   ARENA_HOST         host IsaacLab-Arena checkout -> /workspaces/isaaclab_arena
#   LIBERO_IN_LAB_HOST host libero_in_lab checkout -> /libero_in_lab
#   -- pi05 only --
#   MODEL_PATH         policy checkpoint inside the container
#   MOUNT_REPO=0       do NOT bind-mount the host repo (use the image's baked copy)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

BACKEND="${BACKEND:-gr00t}"

MODE="run"
case "${1:-}" in
  --shell)  MODE="shell" ;;
  --no-run) MODE="none" ;;
  "")       MODE="run" ;;
  *) echo "Unknown option: $1" >&2; exit 1 ;;
esac

log() { echo -e "\033[1;35m[run_docker:$BACKEND]\033[0m $*"; }

RECREATE="${RECREATE:-0}"
MAX_EPISODES="${MAX_EPISODES:-10}"

# ─────────────────────────────────────────────────────────────────────────────
# Backend-specific configuration.
# ─────────────────────────────────────────────────────────────────────────────
DOCKER_MOUNT_ARGS=()
DOCKER_ENV_ARGS=(-e PYTHONDONTWRITEBYTECODE=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y)

case "$BACKEND" in
  gr00t)
    IMAGE="${IMAGE:-isaaclab_arena:cuda_gr00t_gn16}"
    CONTAINER_NAME="${CONTAINER_NAME:-isaaclab_arena-cuda_gr00t_gn16}"
    WORKDIR="${WORKDIR:-/eval}"
    EVAL_SCRIPT="${EVAL_SCRIPT:-examples/arena_sac/run_gr00t_arena_eval.sh}"

    # Checkpoint parent on the host -> /models. Default matches the ranch GR1 fridge finetune.
    MODELS_HOST="${MODELS_HOST:-$HOME/iDataset/VLA/gr00t/ranch_finetune_newcam_wrist_out}"
    GROOT_MODEL_PATH="${GROOT_MODEL_PATH:-/models/checkpoint-5000}"
    # Local IsaacLab-Arena checkout -> /workspaces/isaaclab_arena. Required so the host
    # branch's wrist-cam / env patches win over the image's (often stale) baked copy.
    ARENA_HOST="${ARENA_HOST:-$HOME/Projects/libero_rl_example/IsaacLab-Arena}"
    ARENA_WORKDIR="${ARENA_WORKDIR:-/workspaces/isaaclab_arena}"
    # LIBERO-in-Lab assets (required for arena_libero evals; harmless for GR1).
    LIBERO_IN_LAB_HOST="${LIBERO_IN_LAB_HOST:-$HOME/Projects/libero_rl_example/libero_in_lab}"
    LIBERO_IN_LAB_WORKDIR="${LIBERO_IN_LAB_WORKDIR:-/libero_in_lab}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/outputs/arena_gr00t_gr1_eval}"

    mkdir -p "$HOST_REPO/outputs"
    chmod 777 "$HOST_REPO/outputs" 2>/dev/null || true
    DOCKER_MOUNT_ARGS+=(-v "$HOST_REPO:$WORKDIR")

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
    ;;

  pi05)
    IMAGE="${IMAGE:-vemlp-demo-cn-beijing.cr.volces.com/verl/verl-vla-arena:v0.2}"
    CONTAINER_NAME="${CONTAINER_NAME:-verl-vla-arena}"
    # Non-root user baked into the Isaac Sim image (uid 1234).
    CONTAINER_USER="${CONTAINER_USER:-isaac-sim}"
    WORKDIR="${WORKDIR:-/workspaces/verl-vla}"
    EVAL_SCRIPT="${EVAL_SCRIPT:-examples/arena_sac/run_pi05_arena_g1_eval.sh}"
    MODEL_PATH="${MODEL_PATH:-/workspaces/models/torch_pi05_base}"
    # Bind-mount the host repo (so videos/metrics/certs persist on the host).
    MOUNT_REPO="${MOUNT_REPO:-1}"

    if [[ "$MOUNT_REPO" == "1" ]]; then
      log "Mounting host repo '$HOST_REPO' -> '$WORKDIR'"
      # certs/ is needed by the teleop smoke; outputs/ by everything.
      mkdir -p "$HOST_REPO/outputs" "$HOST_REPO/certs"
      chmod 777 "$HOST_REPO/outputs" "$HOST_REPO/certs"
      DOCKER_MOUNT_ARGS+=(-v "$HOST_REPO:$WORKDIR")
    fi
    ;;

  *)
    echo "Unknown BACKEND='$BACKEND' (expected: gr00t | pi05)" >&2
    exit 1
    ;;
esac

# Do NOT bind-mount host /tmp -> /tmp: host files owned by another uid cause
# "Permission denied" inside the container. Use the image's own /tmp.

# ─────────────────────────────────────────────────────────────────────────────
# (Re)create the container with all GPUs.
# Bypass the image entrypoint and keep a long-lived bash so we can docker exec
# repeatedly.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$RECREATE" == "1" ]]; then
  log "Removing existing container '$CONTAINER_NAME' (RECREATE=1)"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Recreate if a running container is missing any mount we requested.
if [[ "$RECREATE" != "1" ]] && docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  mounts="$(docker inspect -f '{{range .Mounts}}{{println .Source .Destination}}{{end}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  need_recreate=0
  for i in "${!DOCKER_MOUNT_ARGS[@]}"; do
    [[ "${DOCKER_MOUNT_ARGS[$i]}" == "-v" ]] || continue
    spec="${DOCKER_MOUNT_ARGS[$((i + 1))]}"          # host:dest[:ro]
    host="${spec%%:*}"
    rest="${spec#*:}"
    dest="${rest%%:*}"
    if ! grep -qx "$host $dest" <<<"$mounts"; then
      log "Running container is missing mount ($host -> $dest); recreating"
      need_recreate=1
      break
    fi
  done
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

# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks.
# ─────────────────────────────────────────────────────────────────────────────
log "Container user: $(docker exec "$CONTAINER_NAME" whoami)"
log "GPUs visible inside container:"
docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=index,name,memory.total --format=csv

if [[ "$BACKEND" == "gr00t" ]]; then
  if docker exec "$CONTAINER_NAME" test -d /opt/groot_deps; then
    log "Found /opt/groot_deps (GR00T deps OK)"
  else
    log "WARNING: /opt/groot_deps missing — is IMAGE really the cuda_gr00t_gn16 build?"
  fi
  # Confirm the bind-mounted Arena (not the baked image copy) has the wrist-cam patch.
  ARENA_ENV_FILE="$ARENA_WORKDIR/isaaclab_arena_environments/gr1_put_and_close_door_environment.py"
  if docker exec "$CONTAINER_NAME" test -f "$ARENA_ENV_FILE"; then
    if docker exec "$CONTAINER_NAME" grep -q 'GR1T2WristCameraCfg' "$ARENA_ENV_FILE"; then
      log "Arena mount OK: wrist-cam patch present at $ARENA_WORKDIR"
    else
      log "WARNING: Arena at $ARENA_WORKDIR has no GR1T2WristCameraCfg — check ARENA_HOST branch"
    fi
  else
    log "WARNING: Arena env file missing under $ARENA_WORKDIR"
  fi
elif [[ "$BACKEND" == "pi05" && "$MOUNT_REPO" != "1" ]]; then
  # Baked copy is root-owned; make it writable by the non-root container user.
  log "Chowning $WORKDIR to $CONTAINER_USER (baked copy)"
  docker exec -u root "$CONTAINER_NAME" chown -R "$CONTAINER_USER:$CONTAINER_USER" "$WORKDIR"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run inner script / shell / nothing.
# ─────────────────────────────────────────────────────────────────────────────
# Allocate a TTY only when stdin is a terminal (avoids "the input device is not a
# TTY" when this helper is driven by CI / nohup / pipes).
DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS+=(-t)
fi

case "$MODE" in
  none)
    log "Container ready. Attach with: docker exec -ti $CONTAINER_NAME bash"
    ;;
  shell)
    if [[ "$BACKEND" == "pi05" ]]; then
      log "Dropping into an interactive shell as $CONTAINER_USER (workdir=$WORKDIR)"
      docker exec -ti -u "$CONTAINER_USER" -w "$WORKDIR" "$CONTAINER_NAME" bash
    else
      log "Dropping into an interactive shell (workdir=$WORKDIR)"
      docker exec -ti -w "$WORKDIR" "$CONTAINER_NAME" bash
    fi
    ;;
  run)
    if [[ "$BACKEND" == "gr00t" ]]; then
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
        -e ARENA_TASK="${ARENA_TASK:-}" \
        -e TASK_SUITE="${TASK_SUITE:-}" \
        -e TASK_ID="${TASK_ID:-}" \
        "$CONTAINER_NAME" \
        bash "$EVAL_SCRIPT"
      HOST_OUTPUT="${OUTPUT_ROOT/#$WORKDIR/$HOST_REPO}"
      log "Outputs on host: $HOST_OUTPUT"
    else
      log "Running $EVAL_SCRIPT (MODEL_PATH=$MODEL_PATH, MAX_EPISODES=$MAX_EPISODES)"
      docker exec -i "${DOCKER_TTY_ARGS[@]}" -u "$CONTAINER_USER" -w "$WORKDIR" \
        -e MODEL_PATH="$MODEL_PATH" \
        -e MAX_EPISODES="$MAX_EPISODES" \
        "$CONTAINER_NAME" \
        bash "$EVAL_SCRIPT"
      if [[ "$MOUNT_REPO" == "1" ]]; then
        log "Outputs saved on host under: $HOST_REPO/outputs"
      fi
    fi
    ;;
esac
