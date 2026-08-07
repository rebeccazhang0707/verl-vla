#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MAIN_PYTHON="${PYTHON:-$PROJECT_ROOT/.env/bin/python}"

if [[ ! -x "$MAIN_PYTHON" ]]; then
  echo "Missing verl-vla Python environment: $MAIN_PYTHON" >&2
  echo "Install it with: uv pip install --python .env/bin/python -e '.[piper]'" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

overrides=(
  "cluster.resource.env.device=cpu"
  "cluster.resource.env.workers_per_node=1"
  "cluster.resource.env.gpus_per_node=0"
  "cluster.env.env_worker.simulator.simulator_type=piper"
)

exec "$MAIN_PYTHON" -m verl_vla.entrypoints.teleop "${overrides[@]}" "$@"
