#!/usr/bin/env bash
#
# PI0.5 Arena G1 SAC *smoke* flow: teleop-driven data collection (RECAP
# collect_data stage) that exercises the pi05 rollout + LeRobot/video recorder
# end to end. Generates a local HTTPS cert for the WebXR teleop server, then
# records a few short episodes. No SAC gradient step — this is a plumbing check.
#
# Runs inside the verl-vla-arena image (NOT the GR00T image). Launch from host:
#
#   BACKEND=pi05 EVAL_SCRIPT=examples/arena_sac/run_pi05_arena_g1_sac_smoke.sh \
#     examples/arena_sac/run_docker.sh
#
# Overridable via env vars:
#   MODEL_PATH   policy checkpoint (HF-format dir)
#   OUTPUT_ROOT  where lerobot dataset + videos are written
#   CERT_DIR     dir for the generated teleop TLS cert/key
#
# See README.md for the full path / variable reference.
#
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:/workspaces/isaaclab_arena:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/isaac-sim/python.sh}"
MODEL_PATH="${MODEL_PATH:-/workspaces/models/torch_pi05_base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_g1_smoke}"

# Generate the local HTTPS cert used by the browser/WebXR teleop server:
CERT_DIR="${CERT_DIR:-$REPO_ROOT/certs}"
SSL_CERTFILE="${SSL_CERTFILE:-$CERT_DIR/teleop.crt}"
SSL_KEYFILE="${SSL_KEYFILE:-$CERT_DIR/teleop.key}"
if [[ ! -f "$SSL_CERTFILE" || ! -f "$SSL_KEYFILE" ]]; then
  mkdir -p "$CERT_DIR"
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$SSL_KEYFILE" \
    -out "$SSL_CERTFILE" \
    -days 3650 \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
fi

"$PYTHON" -m verl_vla.entrypoints.train.recap \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  "recap.policy_eval.enable=false" \
  "recap.collect_data.enable=true" \
  "recap.compute_return.enable=false" \
  "recap.train_value_model.enable=false" \
  "recap.value_infer.enable=false" \
  "recap.train_policy.enable=false" \
  "recap.collect_data.max_episodes=10" \
  "recap.collect_data.cluster.env.env_loop.max_interactions=32" \
  "recap.collect_data.cluster.env.env_worker.auto_reset=true" \
  "recap.collect_data.cluster.env.env_worker.simulator_start_timeout_s=600" \
  "recap.collect_data.cluster.env.env_worker.simulator.simulator_type=arena" \
  "recap.collect_data.cluster.env.env_worker.simulator.arena.max_episode_steps=256" \
  "recap.collect_data.cluster.actor_rollout_ref.model.path=$MODEL_PATH" \
  "recap.collect_data.cluster.actor_rollout_ref.model.tokenizer_path=$MODEL_PATH" \
  "recap.collect_data.cluster.actor_rollout_ref.model.override_config.policy_type=arena" \
  "recap.collect_data.cluster.actor_rollout_ref.rollout.output_critic_value=false" \
  "recap.collect_data.cluster.env.env_worker.teleop.enable=true" \
  "recap.collect_data.cluster.env.env_worker.teleop.devices=[xr_controller]" \
  "recap.collect_data.cluster.env.env_worker.teleop.server.ssl_certfile=$SSL_CERTFILE" \
  "recap.collect_data.cluster.env.env_worker.teleop.server.ssl_keyfile=$SSL_KEYFILE" \
  "recap.collect_data.cluster.env.env_worker.recorder.lerobot.root=$OUTPUT_ROOT/lerobot" \
  "recap.collect_data.cluster.env.env_worker.recorder.lerobot.repo_id=local/arena_g1_smoke" \
  "recap.collect_data.cluster.env.env_worker.recorder.video.root=$OUTPUT_ROOT/videos" \
  "$@"
