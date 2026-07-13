#!/usr/bin/env bash
#
# Run PI0.5 policy EVALUATION on the Arena G1 task via the RECAP policy_eval
# stage, saving rollout videos to disk. No teleop / no dataset recording.
#
# Runs inside the verl-vla-arena image (NOT the GR00T image). Launch from host:
#
#   BACKEND=pi05 examples/arena_sac/run_docker.sh
#
# See README.md for the full path / variable reference.
#
# Overridable via env vars:
#   MODEL_PATH       policy checkpoint (HF-format dir)
#   OUTPUT_ROOT      where videos + eval metrics are written
#   MAX_EPISODES     number of episodes to evaluate (Arena benchmark size is 1,
#                    so this must be set explicitly to eval more than 1)
#   MAX_INTERACTIONS env_loop interactions per rollout
#
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:/workspaces/isaaclab_arena:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/isaac-sim/python.sh}"
MODEL_PATH="${MODEL_PATH:-/workspaces/models/torch_pi05_base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/arena_g1_eval}"
MAX_EPISODES="${MAX_EPISODES:-10}"
MAX_INTERACTIONS="${MAX_INTERACTIONS:-32}"

"$PYTHON" -m verl_vla.entrypoints.train.recap \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=INFO" \
  "recap.policy_eval.enable=true" \
  "recap.collect_data.enable=false" \
  "recap.compute_return.enable=false" \
  "recap.train_value_model.enable=false" \
  "recap.value_infer.enable=false" \
  "recap.train_policy.enable=false" \
  "recap.policy_eval.model_path=$MODEL_PATH" \
  "recap.policy_eval.max_episodes=$MAX_EPISODES" \
  "recap.policy_eval.result_dir=$OUTPUT_ROOT/eval_metrics" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.path=$MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.tokenizer_path=$MODEL_PATH" \
  "recap.policy_eval.cluster.actor_rollout_ref.model.override_config.policy_type=arena" \
  "recap.policy_eval.cluster.actor_rollout_ref.rollout.output_critic_value=false" \
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
