#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-.data/pi05_sft/output/pi05_libero_spatial_sft}
LATEST_STEP_FILE="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"

if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ ! -f "${LATEST_STEP_FILE}" ]]; then
        echo "No Pi0.5 SFT checkpoint found at ${CHECKPOINT_ROOT}." >&2
        echo "Run examples/fine_tuning/pi05/run_train.sh first, or set MODEL_PATH explicitly." >&2
        exit 1
    fi
    LATEST_STEP=$(<"${LATEST_STEP_FILE}")
    MODEL_PATH="${CHECKPOINT_ROOT}/global_step_${LATEST_STEP}/actor/huggingface"
fi

OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl/sac/pi05/libero-spatial-task-3}
NUM_GPUS=${NUM_GPUS:-8}
NUM_ENV_WORKERS=${NUM_ENV_WORKERS:-4}
NUM_ENVS_PER_WORKER=${NUM_ENVS_PER_WORKER:-8}
ENV_DEVICE=${ENV_DEVICE:-cpu}
ENV_GPUS=${ENV_GPUS:-0}

if [[ "${ENV_DEVICE}" == "cpu" ]]; then
    export MUJOCO_GL=${MUJOCO_GL:-osmesa}
else
    export MUJOCO_GL=${MUJOCO_GL:-egl}
fi
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}

python -m verl_vla.entrypoints.train.sac \
    hydra.run.dir="${OUTPUT_DIR}/hydra" \
    cluster.actor_rollout_ref.model.path="${MODEL_PATH}" \
    cluster.actor_rollout_ref.model.adapter.embodiment=libero \
    cluster.actor_rollout_ref.model.adapter.action_chunk_size=10 \
    cluster.actor_rollout_ref.model.adapter.critic.type=mean_pool \
    cluster.actor_rollout_ref.model.adapter.critic.hidden_dims='[512,256,128]' \
    cluster.actor_rollout_ref.model.adapter.flow_sde_enable=true \
    cluster.actor_rollout_ref.model.adapter.flow_sde_noise_level=0.2 \
    cluster.actor_rollout_ref.model.adapter.flow_sde_noise_schedule_enabled=false \
    cluster.actor_rollout_ref.model.adapter.flow_sde_rollout_noise_scale=0.0 \
    cluster.actor_rollout_ref.model.adapter.flow_sde_train_noise_scale=0.0 \
    cluster.actor_rollout_ref.actor.mini_batch_size=128 \
    cluster.actor_rollout_ref.actor.micro_batch_size=16 \
    cluster.actor_rollout_ref.actor.actor_update_interval=1 \
    cluster.actor_rollout_ref.actor.ema_decay=null \
    cluster.actor_rollout_ref.actor.optim.lr=5e-6 \
    cluster.actor_rollout_ref.actor.optim.warmup_style=constant \
    cluster.actor_rollout_ref.actor.sac.auto_entropy=false \
    cluster.actor_rollout_ref.actor.sac.initial_alpha=0.0 \
    cluster.actor_rollout_ref.actor.td3.enabled=false \
    cluster.actor_rollout_ref.actor.cql.enabled=true \
    cluster.actor_rollout_ref.actor.cql.alpha=0.5 \
    cluster.actor_rollout_ref.actor.cql.temperature=1.0 \
    cluster.actor_rollout_ref.actor.cql.noise_scale=1.0 \
    cluster.actor_rollout_ref.actor.critic.lr=1e-4 \
    cluster.actor_rollout_ref.actor.critic.tau=1.0 \
    cluster.actor_rollout_ref.actor.critic.force_target_tau_one_in_warmup=true \
    cluster.actor_rollout_ref.actor.critic.skip_update_when_actor_update=true \
    cluster.actor_rollout_ref.actor.critic.warmup_steps=200 \
    cluster.actor_rollout_ref.actor.critic.only_steps_after_rollout=0 \
    cluster.actor_rollout_ref.actor.replay.critic_positive_sample_ratio=0.5 \
    cluster.actor_rollout_ref.actor.replay.actor_positive_sample_ratio=0.9 \
    cluster.actor_rollout_ref.actor.replay.save_interval=-1 \
    cluster.actor_rollout_ref.actor.replay.save_dir="${OUTPUT_DIR}/replay" \
    cluster.actor_rollout_ref.rollout.n=2 \
    cluster.actor_rollout_ref.rollout.mode=async_envloop \
    cluster.actor_rollout_ref.rollout.prompt_length=512 \
    cluster.actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    cluster.actor_rollout_ref.rollout.free_cache_engine=false \
    cluster.actor_rollout_ref.rollout.output_critic_value=true \
    cluster.env.env_loop.pipeline_stage_num=2 \
    cluster.env.env_loop.max_interactions=20 \
    cluster.env.env_worker.num_envs="${NUM_ENVS_PER_WORKER}" \
    cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial \
    cluster.env.env_worker.simulator.libero.task_ids='[2]' \
    cluster.env.env_worker.simulator.libero.max_episode_steps=200 \
    cluster.resource.model.nnodes=1 \
    cluster.resource.model.gpus_per_node="${NUM_GPUS}" \
    cluster.resource.model.workers_per_node="${NUM_GPUS}" \
    cluster.resource.env.device="${ENV_DEVICE}" \
    cluster.resource.env.nnodes=1 \
    cluster.resource.env.gpus_per_node="${ENV_GPUS}" \
    cluster.resource.env.workers_per_node="${NUM_ENV_WORKERS}" \
    cluster.checkpoint.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.project_name=verl-vla \
    trainer.experiment_name=pi05-libero-spatial-task-3-sac \
    trainer.logger='["console"]' \
    trainer.total_training_steps=10000 \
    trainer.rollout_interval=200000 \
    trainer.rollout_times=1 \
    trainer.warm_rollout_steps=3 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.eval_episodes=10 \
    trainer.val_before_train=true \
    trainer.val_only=false \
    "$@"
