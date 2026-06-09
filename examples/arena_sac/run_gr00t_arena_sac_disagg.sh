#!/usr/bin/env bash
# Launch GR00T (N1.6) + Isaac Lab Arena + verl-vla SAC, DISAGGREGATED: env (Isaac Sim)
# workers and the FSDP actor/rollout get SEPARATE GPU pools (env_gpu_pool vs
# train_rollout_pool), sized by NUM_ENV_GPUS / NUM_ROLLOUT_GPUS.
#
# Mirrors examples/libero_sac/run_pi05_libero_sac_disagg.sh (env.disagg_sim.enable=True +
# +trainer.n_env_gpus_per_node / +trainer.n_rollout_gpus_per_node) and the source arena
# disagg script (verl/.../run_gr00t_arena_sac_disagg.sh), translated to the verl-vla
# SAC config tree. ENV_DEVICE is cuda (Arena needs Isaac Sim; no MUJOCO_GL block).
#
# Usage:
#   SFT_MODEL_PATH=/path/to/gr1_export bash examples/arena_sac/run_gr00t_arena_sac_disagg.sh
#
# ── MANDATORY pre-docker compose re-check (P1-4, docker-only — NOT runnable on the
#    CPU dev host; the CPU "reb_verl" proxy compose does NOT count) ──
#   Before the first real docker run, validate the Hydra config resolves under the
#   REAL verl/verl-vla install with the exact override set below:
#     python -m verl_vla.trainer.main_sac --config-name rob_sac_trainer_arena_gr00t \
#         <all overrides from this script> --cfg job
#   Confirm fields WITHOUT a leading "+" already exist in the merged config (no
#   "Could not override ... use +" error), in particular
#   `actor_rollout_ref.actor.grad_clip=1` and `actor_rollout_ref.actor.sac.bc_loss_coef=0.05`.
set -x

ARENA_DATA_DIR=${ARENA_DATA_DIR:-"$HOME/data/arena_rl/put_item_in_fridge_and_close_door"}
train_files=${TRAIN_FILES:-"$ARENA_DATA_DIR/train.parquet"}
test_files=${VAL_FILES:-"$ARENA_DATA_DIR/test.parquet"}

OUTPUT_DIR=${OUTPUT_DIR:-"$HOME/models/vla_arena_gr00t_sac"}
VIDEO_OUTPUT=${OUTPUT_DIR}/video
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/checkpoints/gr1_ranch_bottle_into_fridge/checkpoint-5000-export"}
TOKENIZER_PATH="$SFT_MODEL_PATH"

# Physical / role config -- separate sim and train GPU pools.
NUM_NODES=1                                    # train/rollout nodes
SIM_NODES=1                                    # sim nodes
NUM_GPUS=8                                     # total gpus per node
ENV_DEVICE=cuda                                # MUST be cuda for Arena
NUM_ENV_GPUS=4                                 # gpus for Isaac Sim env workers (env_gpu_pool)
NUM_ROLLOUT_GPUS=4                             # gpus for FSDP actor/rollout (train_rollout_pool)

# Rollout config
# NOTE: BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV
ROLLOUT_N=8                                    # responses per prompt (Isaac: == NUM_ENV)
NUM_STAGE=2                                    # pipeline stages
NUM_ENV=8                                      # envs per env worker
BATCH_SIZE=$((NUM_ENV_GPUS * NUM_STAGE * NUM_ENV / ROLLOUT_N))

# Single anchor (see run_gr00t_arena_sac.sh): feeds env + model + critic horizon.
NUM_ACTION_CHUNKS=16
MAX_EPISODE_STEPS=512
VIDEO_FPS=50

# Arena env config
ARENA_ENV_NAME=${ARENA_ENV_NAME:-"put_item_in_fridge_and_close_door"}
ARENA_OBJECT=${ARENA_OBJECT:-"ranch_dressing_hope_robolab"}
ARENA_EMBODIMENT=${ARENA_EMBODIMENT:-"gr1_joint"}
ARENA_CAMERA=${ARENA_CAMERA:-"robot_pov_cam_rgb"}
KITCHEN_STYLE=${KITCHEN_STYLE:-2}

# Training config
MINI_BATCH_SIZE=512
MICRO_BATCH_SIZE=32

FLOW_SDE_NOISE_LEVEL=${FLOW_SDE_NOISE_LEVEL:-0.065}
FLOW_SDE_ROLLOUT_NOISE_SCALE=${FLOW_SDE_ROLLOUT_NOISE_SCALE:-1.0}
FLOW_SDE_TRAIN_NOISE_SCALE=${FLOW_SDE_TRAIN_NOISE_SCALE:-1.0}

PROJECT_NAME="gr00t-arena-sac"
EXPERIMENT_NAME="arena_gr00t_disagg"

PYTHON=${PYTHON:-python}
export VERL_LOGGING_LEVEL=INFO

$PYTHON -m verl_vla.trainer.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
    env.disagg_sim.enable=True \
    env.disagg_sim.nnodes=$SIM_NODES \
    env.rollout.pipeline_stage_num=$NUM_STAGE \
    env.train.simulator_type=arena \
    env.train.device=$ENV_DEVICE \
    env.train.max_episode_steps=$MAX_EPISODE_STEPS \
    env.train.seed=42 \
    +env.train.gr00t_model_path=$SFT_MODEL_PATH \
    +env.train.embodiment_tag=gr1 \
    +env.train.arena_env_name=$ARENA_ENV_NAME \
    +env.train.arena_object=$ARENA_OBJECT \
    +env.train.arena_embodiment=$ARENA_EMBODIMENT \
    +env.train.camera_name=$ARENA_CAMERA \
    +env.train.kitchen_style=$KITCHEN_STYLE \
    +env.train.rl_success_reward=True \
    +env.train.render_on_chunk_boundary=True \
    env.train.step_penalty=0.0 \
    env.train.subtask_reward=True \
    env.train.dense_success_reward=False \
    env.train.critic_privileged_obs=True \
    +env.train.env_spacing=10.0 \
    env.train.video_cfg.save_video=True \
    env.train.video_cfg.video_base_dir=${VIDEO_OUTPUT} \
    +env.train.video_cfg.fps=${VIDEO_FPS} \
    env.actor.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    env.actor.model.action_dim=26 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.embodiment_tag=gr1 \
    +actor_rollout_ref.model.num_action_chunks=$NUM_ACTION_CHUNKS \
    +actor_rollout_ref.model.action_dim=26 \
    actor_rollout_ref.model.override_config.policy_type=gr00t \
    actor_rollout_ref.model.override_config.sac_enable=True \
    actor_rollout_ref.model.override_config.critic_head_num=10 \
    +actor_rollout_ref.model.override_config.action_dim=26 \
    +actor_rollout_ref.model.override_config.embodiment_id=20 \
    +actor_rollout_ref.model.override_config.critic_action_horizon=$NUM_ACTION_CHUNKS \
    +actor_rollout_ref.model.override_config.action_horizon=50 \
    +actor_rollout_ref.model.override_config.sac_action_train_dims="[[7,14],[20,26]]" \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    +actor_rollout_ref.model.override_config.critic_privileged_obs=True \
    +actor_rollout_ref.model.override_config.critic_privileged_obs_dim=4 \
    actor_rollout_ref.model.override_config.flow_sde_enable=True \
    actor_rollout_ref.model.override_config.flow_sde_noise_level=$FLOW_SDE_NOISE_LEVEL \
    actor_rollout_ref.model.override_config.flow_sde_rollout_noise_scale=$FLOW_SDE_ROLLOUT_NOISE_SCALE \
    actor_rollout_ref.model.override_config.flow_sde_train_noise_scale=$FLOW_SDE_TRAIN_NOISE_SCALE \
    actor_rollout_ref.model.override_config.flow_sde_initial_beta=1.0 \
    actor_rollout_ref.model.override_config.flow_sde_beta_min=0.02 \
    actor_rollout_ref.model.override_config.flow_sde_beta_schedule_T=4000 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,Siglip2EncoderLayer,BasicTransformerBlock,MultiEmbodimentActionEncoder,CategorySpecificMLP] \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.sac_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.sac_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.critic_lr=1e-4 \
    actor_rollout_ref.actor.warm_rollout_steps=5 \
    actor_rollout_ref.actor.critic_warmup_steps=200 \
    actor_rollout_ref.actor.actor_ema_enabled=true \
    actor_rollout_ref.actor.actor_ema_decay=0.95 \
    actor_rollout_ref.actor.replay_pool_single_size=6000 \
    actor_rollout_ref.actor.replay_pool_save_dir=$OUTPUT_DIR/replay_pools \
    actor_rollout_ref.actor.load_replay_pool=False \
    actor_rollout_ref.actor.sac.gamma=0.99 \
    actor_rollout_ref.actor.sac.tau=0.005 \
    actor_rollout_ref.actor.sac.initial_alpha=0.05 \
    actor_rollout_ref.actor.sac.bc_loss_coef=0.05 \
    actor_rollout_ref.actor.sac.critic_replay_positive_sample_ratio=0.5 \
    actor_rollout_ref.actor.sac.actor_replay_positive_sample_ratio=0.5 \
    actor_rollout_ref.rollout.mode=async_envloop \
    actor_rollout_ref.rollout.name=gr00t \
    actor_rollout_ref.rollout.output_critic_value=True \
    actor_rollout_ref.rollout.prompt_length=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=$NUM_GPUS \
    +trainer.n_env_gpus_per_node=$NUM_ENV_GPUS \
    +trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    +trainer.rollout_interval=20 \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=500 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1000 \
    trainer.val_only=False \
    trainer.val_before_train=False
