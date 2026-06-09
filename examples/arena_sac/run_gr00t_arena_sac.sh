#!/usr/bin/env bash
# Launch GR00T (N1.6) + Isaac Lab Arena + verl-vla SAC on a SINGLE node (combined
# train/rollout worker group; env workers on their own GPUs).
#
# Mirrors examples/libero_sac/run_pi05_libero_sac.sh but for the Arena/GR00T channel:
#   * rollout.name=gr00t          -> GR00TRolloutRob (registered string path)
#   * env.train.simulator_type=arena -> IsaacLabArenaEnv (env packs obs + decodes actions)
#   * ENV_DEVICE=cuda             -> Arena runs Isaac Sim; CPU env workers are NOT supported
#                                    (no MUJOCO_GL / osmesa block — that is LIBERO-only).
#
# Authoritative field values mirror the source config + run script
#   verl/verl/experimental/vla/config/rob_sac_trainer_arena_gr00t.yaml
#   verl/verl/experimental/vla/run_gr00t_arena_sac.sh
# translated to the verl-vla SAC config tree (sac_* actor fields, override_config).
#
# Usage:
#   SFT_MODEL_PATH=/path/to/gr1_export bash examples/arena_sac/run_gr00t_arena_sac.sh
#
# ── MANDATORY pre-docker compose re-check (P1-4, docker-only — NOT runnable on the
#    CPU dev host; the CPU "reb_verl" proxy compose does NOT count) ──
#   Before the first real docker run, validate the Hydra config resolves under the
#   REAL verl/verl-vla install with the exact override set below:
#     python -m verl_vla.trainer.main_sac --config-name rob_sac_trainer_arena_gr00t \
#         <all overrides from this script> --cfg job
#   Confirm fields WITHOUT a leading "+" already exist in the merged config (no
#   "Could not override ... use +" error), in particular
#   `actor_rollout_ref.actor.grad_clip=1` and `actor_rollout_ref.actor.sac.bc_loss_coef=0.0`.
#   Newly-added keys here use the correct prefix: step_penalty (declared, no +),
#   env_spacing / action_horizon (not declared, use +).
set -x

# Placeholder prompt dataset (drives dataloader length + env reset task/state ids).
# Build it with: python scripts/prepare_arena_dataset.py --num_train <BATCH_SIZE>
ARENA_DATA_DIR=${ARENA_DATA_DIR:-"$HOME/data/arena_rl/put_item_in_fridge_and_close_door"}
train_files=${TRAIN_FILES:-"$ARENA_DATA_DIR/train.parquet"}
test_files=${VAL_FILES:-"$ARENA_DATA_DIR/test.parquet"}

OUTPUT_DIR=${OUTPUT_DIR:-"$HOME/models/vla_arena_gr00t_sac"}
VIDEO_OUTPUT=${OUTPUT_DIR}/video
# GR00T N1.6 (Gr00tN1d6) export dir (config.json + embodiment_id.json + processor assets).
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"$HOME/checkpoints/gr1_ranch_bottle_into_fridge/checkpoint-5000-export"}
TOKENIZER_PATH="$SFT_MODEL_PATH"

# Physical Node Config
NUM_NODES=1                                    # number of nodes
NUM_GPUS=8                                     # total number of gpus per node

# Role Config -- Arena requires GPU env workers (Isaac Sim).
ENV_DEVICE=cuda                                # env worker device: MUST be cuda for Arena
NUM_ENV_WORKERS=4                              # number of (GPU) env workers per node
NUM_ROLLOUT_GPUS=4                             # number of gpus for actor/rollout workers per node

# Rollout Config
# NOTE: BATCH_SIZE * ROLLOUT_N == NUM_ENV_WORKERS * NUM_STAGE * NUM_ENV
ROLLOUT_N=8                                    # responses per prompt (Isaac: == NUM_ENV)
NUM_STAGE=2                                    # number of pipeline stages
NUM_ENV=8                                      # number of envs per env worker
BATCH_SIZE=$((NUM_ENV_WORKERS * NUM_STAGE * NUM_ENV / ROLLOUT_N))

# Single anchor: NUM_ACTION_CHUNKS feeds the env (env.actor.model.num_action_chunks),
# the rollout/model (actor_rollout_ref.model.num_action_chunks) and the critic horizon
# (override_config.critic_action_horizon). Constraint enforced at rollout-worker init:
#   critic_action_horizon <= NUM_ACTION_CHUNKS <= model action_horizon (checkpoint=50).
NUM_ACTION_CHUNKS=16
MAX_EPISODE_STEPS=512                          # max_interactions = MAX_EPISODE_STEPS / NUM_ACTION_CHUNKS
VIDEO_FPS=50                                   # 1/(sim_dt*decimation) = 1/((1/200)*4) = 50 Hz

# Arena env config (put_item_in_fridge_and_close_door; matches the SFT checkpoint).
ARENA_ENV_NAME=${ARENA_ENV_NAME:-"put_item_in_fridge_and_close_door"}
ARENA_OBJECT=${ARENA_OBJECT:-"ranch_dressing_hope_robolab"}
ARENA_EMBODIMENT=${ARENA_EMBODIMENT:-"gr1_joint"}   # joint-position control (gr1_pink is IK; NOT compatible)
ARENA_CAMERA=${ARENA_CAMERA:-"robot_pov_cam_rgb"}
KITCHEN_STYLE=${KITCHEN_STYLE:-2}

# Training Config
MINI_BATCH_SIZE=1024                           # SAC replay mini batch size
MICRO_BATCH_SIZE=16                            # SAC micro batch size per GPU (lower first on OOM)

# Flow-SDE exploration-noise knobs (see source run script).
FLOW_SDE_NOISE_LEVEL=${FLOW_SDE_NOISE_LEVEL:-0.02}
FLOW_SDE_ROLLOUT_NOISE_SCALE=${FLOW_SDE_ROLLOUT_NOISE_SCALE:-1.0}
FLOW_SDE_TRAIN_NOISE_SCALE=${FLOW_SDE_TRAIN_NOISE_SCALE:-1.0}

PROJECT_NAME="gr00t-arena-sac"
EXPERIMENT_NAME="arena_gr00t_single_node"

PYTHON=${PYTHON:-python}
export VERL_LOGGING_LEVEL=INFO

$PYTHON -m verl_vla.trainer.main_sac \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    env.train.num_envs=$NUM_ENV \
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
    env.train.step_penalty=0.001 \
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
    +actor_rollout_ref.model.override_config.freeze_vision_tower=True \
    +actor_rollout_ref.model.override_config.critic_pooling=attn \
    +actor_rollout_ref.model.override_config.critic_use_encoded_state=True \
    +actor_rollout_ref.model.override_config.critic_privileged_obs=True \
    +actor_rollout_ref.model.override_config.critic_privileged_obs_dim=4 \
    actor_rollout_ref.model.override_config.critic_prefix_attn_heads=8 \
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
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.sac_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.sac_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.critic_lr=1e-4 \
    actor_rollout_ref.actor.warm_rollout_steps=5 \
    actor_rollout_ref.actor.critic_warmup_steps=200 \
    actor_rollout_ref.actor.actor_ema_enabled=true \
    actor_rollout_ref.actor.actor_ema_decay=0.95 \
    actor_rollout_ref.actor.replay_pool_single_size=2000 \
    actor_rollout_ref.actor.replay_pool_save_interval=500 \
    actor_rollout_ref.actor.replay_pool_save_dir=$OUTPUT_DIR/replay_pools \
    actor_rollout_ref.actor.load_replay_pool=False \
    actor_rollout_ref.actor.sac.gamma=0.99 \
    actor_rollout_ref.actor.sac.tau=1.0 \
    actor_rollout_ref.actor.sac.initial_alpha=0.01 \
    actor_rollout_ref.actor.sac.alpha_type=exp \
    actor_rollout_ref.actor.sac.bc_loss_coef=0.0 \
    actor_rollout_ref.actor.sac.critic_replay_positive_sample_ratio=0.5 \
    actor_rollout_ref.actor.sac.actor_replay_positive_sample_ratio=0.8 \
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
    trainer.n_env_workers_per_node=$NUM_ENV_WORKERS \
    trainer.n_rollout_gpus_per_node=$NUM_ROLLOUT_GPUS \
    trainer.rollout_interval=20 \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=500 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1000 \
    trainer.val_only=False \
    trainer.val_before_train=False
