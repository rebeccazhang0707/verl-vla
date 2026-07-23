set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "$REPO_ROOT"

OUTPUT_DIR="/file_system/liujincheng/output/act_libero_sft_native_loss"
MODEL_PATH="${ACT_MODEL_PATH:-$REPO_ROOT/assets/hf_models/act_libero}"

SFT_REPO_ID="$REPO_ROOT/outputs/record/lerobot/local/libero_spatial"
SFT_REVISION="main"
SFT_BATCH_SIZE=32
SFT_NUM_WORKERS=8
SFT_PREFETCH_FACTOR=8
SFT_PERSISTENT_WORKERS=True
SFT_PIN_MEMORY=True
SFT_VIDEO_BACKEND="pyav"

NUM_GPUS=4
NUM_NODES=1

TOTAL_EPOCHS=10000
MINI_BATCH_SIZE=32
MICRO_BATCH_SIZE=16
LR=1e-4
SAVE_FREQ=500
MAX_ACTOR_CKPT_TO_KEEP=3

PROJECT_NAME="act-libero-sft"
EXPERIMENT_NAME="libero_sft_native_loss"

PYTHON=python

$PYTHON -m verl_vla.entrypoints.train.sft \
    --config-path "$SCRIPT_DIR" \
    --config-name act_sft \
    "hydra.searchpath=[file://$REPO_ROOT/src/verl_vla/workflows/config]" \
    cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
    data.repo_id="$SFT_REPO_ID" \
    data.revision="$SFT_REVISION" \
    data.batch_size=$SFT_BATCH_SIZE \
    data.drop_last=True \
    data.num_workers=$SFT_NUM_WORKERS \
    data.prefetch_factor=$SFT_PREFETCH_FACTOR \
    data.persistent_workers=$SFT_PERSISTENT_WORKERS \
    data.pin_memory=$SFT_PIN_MEMORY \
    data.video_backend="$SFT_VIDEO_BACKEND" \
    cluster.resource.model.nnodes=$NUM_NODES \
    cluster.resource.model.gpus_per_node=$NUM_GPUS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    cluster.actor_rollout_ref.actor.mini_batch_size=$MINI_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.micro_batch_size=$MICRO_BATCH_SIZE \
    cluster.actor_rollout_ref.actor.optim.lr=$LR \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="['console','tensorboard']" \
    cluster.checkpoint.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq=$SAVE_FREQ \
    cluster.checkpoint.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    "$@"
