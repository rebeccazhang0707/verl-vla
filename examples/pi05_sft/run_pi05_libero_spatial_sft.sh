#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=$REPO_ROOT/.data/pi05_sft
MODEL_PATH=${MODEL_PATH:-Miical/pi05-base}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
SFT_ROOT=$DATA_ROOT/datasets/libero_spatial_image
NORM_STATS_PATH=$SFT_ROOT/norm_stats.json
OUTPUT_DIR=$DATA_ROOT/output/pi05_libero_spatial_sft

if [[ "${MODEL_PATH}" == /* && ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Pi0.5 model config not found: ${MODEL_PATH}/config.json" >&2
  exit 2
fi

if [[ "${TOKENIZER_PATH}" == /* && ! -f "${TOKENIZER_PATH}/tokenizer.json" ]]; then
  echo "Pi0.5 tokenizer not found: ${TOKENIZER_PATH}" >&2
  exit 2
fi

if [[ ! -f "${SFT_ROOT}/meta/info.json" ]]; then
  echo "LeRobot dataset metadata not found: ${SFT_ROOT}/meta/info.json" >&2
  exit 2
fi

if [[ ! -f "$NORM_STATS_PATH" ]]; then
  echo "Normalization statistics not found: $NORM_STATS_PATH" >&2
  echo "Generate them with scripts/compute_norm_stats.py before training." >&2
  exit 2
fi

python3 -m verl_vla.entrypoints.train.sft \
  hydra.run.dir="$OUTPUT_DIR/hydra" \
  cluster.actor_rollout_ref.model.path="$MODEL_PATH" \
  cluster.actor_rollout_ref.model.tokenizer_path="$TOKENIZER_PATH" \
  cluster.actor_rollout_ref.model.enable_gradient_checkpointing=False \
  cluster.actor_rollout_ref.model.use_remove_padding=False \
  cluster.actor_rollout_ref.model.trust_remote_code=False \
  cluster.actor_rollout_ref.model.adapter.embodiment=libero \
  cluster.actor_rollout_ref.model.adapter.norm_stats_path="$NORM_STATS_PATH" \
  cluster.actor_rollout_ref.model.adapter.critic.enabled=False \
  cluster.actor_rollout_ref.actor.strategy=fsdp2 \
  cluster.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  cluster.actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  cluster.actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
  cluster.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="[SiglipEncoderLayer,GemmaDecoderLayerWithExpert]" \
  cluster.actor_rollout_ref.actor.mini_batch_size=64 \
  cluster.actor_rollout_ref.actor.micro_batch_size=1 \
  cluster.actor_rollout_ref.actor.optim.lr=1e-4 \
  cluster.actor_rollout_ref.actor.optim.weight_decay=1e-5 \
  cluster.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
  cluster.resource.model.gpus_per_node=8 \
  cluster.resource.model.nnodes=1 \
  cluster.checkpoint.resume_mode=disable \
  cluster.checkpoint.default_local_dir="$OUTPUT_DIR" \
  data.repo_id=lerobot/libero_spatial_image \
  data.root="$SFT_ROOT" \
  data.revision=main \
  data.batch_size=64 \
  data.drop_last=True \
  data.num_workers=8 \
  data.video_backend=pyav \
  data.action_delta_steps=50 \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.save_last=False \
  trainer.resume_dataloader_state=False \
  trainer.project_name=pi05-libero-sft \
  trainer.experiment_name=pi05_libero_spatial_sft \
  trainer.logger="['console']" \
  "$@"
