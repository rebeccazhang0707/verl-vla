# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a placeholder prompt dataset (train/val parquet) for GR00T + Isaac Lab Arena SAC.

Unlike LIBERO (``prepare_libero_dataset.py``), Arena is **single-task** and resets to a
**random** scene layout each episode — the env ignores the per-row ``state_ids`` value
(``IsaacLabArenaEnv.reset_envs_to_state_ids`` re-inits the Arena scene regardless). The
dataset therefore only needs to:

  * drive the dataloader length (one row == one prompt), and
  * carry the ``state_ids`` / ``task_ids`` columns the SAC trainer reads from
    ``non_tensor_batch`` (``sac_ray_trainer._reset_envs`` / ``_restructure_*``) and hands to
    ``reset_envs_to_state_ids`` (whose length must equal ``num_envs * stage_num`` per worker).

Schema mirrors ``prepare_libero_dataset.py`` (``data_source / prompt / state_ids / task_ids /
ability / extra_info``) so the existing RL dataset/dataloader path consumes it unchanged.

Single-task convention:
  * ``task_ids = 0`` for every row (the model / critic ignore the value — single-task).
  * ``state_ids = 0 .. N-1`` (distinct placeholders; Arena ignores their value but the
    env-loop restructure needs one id per env slot).

Row-count constraint (see ``examples/arena_sac/run_gr00t_arena_sac*.sh``):
  ``TRAIN_BATCH_SIZE * ROLLOUT_N == NUM_ENV_GPUS * NUM_STAGE * NUM_ENV``.
  ``--num_train`` should be a multiple of ``TRAIN_BATCH_SIZE`` (>= 1 batch); ``--num_val``
  likewise of the val batch size. Pure ``datasets`` / ``pandas`` — no Isaac / gr00t / torch.
"""

import argparse
import os

from datasets import Dataset

# Default task wording matches the gr1_ranch_bottle_into_fridge checkpoint /
# rob_sac_trainer_arena_gr00t.yaml; override with --prompt for other Arena envs.
DEFAULT_ARENA_ENV_NAME = "put_item_in_fridge_and_close_door"
DEFAULT_PROMPT = "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."


def build_rows(num_rows: int, data_source: str, split: str, prompt: str, arena_env_name: str) -> list[dict]:
    """Build ``num_rows`` placeholder rows (single task_id=0, distinct state_ids)."""
    rows = []
    for idx in range(num_rows):
        rows.append(
            {
                "data_source": data_source,
                "prompt": prompt,
                # Single-task: constant task id. Distinct dummy state ids (Arena ignores
                # the value but the env-loop restructure expects one id per env slot).
                "state_ids": idx,
                "task_ids": 0,
                "ability": "robot",
                "extra_info": {
                    "split": split,
                    "state_ids": idx,
                    "index": idx,
                    "task_ids": 0,
                    "arena_env_name": arena_env_name,
                    "task_description": prompt,
                },
            }
        )
    return rows


def build_dataset(
    local_save_dir: str,
    arena_env_name: str,
    prompt: str,
    num_train: int,
    num_val: int,
) -> None:
    save_dir = os.path.join(os.path.expanduser(local_save_dir), arena_env_name)
    os.makedirs(save_dir, exist_ok=True)

    train_rows = build_rows(num_train, data_source="train", split="train", prompt=prompt, arena_env_name=arena_env_name)
    val_rows = build_rows(num_val, data_source="validation", split="test", prompt=prompt, arena_env_name=arena_env_name)

    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)

    train_path = os.path.join(save_dir, "train.parquet")
    val_path = os.path.join(save_dir, "test.parquet")
    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print("\n--- Arena placeholder dataset ---")
    print(f"arena_env_name : {arena_env_name}")
    print(f"prompt         : {prompt}")
    print(f"train rows     : {len(train_dataset)} -> {train_path}")
    print(f"val rows       : {len(val_dataset)} -> {val_path}")
    print("task_ids       : all 0 (single task)")
    print("state_ids      : 0 .. N-1 (placeholders; Arena ignores the value)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/arena_rl",
        help="Directory for the preprocessed dataset (a per-env subdir is created under it).",
    )
    parser.add_argument(
        "--arena_env_name",
        default=DEFAULT_ARENA_ENV_NAME,
        help="Arena example-environment name (used for the output subdir + extra_info).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Task description string stored in every row (the live task comes from the Arena env).",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=64,
        help="Number of train rows. Should be a multiple of the run's TRAIN_BATCH_SIZE.",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=8,
        help="Number of validation rows.",
    )
    args = parser.parse_args()

    build_dataset(
        local_save_dir=args.local_save_dir,
        arena_env_name=args.arena_env_name,
        prompt=args.prompt,
        num_train=args.num_train,
        num_val=args.num_val,
    )
