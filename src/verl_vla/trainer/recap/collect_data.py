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

import logging

import hydra
import numpy as np
import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf
from verl import DataProto

from verl_vla.trainer.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options

logger = logging.getLogger(__name__)

DEFAULT_TASK_ID = 0
DEFAULT_STATE_ID = 0


def get_collect_config(config):
    return OmegaConf.select(config, "recap.collect_data", default=config)


def collect_recap_env_data(config):
    """Run RECAP env data collection and return existing/new datasets."""
    ensure_ray_initialized(config)
    collect_config = get_collect_config(config)
    remote_options = get_controller_remote_options(collect_config)
    return ray.get(run_env_loop.options(**remote_options).remote(config))


@ray.remote
def run_env_loop(config):
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    collect_config = get_collect_config(config)
    cluster = TrainCluster(instantiate(collect_config.cluster, _recursive_=False))
    cluster.start()
    try:
        collected_datasets = {}
        for rollout_idx in range(int(collect_config.num_rollouts)):
            prompts = build_rollout_prompts(collect_config, rollout_idx)
            rollout_output, collected_datasets = cluster.rollout(prompts)
            logger.info(
                "Finished recap env loop rollout %s: %s", rollout_idx, rollout_output.meta_info.get("metrics", {})
            )

        return collected_datasets
    finally:
        cluster.shutdown()


def build_rollout_prompts(collect_config, global_step: int) -> DataProto:
    env_resource = collect_config.cluster.resource.env
    env_workers_per_node = (
        int(env_resource.workers_per_node) if env_resource.device == "cpu" else int(env_resource.gpus_per_node)
    )
    env_worker_world_size = int(env_resource.nnodes) * env_workers_per_node
    num_envs_per_worker = int(collect_config.cluster.env.env_worker.num_envs)
    pipeline_stage_num = int(collect_config.cluster.env.env_loop.pipeline_stage_num)
    total_envs = env_worker_world_size * num_envs_per_worker * pipeline_stage_num
    task_ids = np.full(total_envs, DEFAULT_TASK_ID, dtype=np.int64)
    state_ids = np.full(total_envs, DEFAULT_STATE_ID, dtype=np.int64)
    return DataProto.from_dict(
        non_tensors={
            "state_ids": state_ids,
            "task_ids": task_ids,
        },
        meta_info={
            "task_ids": task_ids,
            "global_steps": global_step,
        },
    )


@hydra.main(config_path="../config", config_name="rob_recap_trainer", version_base=None)
def main(config):
    collected_datasets = collect_recap_env_data(config)
    logger.info("RECAP collect finished: %s", collected_datasets)
    print(collected_datasets)


if __name__ == "__main__":
    main()
