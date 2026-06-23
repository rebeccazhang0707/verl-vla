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
from pathlib import Path

import numpy as np
import ray
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import Role

from verl_vla.env_loop.env_loop import EnvLoop
from verl_vla.recorder.config import load_recorder_config
from verl_vla.trainer.main_sac import VLAResourcePoolManager
from verl_vla.utils.recorder import merge_lerobot_datasets
from verl_vla.utils.recorder.lerobot import REQUIRED_LEROBOT_META_FILES
from verl_vla.workers.engine import VLAActorRolloutRefWorker
from verl_vla.workers.env.env_worker import EnvWorker

logger = logging.getLogger(__name__)


def collect_recap_env_data(config):
    """Run RECAP env data collection and return existing/new datasets."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_kwargs = OmegaConf.select(config, "ray_kwargs", default={})
        ray_init_kwargs = ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    return ray.get(run_env_loop.remote(config))


@ray.remote
def run_env_loop(config):
    OmegaConf.resolve(config)

    from verl.single_controller.ray import RayWorkerGroup

    env_device = str(config.env.train.get("device", "cuda")).lower()
    env_pool_name = "env_cpu_pool" if env_device == "cpu" else "env_gpu_pool"
    resource_pool_manager = VLAResourcePoolManager(
        resource_pool_spec={
            "train_rollout_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            env_pool_name: [config.trainer.n_env_workers_per_node] * config.trainer.nnodes,
        },
        mapping={
            Role.ActorRollout: "train_rollout_pool",
            Role.Env: env_pool_name,
        },
        cpu_pool_names={env_pool_name} if env_device == "cpu" else set(),
    )
    resource_pool_manager.create_resource_pool()

    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}
    resource_pool_to_cls[resource_pool_manager.get_resource_pool(Role.ActorRollout)]["actor_rollout"] = (
        RayClassWithInitArgs(
            cls=ray.remote(VLAActorRolloutRefWorker),
            config=config.actor_rollout_ref,
            role="actor_rollout",
        )
    )
    resource_pool_to_cls[resource_pool_manager.get_resource_pool(Role.Env)]["env"] = RayClassWithInitArgs(
        cls=ray.remote(EnvWorker),
        config=config.env,
    )

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            device_name=config.trainer.device,
        )
        all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()
    env_wg = all_wg["env"]

    env_loop = EnvLoop(config=config, rollout_wg=actor_rollout_wg, env_wg=env_wg)
    for rollout_idx in range(int(config.recap.collect.num_rollouts)):
        prompts = build_rollout_prompts(config, env_loop, rollout_idx)
        reset_future = env_wg.reset_envs_to_state_ids(
            DataProto.from_dict(
                non_tensors={
                    "state_ids": prompts.non_tensor_batch["state_ids"],
                    "task_ids": prompts.non_tensor_batch["task_ids"],
                }
            )
        )
        rollout_output = env_loop.generate_sequences(prompts, reset_future)
        logger.info("Finished recap env loop rollout %s: %s", rollout_idx, rollout_output.meta_info.get("metrics", {}))

    return collect_lerobot_datasets(config, env_wg)


def collect_lerobot_datasets(config, env_wg):
    recorder_cfg = load_recorder_config(config.env.train)
    if not recorder_cfg.enable or not recorder_cfg.lerobot.enable:
        return {}

    rank_datasets = [dataset for dataset in env_wg.pop_lerobot_dataset() if dataset is not None]
    root = Path(recorder_cfg.lerobot.root)
    repo_id = recorder_cfg.lerobot.repo_id
    existing_root = root / repo_id
    collected_datasets = {}
    if all((existing_root / path).exists() for path in REQUIRED_LEROBOT_META_FILES):
        collected_datasets["existing_dataset"] = {"root": existing_root, "repo_id": repo_id}

    collected_dataset = _merge_rank_lerobot_datasets(config, rank_datasets, recorder_cfg)
    if collected_dataset:
        collected_datasets["collected_dataset"] = collected_dataset
    return collected_datasets


def _merge_rank_lerobot_datasets(config, rank_datasets, recorder_cfg):
    if not rank_datasets:
        return None

    repo_id = str(
        OmegaConf.select(config, "recap.collected_repo_id", default=f"{recorder_cfg.lerobot.repo_id}_collected")
    )
    return merge_lerobot_datasets(
        roots=[dataset["root"] for dataset in rank_datasets],
        output_root=Path(recorder_cfg.lerobot.root) / repo_id,
        repo_id=repo_id,
        repo_ids=[dataset["repo_id"] for dataset in rank_datasets],
        overwrite=True,
        append=False,
        video_files_size_in_mb=recorder_cfg.lerobot.video_files_size_in_mb,
    )


def build_rollout_prompts(config, env_loop: EnvLoop, global_step: int) -> DataProto:
    if bool(config.env.train.get("single_env_rollout", False)):
        total_envs = env_loop.env_wg.world_size
    else:
        total_envs = (
            env_loop.env_wg.world_size * int(config.env.train.num_envs) * int(config.env.rollout.pipeline_stage_num)
        )
    task_ids = np.full(total_envs, int(config.recap.collect.task_id), dtype=np.int64)
    state_ids = np.full(total_envs, int(config.recap.collect.state_id), dtype=np.int64)
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
