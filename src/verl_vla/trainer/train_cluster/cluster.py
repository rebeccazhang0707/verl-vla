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

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import ray
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.utils import Role

from verl_vla.env_loop.env_loop import EnvLoop
from verl_vla.trainer.train_cluster.config import EnvLoopTrainClusterConfig, ResourceConfig, SFTTrainClusterConfig
from verl_vla.trainer.train_cluster.resource_pool import VLAResourcePoolManager
from verl_vla.utils.recorder import merge_lerobot_datasets
from verl_vla.utils.recorder.lerobot import REQUIRED_LEROBOT_META_FILES
from verl_vla.workers.engine import VLAActorRolloutRefWorker, VLAActorWorker, VLARolloutWorker
from verl_vla.workers.env.env_worker import EnvWorker

__all__ = [
    "TrainCluster",
    "VLAResourcePoolManager",
]

ROLE_TO_WORKER_NAME = {
    Role.Actor: "actor",
    Role.Rollout: "rollout",
    Role.ActorRollout: "actor_rollout",
    Role.Env: "env",
}


class TrainCluster:
    def __init__(self, config: SFTTrainClusterConfig | EnvLoopTrainClusterConfig):
        if not isinstance(config, SFTTrainClusterConfig | EnvLoopTrainClusterConfig):
            raise TypeError(
                "TrainCluster config must be SFTTrainClusterConfig or EnvLoopTrainClusterConfig, "
                f"got {type(config).__name__}."
            )
        self.config = config
        self.cluster_type = "sft" if isinstance(config, SFTTrainClusterConfig) else "env_loop"
        self.resource_pool_spec: dict[str, list[int]] = {}
        self.role_to_pool: dict[Role, str] = {}
        self.cpu_pool_names: set[str] = set()
        self.resource_labels: dict[str, str] = {}
        self.resource_pool_manager: VLAResourcePoolManager | None = None
        self.worker_groups: dict[str, Any] = {}
        self.env_loop: EnvLoop | None = None
        self._lerobot_collected_once = False

    def start(self) -> None:
        self._build_resource_pool_plan()
        self.resource_pool_manager = VLAResourcePoolManager(
            resource_pool_spec=self.resource_pool_spec,
            mapping=cast(dict[int, str], self.role_to_pool),
            cpu_pool_names=self.cpu_pool_names,
            resource_labels=self.resource_labels,
        )
        self._init_workers()
        if self.cluster_type == "env_loop":
            env_wg = self.worker_groups[ROLE_TO_WORKER_NAME[Role.Env]]
            rollout_wg = (
                self.worker_groups.get(ROLE_TO_WORKER_NAME[Role.ActorRollout])
                or self.worker_groups[ROLE_TO_WORKER_NAME[Role.Rollout]]
            )

            self.env_loop = EnvLoop(
                config=self.config.env.env_loop,
                switch_actor_rollout_mode=(
                    self.config.actor_rollout_ref.actor is not None
                    and not self.config.resource.separate_rollout_model.enabled
                ),
                rollout_wg=rollout_wg,
                env_wg=env_wg,
            )

    def _init_workers(self) -> None:
        if self.resource_pool_manager is None:
            raise RuntimeError("Resource pool manager is not initialized. Call start() first.")
        assert self.resource_pool_manager is not None
        self.resource_pool_manager.create_resource_pool()
        self.worker_groups = {}

        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        role_worker_mapping = self._role_worker_mapping()
        for role, pool_name in self.role_to_pool.items():
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            worker_name = ROLE_TO_WORKER_NAME[role]
            worker_config = self._worker_config(role)
            ray_cls_with_init = RayClassWithInitArgs(
                cls=ray.remote(role_worker_mapping[role]),
                config=worker_config,
                role=worker_name,
            )
            resource_pool_to_cls[resource_pool][worker_name] = ray_cls_with_init

        for resource_pool, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.config.resource.model.device,
            )
            self.worker_groups.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        for worker_name in [
            ROLE_TO_WORKER_NAME[Role.Actor],
            ROLE_TO_WORKER_NAME[Role.Rollout],
            ROLE_TO_WORKER_NAME[Role.ActorRollout],
        ]:
            if worker_name in self.worker_groups:
                self.worker_groups[worker_name].init_model()
        if ROLE_TO_WORKER_NAME[Role.Env] in self.worker_groups:
            self.worker_groups[ROLE_TO_WORKER_NAME[Role.Env]].init_worker()

    def _build_resource_pool_plan(self) -> None:
        self.resource_pool_spec = {}
        self.role_to_pool = {}
        self.cpu_pool_names = set()
        self.resource_labels = {}

        if self.cluster_type == "sft":
            self._add_resource_pool(pool_name="train_rollout_pool", resource=self.config.resource.model)
            self.role_to_pool = {Role.Actor: "train_rollout_pool"}

        elif self.cluster_type == "env_loop":
            resource = self.config.resource
            env_pool_name = "env_cpu_pool" if resource.env.device == "cpu" else "env_gpu_pool"
            self._add_resource_pool(pool_name=env_pool_name, resource=resource.env)
            self.role_to_pool[Role.Env] = env_pool_name

            if resource.separate_rollout_model.enabled:
                if self.config.actor_rollout_ref.actor is None:
                    raise ValueError(
                        "Env-loop train cluster with separate_rollout_model enabled requires actor config."
                    )
                self._add_resource_pool(pool_name="train_pool", resource=resource.model)
                self._add_resource_pool(pool_name="rollout_pool", resource=resource.separate_rollout_model)
                self.role_to_pool[Role.Actor] = "train_pool"
                self.role_to_pool[Role.Rollout] = "rollout_pool"
            else:
                self._add_resource_pool(pool_name="train_rollout_pool", resource=resource.model)
                role = Role.ActorRollout if self.config.actor_rollout_ref.actor is not None else Role.Rollout
                self.role_to_pool[role] = "train_rollout_pool"
        else:
            raise ValueError(f"Unsupported train cluster type: {self.cluster_type}")

    def _add_resource_pool(self, pool_name: str, resource: ResourceConfig) -> None:
        processes_per_node = resource.workers_per_node if resource.device == "cpu" else resource.gpus_per_node
        self.resource_pool_spec[pool_name] = [processes_per_node] * resource.nnodes
        if resource.device == "cpu":
            self.cpu_pool_names.add(pool_name)
        if resource.resource_label is not None:
            self.resource_labels[pool_name] = resource.resource_label

    def _role_worker_mapping(self):
        if self.cluster_type == "sft":
            return {Role.Actor: VLAActorRolloutRefWorker}

        elif self.cluster_type == "env_loop":
            separate_rollout_model = self.config.resource.separate_rollout_model.enabled
            has_actor = self.config.actor_rollout_ref.actor is not None

            if separate_rollout_model:
                if not has_actor:
                    raise ValueError(
                        "Env-loop train cluster with separate_rollout_model enabled requires actor config."
                    )
                return {
                    Role.Actor: VLAActorWorker,
                    Role.Rollout: VLARolloutWorker,
                    Role.Env: EnvWorker,
                }

            if has_actor:
                return {
                    Role.ActorRollout: VLAActorRolloutRefWorker,
                    Role.Env: EnvWorker,
                }

            return {
                Role.Rollout: VLARolloutWorker,
                Role.Env: EnvWorker,
            }
        else:
            raise ValueError(f"Unsupported train cluster type: {self.cluster_type}")

    def _worker_config(self, role: Role):
        if role == Role.Env:
            if not isinstance(self.config, EnvLoopTrainClusterConfig) or self.config.env is None:
                raise ValueError("Env worker requires EnvLoopTrainClusterConfig.env.")
            return self.config.env
        elif role in {Role.Actor, Role.Rollout, Role.ActorRollout}:
            return self.config.actor_rollout_ref
        else:
            raise ValueError(f"Unsupported worker role: {role}")

    def rollout(self, prompts: DataProto) -> tuple[DataProto, dict[str, dict[str, Any]]]:
        if self.cluster_type != "env_loop":
            raise RuntimeError("rollout is only wired for env-loop train clusters.")

        reset_future = self.worker_groups["env"].reset_envs_to_state_ids(
            DataProto.from_dict(
                non_tensors={
                    "state_ids": prompts.non_tensor_batch["state_ids"],
                    "task_ids": prompts.non_tensor_batch["task_ids"],
                }
            )
        )
        assert self.env_loop is not None
        output = self.env_loop.generate_sequences(prompts, reset_future)
        return output, self._collect_lerobot_datasets()

    def _collect_lerobot_datasets(self) -> dict[str, dict[str, Any]]:
        if not isinstance(self.config, EnvLoopTrainClusterConfig):
            return {}

        recorder_cfg = self.config.env.env_worker.recorder
        if not recorder_cfg.enable or not recorder_cfg.lerobot.enable:
            return {}

        env_wg = self.worker_groups[ROLE_TO_WORKER_NAME[Role.Env]]
        rank_datasets = [dataset for dataset in env_wg.pop_lerobot_dataset() if dataset is not None]
        root = Path(recorder_cfg.lerobot.root)
        repo_id = recorder_cfg.lerobot.repo_id
        existing_root = root / repo_id
        collected_datasets: dict[str, dict[str, Any]] = {}
        if all((existing_root / path).exists() for path in REQUIRED_LEROBOT_META_FILES):
            collected_datasets["existing_dataset"] = {"root": existing_root, "repo_id": repo_id}

        collected_dataset = self._merge_rank_lerobot_datasets(rank_datasets)
        if collected_dataset:
            collected_datasets["collected_dataset"] = collected_dataset
        return collected_datasets

    def _merge_rank_lerobot_datasets(self, rank_datasets: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rank_datasets:
            return None
        if not isinstance(self.config, EnvLoopTrainClusterConfig):
            return None

        recorder_cfg = self.config.env.env_worker.recorder
        repo_id = f"{recorder_cfg.lerobot.repo_id}_collected"
        collected_dataset = merge_lerobot_datasets(
            roots=[dataset["root"] for dataset in rank_datasets],
            output_root=Path(recorder_cfg.lerobot.root) / repo_id,
            repo_id=repo_id,
            repo_ids=[dataset["repo_id"] for dataset in rank_datasets],
            overwrite=not self._lerobot_collected_once,
            append=self._lerobot_collected_once,
            video_files_size_in_mb=recorder_cfg.lerobot.video_files_size_in_mb,
        )
        self._lerobot_collected_once = True
        return collected_dataset

    def train(self, *args: Any, **kwargs: Any) -> Any: ...

    def eval(self, *args: Any, **kwargs: Any) -> Any: ...

    def update_weights(self, *args: Any, **kwargs: Any) -> Any: ...

    def load_checkpoint(self, *args: Any, **kwargs: Any) -> Any: ...

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> Any: ...
