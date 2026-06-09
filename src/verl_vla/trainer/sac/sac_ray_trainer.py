# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import math
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ValidationGenerationsLogger
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

from verl_vla.utils.data import add_transition_prefixes, flatten_trajectories
from verl_vla.utils.rlpd import (
    iter_rlpd_replay_prefill_batches,
    pad_dataproto_to_divisor_with_valid_mask,
)


def _reduce_time_tensor(value: torch.Tensor, *, reduction: str) -> torch.Tensor:
    """Reduce chunk/substep dimensions while preserving batch and rollout time."""
    if value.ndim <= 2:
        return value

    while value.ndim > 2:
        if reduction == "any":
            value = value.any(dim=-1)
        elif reduction == "sum":
            value = value.sum(dim=-1)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    return value


def _trajectory_masks(done_steps: torch.Tensor, reward_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build masks for rows that may contain zero, one, or many completed episodes."""
    valid_mask = torch.zeros_like(done_steps, dtype=torch.bool)
    positive_mask = torch.zeros_like(done_steps, dtype=torch.bool)
    completed_returns = []
    positive_lengths = []
    success_count = 0

    batch_size, _num_steps = done_steps.shape
    for batch_idx in range(batch_size):
        start_idx = 0
        done_indices = torch.nonzero(done_steps[batch_idx], as_tuple=False).flatten().tolist()
        for done_idx in done_indices:
            if done_idx < start_idx:
                continue

            segment = slice(start_idx, done_idx + 1)
            segment_return = reward_steps[batch_idx, segment].sum()
            valid_mask[batch_idx, segment] = True
            completed_returns.append(segment_return)

            if segment_return > 0:
                positive_mask[batch_idx, segment] = True
                positive_lengths.append(done_idx - start_idx + 1)
                success_count += 1

            start_idx = done_idx + 1

    trajectory_count = len(completed_returns)
    failed_count = trajectory_count - success_count
    avg_reward = torch.stack(completed_returns).mean(dtype=torch.float32).item() if completed_returns else 0.0
    avg_positive_length = float(np.mean(positive_lengths)) if positive_lengths else 0.0

    return (
        valid_mask,
        positive_mask,
        {
            "data/trajectory_count": trajectory_count,
            "data/success_trajectory_count": success_count,
            "data/failed_trajectory_count": failed_count,
            "data/trajectory_success_rate": success_count / trajectory_count if trajectory_count > 0 else 0.0,
            "data/trajectory_avg_reward": avg_reward,
            "data/avg_positive_trajectory_length": avg_positive_length,
        },
    )


def trajectory_success_stats(next_done: torch.Tensor, next_reward: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row success flags and average positive trajectory lengths."""
    done_steps = _reduce_time_tensor(next_done.bool(), reduction="any")
    reward_steps = _reduce_time_tensor(next_reward.float(), reduction="sum")
    success_flags = []
    positive_lengths = []

    for batch_idx in range(done_steps.shape[0]):
        start_idx = 0
        row_success = False
        row_positive_lengths = []
        done_indices = torch.nonzero(done_steps[batch_idx], as_tuple=False).flatten().tolist()
        for done_idx in done_indices:
            segment = slice(start_idx, done_idx + 1)
            if reward_steps[batch_idx, segment].sum() > 0:
                row_success = True
                row_positive_lengths.append(done_idx - start_idx + 1)
            start_idx = done_idx + 1
        success_flags.append(row_success)
        positive_lengths.append(float(np.mean(row_positive_lengths)) if row_positive_lengths else 0.0)

    return np.asarray(success_flags, dtype=bool), np.asarray(positive_lengths, dtype=np.float32)


def trajectory_row_stats(next_done: torch.Tensor, next_reward: torch.Tensor) -> dict[str, np.ndarray]:
    """Return completed trajectory counts for each rollout row."""
    done_steps = _reduce_time_tensor(next_done.bool(), reduction="any")
    reward_steps = _reduce_time_tensor(next_reward.float(), reduction="sum")
    trajectory_counts = []
    success_counts = []
    positive_length_sums = []
    positive_length_counts = []

    for batch_idx in range(done_steps.shape[0]):
        start_idx = 0
        row_trajectory_count = 0
        row_success_count = 0
        row_positive_length_sum = 0.0
        row_positive_length_count = 0
        done_indices = torch.nonzero(done_steps[batch_idx], as_tuple=False).flatten().tolist()
        for done_idx in done_indices:
            segment = slice(start_idx, done_idx + 1)
            row_trajectory_count += 1
            if reward_steps[batch_idx, segment].sum() > 0:
                row_success_count += 1
                row_positive_length_sum += done_idx - start_idx + 1
                row_positive_length_count += 1
            start_idx = done_idx + 1

        trajectory_counts.append(row_trajectory_count)
        success_counts.append(row_success_count)
        positive_length_sums.append(row_positive_length_sum)
        positive_length_counts.append(row_positive_length_count)

    trajectory_counts = np.asarray(trajectory_counts, dtype=np.int64)
    success_counts = np.asarray(success_counts, dtype=np.int64)
    return {
        "trajectory_count": trajectory_counts,
        "success_trajectory_count": success_counts,
        "failed_trajectory_count": trajectory_counts - success_counts,
        "positive_length_sum": np.asarray(positive_length_sums, dtype=np.float32),
        "positive_length_count": np.asarray(positive_length_counts, dtype=np.int64),
    }


def prepare_sac_actor_input(
    rollout_output: DataProto,
    *,
    config,
    global_steps: int,
) -> DataProto:
    """Prepare env-loop output for SAC updates."""
    action_steps = rollout_output.batch["action.action"].shape[1]
    done_steps = _reduce_time_tensor(rollout_output.batch["next.done"].bool(), reduction="any")
    reward_steps = _reduce_time_tensor(rollout_output.batch["next.reward"].float(), reduction="sum")

    if done_steps.shape[1] != action_steps:
        raise ValueError(f"done steps {done_steps.shape} do not match action steps {action_steps}.")
    if reward_steps.shape[1] != action_steps:
        raise ValueError(f"reward steps {reward_steps.shape} do not match action steps {action_steps}.")

    valid_mask, positive_mask, metrics = _trajectory_masks(done_steps, reward_steps)
    step_penalty = float(config.env.train.get("step_penalty", 0.0))

    rollout_output.batch["info.dones"] = done_steps.float()
    rollout_output.batch["info.valids"] = valid_mask.float()
    rollout_output.batch["info.rewards"] = (reward_steps - step_penalty) * valid_mask.float()
    rollout_output.batch["info.positive_sample_mask"] = positive_mask.float()

    task_ids = rollout_output.meta_info["task_ids"]
    if config.env.train.get("single_env_rollout", False):
        task_ids = task_ids[:1]
    rollout_output.batch["info.task_ids"] = torch.as_tensor(
        task_ids,
        dtype=torch.long,
        device=rollout_output.batch["action.action"].device,
    )

    rollout_output.meta_info["global_token_num"] = [0]
    rollout_output.meta_info["global_steps"] = global_steps
    rollout_output.meta_info.update(metrics)

    rollout_output = add_transition_prefixes(rollout_output)
    return flatten_trajectories(rollout_output)


def compute_per_task_trajectory_metrics(rollout_batch: DataProto, metric_prefix: str) -> dict[str, float]:
    task_ids = rollout_batch.meta_info.get("task_ids")
    if task_ids is None:
        return {}

    row_stats = trajectory_row_stats(
        rollout_batch.batch["next.done"],
        rollout_batch.batch["next.reward"],
    )
    task_ids = np.asarray(task_ids)[: row_stats["trajectory_count"].shape[0]]

    metrics = {}
    for task_id in np.unique(task_ids):
        task_mask = task_ids == task_id
        if not task_mask.any():
            continue

        task_key = int(task_id)
        trajectory_count = int(row_stats["trajectory_count"][task_mask].sum())
        success_count = int(row_stats["success_trajectory_count"][task_mask].sum())
        failed_count = int(row_stats["failed_trajectory_count"][task_mask].sum())
        positive_length_count = int(row_stats["positive_length_count"][task_mask].sum())
        positive_length_sum = float(row_stats["positive_length_sum"][task_mask].sum())

        metrics[f"{metric_prefix}/per_task_trajectory_count/task_{task_key}"] = trajectory_count
        metrics[f"{metric_prefix}/per_task_success_trajectory_count/task_{task_key}"] = success_count
        metrics[f"{metric_prefix}/per_task_failed_trajectory_count/task_{task_key}"] = failed_count
        metrics[f"{metric_prefix}/per_task_success_rate/task_{task_key}"] = (
            success_count / trajectory_count if trajectory_count > 0 else 0.0
        )
        metrics[f"{metric_prefix}/per_task_avg_positive_trajectory_length/task_{task_key}"] = (
            positive_length_sum / positive_length_count if positive_length_count > 0 else 0.0
        )

    return metrics


class RobRaySACTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        processor=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager

        # SAC uses a single actor-rollout worker to manage actor/critic updates.
        self.use_reference_policy = False
        self.use_rm = False
        self.use_critic = False

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.use_prefix_grouper = False
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        actor_optim_total_training_steps = self.config.actor_rollout_ref.actor.optim.total_training_steps
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        if actor_optim_total_training_steps is not None and actor_optim_total_training_steps > 0:
            self.config.actor_rollout_ref.actor.optim.total_training_steps = actor_optim_total_training_steps
        self.checkpoint_manager = None

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups including env workers."""
        super()._start_profiling(do_profile)
        if do_profile and hasattr(self, "env_wg"):
            self.env_wg.start_profile(role="env", profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups including env workers."""
        super()._stop_profiling(do_profile)
        if do_profile and hasattr(self, "env_wg"):
            self.env_wg.stop_profile()

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        if self.config.env.disagg_sim.enable:
            # pin EnvWorker to Simulator GPU nodes
            self.resource_pool_manager.get_resource_pool(Role.Env).accelerator_type = "sim"
            self.resource_pool_manager.get_resource_pool(Role.ActorRollout).accelerator_type = "train_rollout"

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        assert Role.Env in self.role_worker_mapping
        if Role.Env in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Env)
            env_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.Env], config=self.config.env)
            self.resource_pool_to_cls[resource_pool]["env"] = env_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self.env_wg = all_wg["env"]

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async_envloop":
            from verl_vla.env_loop.env_loop import EnvLoop

            self.async_rollout_mode = True
            self.async_rollout_manager = EnvLoop(
                config=self.config, rollout_wg=self.actor_rollout_wg, env_wg=self.env_wg
            )

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys())
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        return gen_batch

    def _reset_envs(self, gen_batch: DataProto) -> asyncio.Future:
        initial_state_ids = gen_batch.non_tensor_batch["state_ids"]
        task_ids = gen_batch.non_tensor_batch["task_ids"]
        if self.config.env.train.get("single_env_rollout", False):
            assert self.config.env.rollout.pipeline_stage_num == 1, (
                "single_env_rollout only supports pipeline_stage_num == 1"
            )
            initial_state_ids = initial_state_ids[:1]
            task_ids = task_ids[:1]
        reset_prompts = DataProto.from_dict(non_tensors={"state_ids": initial_state_ids, "task_ids": task_ids})
        reset_future = self.env_wg.reset_envs_to_state_ids(reset_prompts)
        return reset_future

    def _next_rollout_batch(self, train_iter) -> Optional[DataProto]:
        try:
            batch_dict = next(train_iter)
        except StopIteration:
            return None

        rollout_batch = DataProto.from_single_dict(batch_dict)
        rollout_batch = self._get_gen_batch(rollout_batch)
        rollout_batch = rollout_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        rollout_batch.meta_info["task_ids"] = np.asarray(rollout_batch.non_tensor_batch["task_ids"], dtype=np.int64)
        rollout_batch.meta_info["global_steps"] = self.global_steps

        return rollout_batch

    def _prefill_replay_pool_from_rlpd(self) -> None:
        rlpd_config = OmegaConf.select(self.config, "data.rlpd")
        if not rlpd_config or not rlpd_config.get("enable", False):
            return

        for prefill_batch in iter_rlpd_replay_prefill_batches(self.config, global_steps=self.global_steps):
            self._submit_rlpd_prefill_batch(prefill_batch)

    def _submit_rlpd_prefill_batch(self, prefill_batch: DataProto) -> None:
        prefill_batch = pad_dataproto_to_divisor_with_valid_mask(
            prefill_batch,
            int(self.actor_rollout_wg.world_size),
            valid_key="info.valids",
        )
        prefill_batch.meta_info["global_steps"] = self.global_steps
        prefill_batch.meta_info["global_token_num"] = [0]
        prefill_batch.meta_info["add_to_offline_replay_only"] = True
        self.actor_rollout_wg.add_offline_replay_data(prefill_batch)

    def _prepare_actor_input(self, rollout_output: Optional[DataProto]) -> DataProto:
        return prepare_sac_actor_input(
            rollout_output,
            config=self.config,
            global_steps=self.global_steps,
        )

    @staticmethod
    def _pad_validation_batch(batch: DataProto, target_batch_size: int) -> DataProto:
        valid_batch_size = len(batch)
        if valid_batch_size >= target_batch_size:
            return batch

        pad_size = target_batch_size - valid_batch_size
        pad_part = batch.select_idxs([0]).repeat(pad_size)

        padded_batch = None
        if batch.batch is not None:
            padded_batch = torch.cat([batch.batch, pad_part.batch], dim=0)

        padded_non_tensor_batch = {
            key: np.concatenate([value, pad_part.non_tensor_batch[key]], axis=0)
            for key, value in batch.non_tensor_batch.items()
        }

        padded_meta_info = dict(batch.meta_info)
        task_ids = padded_meta_info.get("task_ids")
        if isinstance(task_ids, np.ndarray):
            padded_meta_info["task_ids"] = np.concatenate(
                [task_ids, np.repeat(task_ids[:1], pad_size, axis=0)],
                axis=0,
            )

        return DataProto(
            batch=padded_batch,
            non_tensor_batch=padded_non_tensor_batch,
            meta_info=padded_meta_info,
        )

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._prefill_replay_pool_from_rlpd()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        rollout_times = int(getattr(self.config.trainer, "rollout_times", 1))
        rollout_interval = int(self.config.trainer.rollout_interval)
        critic_only_steps_after_rollout = int(
            getattr(self.config.actor_rollout_ref.actor, "critic_only_steps_after_rollout", 0)
        )

        self.total_training_steps = (
            self.config.trainer.total_epochs * math.ceil(len(self.train_dataloader) / rollout_times) * rollout_interval
        )
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            train_iter = iter(self.train_dataloader)
            reset_future = None
            next_rollout_batch = self._next_rollout_batch(train_iter)
            if next_rollout_batch is None:
                continue

            print(f"Starting epoch {epoch}, dataloader length: {len(self.train_dataloader)}")
            while next_rollout_batch is not None:
                for training_step in range(rollout_interval):
                    metrics = {}
                    timing_raw = {}

                    # === start profiling ===
                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )

                    with marked_timer("step", timing_raw):
                        # === rollout ===
                        # Determine whether to perform rollout:
                        # enable at start and early warmup, disable during critic warmup phase
                        warm_rollout_steps = int(getattr(self.config.actor_rollout_ref.actor, "warm_rollout_steps", 0))
                        need_rollout = (training_step < rollout_times) or self.global_steps < warm_rollout_steps
                        if (
                            warm_rollout_steps
                            <= self.global_steps
                            < self.config.actor_rollout_ref.actor.critic_warmup_steps
                        ):
                            need_rollout = False
                        if need_rollout and next_rollout_batch is None:
                            break

                        actor_input = None
                        if need_rollout:
                            with marked_timer("rollout", timing_raw):
                                # execute rollout
                                rollout_batch = next_rollout_batch
                                assert rollout_batch is not None
                                if reset_future is None:
                                    reset_future = self._reset_envs(rollout_batch)
                                with marked_timer("generate", timing_raw, color="red"):
                                    rollout_output = self.async_rollout_manager.generate_sequences(
                                        rollout_batch, reset_future
                                    )

                                # prepare for next batch's env reset
                                next_rollout_batch = self._next_rollout_batch(train_iter)
                                if next_rollout_batch is not None:
                                    reset_future = self._reset_envs(next_rollout_batch)

                                # compute rewards and other metrics, and prepare for actor update
                                metrics.update(
                                    compute_per_task_trajectory_metrics(rollout_output, metric_prefix="data")
                                )
                                metrics.update(rollout_output.meta_info["metrics"])
                                actor_input = self._prepare_actor_input(rollout_output)
                                actor_input.meta_info["global_steps"] = self.global_steps

                        # === update policy ===
                        critic_only_update = training_step < rollout_times + critic_only_steps_after_rollout
                        with marked_timer("update_actor", timing_raw, color="red"):
                            if actor_input is not None:
                                actor_input.meta_info["critic_only_update"] = critic_only_update
                                actor_output = self.actor_rollout_wg.update_actor(actor_input)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(
                                    DataProto(
                                        meta_info={
                                            "empty_batch": True,
                                            "global_steps": self.global_steps,
                                            "global_token_num": [0],
                                            "critic_only_update": critic_only_update,
                                        }
                                    )
                                )
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # === validate ===
                    is_last_step = self.global_steps >= self.total_training_steps
                    if (
                        self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        and self.global_steps >= self.config.actor_rollout_ref.actor.critic_warmup_steps
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        reset_future = None

                    # === save checkpoint ===
                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # === stop profiling ===
                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # === training metrics ===
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
                    if actor_input is not None:
                        metrics.update(
                            {key: value for key, value in actor_input.meta_info.items() if key.startswith("data/")}
                        )
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

    def _validate(self) -> dict:
        metric_list = []
        rollout_metric_lists = {}
        per_task_metric_lists = {}
        val_iter = iter(self.val_dataloader)
        test_batch = self._next_rollout_batch(val_iter)
        while test_batch is not None:
            valid_batch_size = len(test_batch)
            target_batch_size = self.config.data.val_batch_size * self.config.actor_rollout_ref.rollout.n
            if valid_batch_size < target_batch_size:
                test_batch = self._pad_validation_batch(test_batch, target_batch_size)

            test_batch.meta_info["validate"] = True
            reset_future = self._reset_envs(test_batch)
            rollout_output = self.async_rollout_manager.generate_sequences(test_batch, reset_future)
            for key, value in rollout_output.meta_info["metrics"].items():
                rollout_metric_lists.setdefault(key, []).append(float(value))
            test_batch = self._next_rollout_batch(val_iter)
            valid_rollout_output = rollout_output[:valid_batch_size]
            valid_rollout_output.meta_info = dict(valid_rollout_output.meta_info)
            if "task_ids" in valid_rollout_output.meta_info:
                valid_rollout_output.meta_info["task_ids"] = valid_rollout_output.meta_info["task_ids"][
                    :valid_batch_size
                ]
            per_task_metrics = compute_per_task_trajectory_metrics(valid_rollout_output, metric_prefix="val")
            for key, value in per_task_metrics.items():
                per_task_metric_lists.setdefault(key, []).append(value)

            actor_input = self._prepare_actor_input(valid_rollout_output)
            actor_metrics = {key: value for key, value in actor_input.meta_info.items() if key.startswith("data/")}

            metric_list.append(
                {
                    "val/avg_reward": actor_metrics["data/trajectory_avg_reward"],
                    "val/avg_positive_trajectory_length": actor_metrics["data/avg_positive_trajectory_length"],
                    "val/trajectory_count": actor_metrics["data/trajectory_count"],
                    "val/success_trajectory_count": actor_metrics["data/success_trajectory_count"],
                    "val/failed_trajectory_count": actor_metrics["data/failed_trajectory_count"],
                    "val/trajectory_success_rate": actor_metrics["data/trajectory_success_rate"],
                }
            )

        metrics = {}
        if metric_list:
            count_keys = {"val/trajectory_count", "val/success_trajectory_count", "val/failed_trajectory_count"}
            for key in metric_list[0]:
                values = [m[key] for m in metric_list]
                metrics[key] = float(np.sum(values)) if key in count_keys else float(np.mean(values))
            trajectory_count = metrics["val/trajectory_count"]
            metrics["val/trajectory_success_rate"] = (
                metrics["val/success_trajectory_count"] / trajectory_count if trajectory_count > 0 else 0.0
            )
        for key, values in per_task_metric_lists.items():
            metrics[key] = float(np.mean(values))
        for key, values in rollout_metric_lists.items():
            metrics[key] = float(np.mean(values))

        return metrics
