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
from verl_vla.utils.rlpd import iter_rlpd_replay_prefill_batches


def infer_num_subtasks(graded_rews: torch.Tensor, max_subtasks: int = 16) -> int:
    """Infer the SEQUENTIAL subtask count N from a graded-progress tensor.

    The graded subtask reward is ``completed / total`` so every value is an exact
    fraction ``k / N``. N is the smallest denominator that makes every observed
    *partial* level (strictly between 0 and 1) integral when multiplied by it.
    Returns 1 when there is no partial progress (e.g. a plain 0/1 success reward),
    so callers degrade gracefully to a single composite metric.
    """
    levels = torch.unique(graded_rews.reshape(-1))
    levels = levels[(levels > 1e-6) & (levels < 1.0 - 1e-6)].tolist()
    if not levels:
        return 1
    for n in range(2, max_subtasks + 1):
        if all(abs(v * n - round(v * n)) < 1e-3 for v in levels):
            return n
    return 1


def compute_avg_positive_trajectory_length(batch: DataProto) -> float:
    dones = batch.batch["info.dones"].bool()  # (B, T)
    positive_mask = batch.batch["info.positive_sample_mask"]  # (B, T)
    positive_traj = positive_mask.any(dim=1)  # (B,)

    if positive_traj.sum() == 0:
        return 0.0

    B, T = dones.shape
    done_idx = torch.argmax(dones.int(), dim=1)  # (B,)
    traj_lens = done_idx + 1

    return traj_lens[positive_traj].float().mean().item()


def compute_per_task_trajectory_metrics(rollout_batch: DataProto, metric_prefix: str) -> dict[str, float]:
    task_ids = rollout_batch.meta_info.get("task_ids")
    if task_ids is None:
        return {}

    complete_any = rollout_batch.batch["feedback.terminations"].any(dim=-1)  # (B, T)
    success_np = complete_any.any(dim=-1).detach().float().cpu().numpy()  # (B,)
    task_ids = np.asarray(task_ids)[: success_np.shape[0]]

    dones = complete_any.detach().cpu()
    done_idx = torch.argmax(dones.int(), dim=1)
    traj_lens = (done_idx + 1).float().numpy()

    metrics = {}
    for task_id in np.unique(task_ids):
        task_mask = task_ids == task_id
        if not task_mask.any():
            continue

        task_key = int(task_id)
        task_success = success_np[task_mask]
        metrics[f"{metric_prefix}/per_task_success_rate/task_{task_key}"] = float(task_success.mean())

        positive_lens = traj_lens[task_mask][task_success.astype(bool)]
        metrics[f"{metric_prefix}/per_task_avg_positive_trajectory_length/task_{task_key}"] = (
            float(positive_lens.mean()) if len(positive_lens) > 0 else 0.0
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
        prefill_batch.meta_info["global_steps"] = self.global_steps
        prefill_batch.meta_info["global_token_num"] = [0]
        prefill_batch.meta_info["add_to_offline_replay_only"] = True
        self.actor_rollout_wg.add_offline_replay_data(prefill_batch)

    def _prepare_actor_input(self, rollout_output: Optional[DataProto]) -> DataProto:
        # dones / reward
        complete_any = rollout_output.batch["feedback.terminations"].any(dim=-1)  # (B, T), latched success
        sparse_rewards = complete_any.float()
        step_penalty = float(self.config.env.train.get("step_penalty", 0.0))
        dense = bool(self.config.env.train.get("dense_success_reward", False))
        subtask = (
            bool(self.config.env.train.get("subtask_reward", False))
            and "feedback.rewards" in rollout_output.batch
        )

        if subtask:
            # Graded subtask reward for the long-horizon SEQUENTIAL task. `feedback.rewards`
            # (B, T, chunk) carries the latched per-substep progress level (0/0.5/1.0 = fraction
            # of subtasks done); use the chunk-end level as the per-chunk-step reward, dense (all
            # steps valid). `feedback.terminations` is the COMPOSITE success (both subtasks; arena
            # chunk_step derives it from reward>=1.0), so done/positive/success stay tied to full
            # success. The -1 timeout applies only to never-composite trajectories.
            graded = rollout_output.batch["feedback.rewards"][..., -1].float()  # (B, T) chunk-end level
            ever_success = complete_any.any(dim=-1)  # (B,) reached composite
            valids = torch.ones_like(complete_any, dtype=torch.float32)
            rewards = graded - step_penalty * valids
            failed = ~ever_success
            rewards[:, -2] = torch.where(failed, torch.full_like(rewards[:, -2], -1.0), rewards[:, -2])
            dones_step = complete_any.clone()
            dones_step[:, -2] = dones_step[:, -2] | failed
            rollout_output.batch["info.dones"] = dones_step.float()
            rollout_output.batch["info.valids"] = valids
            rollout_output.batch["info.rewards"] = rewards
        elif dense:
            # Dense success reward: keep post-success steps VALID so the latched +1 (task stays
            # solved → reward>0 every step) enters the pool for all of them (many +1 anchors), and
            # apply the -1 timeout penalty ONLY to trajectories that never succeeded (removes the
            # contradictory -1 on success-path states). Mitigates the Q~=0 fixed point on the
            # near-solved task where the default one-+1-per-success signal is too sparse.
            ever_success = complete_any.any(dim=-1)  # (B,)
            valids = torch.ones_like(complete_any, dtype=torch.float32)
            rewards = sparse_rewards - step_penalty * valids
            failed = ~ever_success
            rewards[:, -2] = torch.where(failed, torch.full_like(rewards[:, -2], -1.0), rewards[:, -2])
            dones_step = complete_any.clone()
            dones_step[:, -2] = dones_step[:, -2] | failed
            rollout_output.batch["info.dones"] = dones_step.float()
            rollout_output.batch["info.valids"] = valids
            rollout_output.batch["info.rewards"] = rewards
        else:
            dones_step = complete_any.clone()
            dones_step[:, -2] = True
            rollout_output.batch["info.dones"] = dones_step.float()
            rollout_output.batch["info.valids"] = (
                ~rollout_output.batch["feedback.terminations"]
            ).any(dim=-1).float()
            rollout_output.batch["info.rewards"] = (
                sparse_rewards - step_penalty * rollout_output.batch["info.valids"]
            )
            rollout_output.batch["info.rewards"][:, -2] = -1.0

        # mark samples in successful trajectories as positive samples
        rollout_output.batch["info.positive_sample_mask"] = (
            sparse_rewards.any(dim=-1)
            .unsqueeze(-1)
            .repeat_interleave(rollout_output.batch["action.action"].shape[1], dim=-1)
        )

        # task id
        task_ids = rollout_output.meta_info["task_ids"]
        if self.config.env.train.get("single_env_rollout", False):
            task_ids = task_ids[:1]
        rollout_output.batch["info.task_ids"] = torch.as_tensor(
            task_ids,
            dtype=torch.long,
            device=rollout_output.batch["action.action"].device,
        )

        rollout_output.meta_info["global_token_num"] = [0]
        rollout_output.meta_info["data/trajectory_avg_reward"] = (
            sparse_rewards.any(dim=-1).mean(dtype=torch.float32).item()
        )
        rollout_output.meta_info["data/avg_positive_trajectory_length"] = compute_avg_positive_trajectory_length(
            rollout_output
        )

        # Per-subtask success rate (reward-independent ground truth) from the graded progress
        # carried in `feedback.rewards` = fraction of subtasks done (k/N, latched). For an
        # N-subtask SEQUENTIAL task the levels are {0, 1/N, ..., 1}; we report the SR of reaching
        # each subtask k (max_level >= k/N) as `sr_subtask{k}`, plus `sr_composite` = reaching all
        # N. N comes from `env.train.num_subtasks` when set, else is inferred from the distinct
        # graded levels. Gated on subtask_reward (no graded levels without it).
        # NOTE: these reflect the policy at *rollout* time — at warmup they are the INITIAL policy.
        if subtask:
            graded_rews = rollout_output.batch["feedback.rewards"]
            max_level = graded_rews.reshape(graded_rews.shape[0], -1).amax(dim=-1)
            num_subtasks = self.config.env.train.get("num_subtasks", None) or infer_num_subtasks(graded_rews)
            for k in range(1, num_subtasks):
                rollout_output.meta_info[f"data/sr_subtask{k}"] = (
                    (max_level >= k / num_subtasks - 1e-6).float().mean().item()
                )
            rollout_output.meta_info["data/sr_composite"] = (max_level >= 1.0 - 1e-6).float().mean().item()

        rollout_output = add_transition_prefixes(rollout_output)
        rollout_output = flatten_trajectories(rollout_output)

        return rollout_output

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
                        metrics["data/trajectory_avg_reward"] = actor_input.meta_info["data/trajectory_avg_reward"]
                        metrics["data/avg_positive_trajectory_length"] = actor_input.meta_info[
                            "data/avg_positive_trajectory_length"
                        ]
                        for _k in ("data/sr_composite", "data/sr_subtask1"):
                            if _k in actor_input.meta_info:
                                metrics[_k] = actor_input.meta_info[_k]
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

            metric_list.append(
                {
                    "val/avg_reward": actor_input.meta_info["data/trajectory_avg_reward"],
                    "val/avg_positive_trajectory_length": actor_input.meta_info["data/avg_positive_trajectory_length"],
                }
            )

        metrics = {}
        if metric_list:
            metrics["val/avg_reward"] = np.mean([m["val/avg_reward"] for m in metric_list])
            metrics["val/avg_positive_trajectory_length"] = np.mean(
                [m["val/avg_positive_trajectory_length"] for m in metric_list]
            )
        for key, values in per_task_metric_lists.items():
            metrics[key] = float(np.mean(values))
        for key, values in rollout_metric_lists.items():
            metrics[key] = float(np.mean(values))

        return metrics
