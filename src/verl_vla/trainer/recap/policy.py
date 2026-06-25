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

import ray
from omegaconf import OmegaConf

from verl_vla.trainer.main_sft import main_task as run_sft_main_task
from verl_vla.trainer.recap.returns import RECAP_INDICATOR_FIELD


def _build_policy_sft_config(config, collected_datasets):
    sft_config_node = OmegaConf.select(config, "recap.policy.config")
    if sft_config_node is None:
        raise ValueError("`recap.policy.config` is required when RECAP policy training is enabled.")

    sft_config = OmegaConf.create(OmegaConf.to_container(sft_config_node, resolve=False))
    OmegaConf.set_struct(sft_config, False)

    dataset = collected_datasets["collected_dataset"]
    OmegaConf.update(sft_config, "data.repo_id", str(dataset["repo_id"]))
    OmegaConf.update(sft_config, "data.root", str(dataset["root"]))
    OmegaConf.update(sft_config, "actor_rollout_ref.actor.acp.enable", True)
    OmegaConf.update(sft_config, "actor_rollout_ref.actor.data_keys.indicator", RECAP_INDICATOR_FIELD)

    OmegaConf.resolve(sft_config)
    return sft_config


def train_recap_policy(config, collected_datasets) -> None:
    sft_config = _build_policy_sft_config(config, collected_datasets)
    ray.get(run_sft_main_task.remote(sft_config))
