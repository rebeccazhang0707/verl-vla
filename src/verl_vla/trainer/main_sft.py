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

import logging
from pprint import pprint
from typing import Any, cast

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.trainer.sft.sft_ray_trainer import RobRaySFTTrainer
from verl_vla.trainer.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized, get_controller_remote_options

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_sft", version_base=None)
def main(config):
    ensure_ray_initialized(config)
    remote_options = get_controller_remote_options(config)
    ray.get(main_task.options(**remote_options).remote(config))


@ray.remote
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        trainer = RobRaySFTTrainer(
            data_config=config.data,
            trainer_config=config.trainer,
            cluster=cluster,
            tracking_config=cast(dict[str, Any], OmegaConf.to_container(config, resolve=True)),
        )
        trainer.fit()
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()
