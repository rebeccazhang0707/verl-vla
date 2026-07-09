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
import shutil
from pathlib import Path
from pprint import pprint

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from verl_vla.trainer.train_cluster import TrainCluster
from verl_vla.utils.ray_utils import ensure_ray_initialized
from verl_vla.utils.rollout_collection import collect_lerobot_rollout_dataset

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main_dagger", version_base=None)
def main(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)

    ensure_ray_initialized(config)
    cluster = TrainCluster(instantiate(config.cluster, _recursive_=False))
    cluster.start()
    try:
        target_episodes = int(config.max_episodes)
        collected_datasets = collect_lerobot_rollout_dataset(
            cluster,
            target_episodes=target_episodes,
            log_prefix="DAgger rollout",
            max_episodes_name="max_episodes",
            log=logger,
        )
        dataset_root = Path(config.cluster.env.env_worker.recorder.lerobot.root) / str(
            config.cluster.env.env_worker.recorder.lerobot.repo_id
        )
        collected_root = Path(collected_datasets["collected_dataset"]["root"])
        assert collected_root.exists()
        print(f"DAgger collected dataset before move: {collected_root}")
        if dataset_root.exists():
            print(f"DAgger dataset destination already exists, skip moving: {dataset_root}")
            print(f"DAgger collected dataset remains at: {collected_root}")
        else:
            shutil.move(str(collected_root), str(dataset_root))
            print(f"DAgger dataset saved to: {dataset_root}")
    finally:
        cluster.shutdown()


if __name__ == "__main__":
    main()
