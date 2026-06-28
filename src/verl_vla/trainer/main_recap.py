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

import hydra
from omegaconf import OmegaConf

from verl_vla.trainer.recap.collect_data import collect_recap_env_data
from verl_vla.trainer.recap.compute_return import CollectedDatasets, ensure_recap_fields
from verl_vla.trainer.recap.policy_eval import eval_recap_policy
from verl_vla.trainer.recap.train_policy import train_recap_policy
from verl_vla.trainer.recap.train_value_model import train_recap_value_model
from verl_vla.trainer.recap.value_infer import infer_recap_values


@hydra.main(config_path="config", config_name="rob_recap_trainer", version_base=None)
def main(config):
    # Step 1: collect rollout data into a LeRobot dataset.
    if OmegaConf.select(config, "recap.collect_data.enable", default=True):
        collected_datasets: CollectedDatasets | None = collect_recap_env_data(config)
    else:
        collected_datasets = None

    # Step 2: add RECAP return fields to the collected or configured dataset.
    if OmegaConf.select(config, "recap.compute_return.enable", default=True):
        if collected_datasets is None:
            dataset_cfg = config.recap.compute_return.dataset
            collected_datasets = {
                "collected_dataset": {
                    "root": str(dataset_cfg.root),
                    "repo_id": str(dataset_cfg.repo_id),
                }
            }
        collected_datasets = ensure_recap_fields(config, collected_datasets)
    else:
        collected_datasets = None

    # Step 3: train the RECAP value model with the SFT trainer.
    if OmegaConf.select(config, "recap.train_value_model.enable", default=True):
        if collected_datasets is None:
            dataset_cfg = config.recap.train_value_model.dataset
            collected_datasets = {
                "collected_dataset": {
                    "root": str(dataset_cfg.root),
                    "repo_id": str(dataset_cfg.repo_id),
                }
            }
        value_model_path = train_recap_value_model(config, collected_datasets)
        print(f"ReCap value-model training finished: {value_model_path}")
    else:
        value_model_path = None

    # Step 4: infer RECAP values and write them back to the LeRobot dataset.
    if OmegaConf.select(config, "recap.value_infer.enable", default=True):
        if collected_datasets is None:
            dataset_cfg = config.recap.value_infer.dataset
            collected_datasets = {
                "collected_dataset": {
                    "root": str(dataset_cfg.root),
                    "repo_id": str(dataset_cfg.repo_id),
                }
            }
        dataset = collected_datasets.get("collected_dataset") or collected_datasets.get("existing_dataset")
        if dataset is None:
            raise ValueError("RECAP value inference is enabled but no LeRobot dataset was collected or configured.")
        model_path = value_model_path or str(config.recap.value_infer.model_path)
        value_infer_metrics = infer_recap_values(config, dataset, model_path)
        print(f"ReCap value inference finished: {value_infer_metrics}")

    # Step 5: train the final RECAP policy with the SFT trainer.
    if OmegaConf.select(config, "recap.train_policy.enable", default=True):
        if collected_datasets is None:
            dataset_cfg = config.recap.train_policy.dataset
            collected_datasets = {
                "collected_dataset": {
                    "root": str(dataset_cfg.root),
                    "repo_id": str(dataset_cfg.repo_id),
                }
            }
        policy_path = train_recap_policy(config, collected_datasets)
        print(f"ReCap policy training finished: {policy_path}")
    else:
        policy_path = None

    # Step 6: evaluate the RECAP policy on the environment benchmark.
    if OmegaConf.select(config, "recap.policy_eval.enable", default=False):
        if policy_path is None:
            policy_path = OmegaConf.select(config, "recap.policy_eval.model_path")
        if policy_path is None:
            raise ValueError(
                "recap.policy_eval.enable=True requires recap.train_policy.enable=True or recap.policy_eval.model_path."
            )
        metrics = eval_recap_policy(config, policy_path)
        print(f"ReCap policy eval finished: {metrics}")


if __name__ == "__main__":
    main()
