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

from omegaconf import OmegaConf

from verl_vla.workflows.train.recap import workflow


def test_recap_uses_fixed_training_configs_and_evaluates_final_policy(monkeypatch):
    config = OmegaConf.create(
        {
            "recap": {
                "num_iterations": 3,
                "resume_iteration": 1,
                "resume_step": 1,
                "policy_eval": {
                    "enable": True,
                    "model_path": "initial-policy",
                    "disable_acp_on_first_iteration": True,
                },
                "collect_data": {"enable": True},
                "compute_return": {"enable": True},
                "train_value_model": {
                    "enable": True,
                    "trainer": {"experiment_name": "value", "total_epochs": 10},
                    "cluster": {"actor_rollout_ref": {"model": {"path": "initial-value"}}},
                },
                "value_infer": {"enable": True},
                "train_policy": {
                    "enable": True,
                    "trainer": {"experiment_name": "policy", "total_epochs": 3},
                    "cluster": {"actor_rollout_ref": {"model": {"path": "initial-policy"}}},
                },
            }
        }
    )
    dataset = {"collected_dataset": {"root": "dataset", "repo_id": "local/dataset"}}
    eval_paths = []
    collect_paths = []
    value_inputs = []
    policy_inputs = []

    monkeypatch.setattr(workflow, "ensure_ray_initialized", lambda _: None)
    monkeypatch.setattr(
        workflow,
        "eval_recap_policy",
        lambda _, policy_path, **__: eval_paths.append(policy_path) or {},
    )
    monkeypatch.setattr(
        workflow,
        "collect_recap_env_data",
        lambda _, policy_path: collect_paths.append(policy_path) or dataset,
    )
    monkeypatch.setattr(workflow, "ensure_recap_fields", lambda _, datasets: datasets)
    monkeypatch.setattr(workflow, "merge_recap_collected_dataset_into_sft_dataset", lambda _, datasets: datasets)

    def train_value_model(current_config, _):
        value_inputs.append(
            (
                current_config.recap.train_value_model.cluster.actor_rollout_ref.model.path,
                current_config.recap.train_value_model.trainer.total_epochs,
            )
        )
        return f"value-{len(value_inputs)}"

    monkeypatch.setattr(workflow, "train_recap_value_model", train_value_model)
    monkeypatch.setattr(workflow, "infer_recap_values", lambda *_: {})

    def train_policy(current_config, _):
        policy_inputs.append(current_config.recap.train_policy.cluster.actor_rollout_ref.model.path)
        return f"policy-{len(policy_inputs)}"

    monkeypatch.setattr(workflow, "train_recap_policy", train_policy)

    workflow.run_recap(config)

    assert eval_paths == ["initial-policy", "policy-1", "policy-2", "policy-3"]
    assert collect_paths == [None, "policy-1", "policy-2"]
    assert value_inputs == [("initial-value", 10), ("initial-value", 10), ("initial-value", 10)]
    assert policy_inputs == ["initial-policy", "initial-policy", "initial-policy"]
