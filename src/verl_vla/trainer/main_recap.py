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

from verl_vla.trainer.recap.collect import collect_recap_env_data
from verl_vla.trainer.recap.returns import ensure_recap_fields


@hydra.main(config_path="config", config_name="rob_recap_trainer", version_base=None)
def main(config):
    collected_datasets = collect_recap_env_data(config)
    ensure_recap_fields(config, collected_datasets)


if __name__ == "__main__":
    main()
