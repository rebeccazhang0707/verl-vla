# Framework Overview

verl-vla is a post-training framework for vision-language-action (VLA)
policies built on top of verl. It provides one workflow for the complete
post-training lifecycle, from human-in-the-loop data collection and supervised
fine-tuning to reinforcement learning and policy evaluation, while reusing
verl's distributed training infrastructure.

Training and environment workers can be placed on different nodes according
to their hardware and connectivity requirements. Training may run on a GPU
cluster, a simulator on a GPU with graphics rendering support, and a physical
robot on its local control computer. Through the observation server, operators
can monitor, teleoperate, and intervene in execution from any connected
device.

verl-vla uses a unified execution architecture for SFT-style, on-policy, and
off-policy algorithms, allowing a policy to move seamlessly from the preceding
workflow stages into reinforcement learning post-training.

## Contents

```{toctree}
:maxdepth: 1

architecture
workflows-and-trainers
train-cluster
model-integration
environment-integration
resource-and-configuration
```
