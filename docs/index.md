# verl-vla

**A unified post-training framework for vision-language-action policies, built
on top of [verl](https://github.com/verl-project/verl).**

verl-vla connects human-in-the-loop data collection, supervised fine-tuning,
reinforcement learning, and policy evaluation through one distributed system.
It provides a continuous path from collecting robot experience to training,
evaluating, and improving VLA policies.

The [Quick Start](getting-started/index.md) offers a practical introduction to
verl-vla's complete embodied-model post-training workflow, from teleoperation
and data recording to training and evaluation. We recommend reading or trying
the guide to become familiar with the overall process. You can then adapt the
same workflow to your choice of simulator or physical robot, embodied model,
and reinforcement learning algorithm.

## Documentation

- **[Framework overview](framework-overview/index.md):** understand workflows,
  trainers, `TrainCluster`, model integrations, and environment integrations.
- **[Data collection](data-collection/index.md):** collect demonstrations,
  interventions, and autonomous trajectories.
- **[Fine-tuning](fine-tuning/index.md):** train supported VLA policies with
  maintained supervised fine-tuning recipes.
- **[Reinforcement learning](reinforcement-learning/index.md):** run supported
  RL and RL-based post-training workflows.
- **[Troubleshooting](troubleshooting/index.md):** diagnose simulator and
  distributed execution problems.

```{toctree}
:hidden:
:maxdepth: 2
:titlesonly:

getting-started/index
framework-overview/index
data-collection/index
fine-tuning/index
reinforcement-learning/index
troubleshooting/index
```
