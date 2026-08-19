# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-19

This first public release extends verl into a unified post-training system for
vision-language-action policies. It provides reusable execution paths from
human-in-the-loop data collection through supervised fine-tuning, distributed
rollout and evaluation, and reinforcement learning across simulators, robots,
operator devices, and cloud or local compute resources.

### Added

- **Distributed execution:** `TrainCluster` coordinates actor, rollout, and
  environment workers; manages Ray resources and worker lifecycles; and
  provides rollout, training, evaluation, checkpoint, and weight-sync
  operations for colocated and disaggregated deployments.
- **Composable workflows:** shared entrypoints and Hydra configuration compose
  data collection, SFT, evaluation, SAC, and multi-stage RECAP procedures while
  keeping orchestration separate from trainers, workers, models, and
  environments.
- **Model integrations:** ACT, Gaussian Actor, Pi0.5, and GR00T N1.6 run through
  a common trainable-model contract while retaining their upstream-native
  implementations, checkpoint formats, processors, and Hugging Face exports.
- **Training capabilities:** FSDP-based SFT, LoRA, checkpoint resume, profiling,
  episodic replay, RLPD, TD3+BC, CQL, DSRL latent-noise steering, and the full
  RECAP sequence from trajectory collection to value-guided policy updates.
- **Environments and human input:** LIBERO, Isaac Lab Arena, and Piper integrate
  with a normalized environment contract and a browser interface for remote
  observation, keyboard, gamepad, and XR teleoperation, and policy
  intervention.
- **Data lifecycle:** resumable LeRobot and video recording, autonomous and
  intervention trajectory capture, DAgger-style collection, dataset replay,
  and explicit episode-boundary handling connect collected experience directly
  to later training stages.
- **Reproducible workflows:** maintained launchers, pinned environments,
  reference results, and end-to-end documentation cover ACT SFT and SAC,
  Gaussian Actor SFT and TD3+BC, Pi0.5 SFT, TD3+BC, DSRL, and RECAP, and GR00T
  SFT, Arena SAC, DSRL, and RECAP.
- **Distribution:** the Python package includes its Hydra configurations, Web
  assets, and console commands for teleoperation, recording, replay, DAgger,
  evaluation, SFT, SAC, and RECAP.

[Unreleased]: https://github.com/verl-project/verl-vla/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/verl-project/verl-vla/releases/tag/v0.1.0
