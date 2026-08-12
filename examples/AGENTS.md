# Example Workflow Agent Guide

This file defines the conventions for adding or materially revising workflows
under `examples/`. The repository-level `AGENTS.md` remains authoritative; this
guide narrows those rules for the public example surface.

## Purpose and Scope

- Treat every maintained example as a supported, reproducible user workflow,
  not as a record of an internal experiment.
- Pair every example with user-facing documentation under `docs/`. Follow the
  repository's documentation authoring conventions, including the hierarchy
  and indexing requirements below. Guide a new user from environment setup
  through data preparation, execution, output inspection, and evaluation, and
  include reference results from a verified run so users can judge whether
  they reproduced the expected behavior.
- Keep algorithm and framework implementations out of `examples/`. An example
  selects a workflow config, supplies one experiment recipe, and invokes a
  public `vvla-*` entrypoint.
- Do not commit credentials, private URLs, machine-specific absolute paths,
  generated datasets, checkpoints, videos, logs, or Ray state.

## Directory and Naming Layout

The `examples/` directory has exactly three top-level workflow categories:

- `data_collection/` contains workflows that collect demonstrations,
  interventions, or autonomous trajectories from simulators or physical
  robots.
- `fine_tuning/` contains supervised fine-tuning workflows for supported
  policy models.
- `rl/` contains reinforcement-learning and RL-based post-training workflows.

Do not introduce another top-level workflow category. Keep evaluation beside
the training or collection example that produces the policy or data being
evaluated.

Organize concrete examples according to their category:

```text
examples/
  data_collection/
    <environment_or_robot>/
      <collection_method_or_experiment>/
  fine_tuning/
    <model>/
      <descriptive_experiment>/
        <experiment>.yaml
        run_train.sh
        run_eval.sh             # only when evaluation is a separate workflow
  rl/
    <algorithm>/
      <model>/
        <descriptive_experiment>/
          <experiment>.yaml
          run_train.sh
          run_eval.sh             # only when evaluation is a separate workflow
```

- Use lowercase `snake_case` for directories and YAML filenames.
- Use the framework's canonical model identifier, such as `pi05`, in paths.
- Under `data_collection/`, organize first by the environment or physical robot
  and then by the collection method or concrete experiment. Do not add model or
  algorithm levels when the collection workflow does not own them.
- Under `fine_tuning/`, organize by model and experiment. The workflow category
  already identifies supervised fine-tuning, so do not add a redundant
  algorithm level.
- Under `rl/`, organize by algorithm, model, and experiment.
- Make the experiment directory describe the benchmark task and the important
  starting condition or training mode. For example,
  `libero_spatial_task2_online_from_sft_step100` is preferable to `test` or
  `run_1`.
- Name launchers by user intent: `run_train.sh`, `run_eval.sh`, or
  `run_collect.sh`. Do not repeat the model, algorithm, and task in the launcher
  filename when its directory already provides that context.
- Keep related variants in separate experiment directories when they produce
  meaningfully different supported results. Do not encode a parameter sweep in
  one launcher.

Legacy examples need not be reorganized incidentally. A new example or a
material rewrite must follow this layout.

## Configuration Ownership

Keep each setting at the layer that owns it:

1. Reusable workflow, algorithm, model-adapter, environment, and training-stage
   defaults belong under `src/verl_vla/workflows/config/`.
2. The example YAML owns the stable recipe required to reproduce the published
   experiment: benchmark and task selection, algorithm hyperparameters,
   rollout and evaluation cadence, stage schedules, recorder choices, and
   paths derived from the experiment output directory.
3. The launcher owns deployment choices that vary with the user's machine:
   model or dataset inputs the user must select, output root, GPU and CPU worker
   counts, batch sizes constrained by hardware, renderer selection, and runtime
   logging destinations.
4. Documentation owns installation, environment activation, downloads,
   artifact preparation, TensorBoard startup, and result interpretation.

Compose the public workflow config instead of copying it:

```yaml
hydra:
  searchpath:
    - pkg://verl_vla.workflows.config

defaults:
  - /train/<workflow>@_global_
  - override /model/adapter@<owned_config_path>: <adapter>  # when required
  - _self_
```

- Keep `_self_` last so the experiment intentionally overrides composed
  defaults.
- Do not introduce new configuration fields in an example YAML. It may only
  override fields already defined by the composed workflow configuration. If a
  new field is required, define it first in the configuration of the workflow
  or component that owns it.
- Preserve the ownership hierarchy in override paths. Do not add flattened
  aliases for launcher convenience.
- Use `???` for required values supplied by the launcher or user. Do not invent
  silent local defaults for required inputs.
- Define one output root and derive checkpoints, replay data, datasets, videos,
  evaluation results, and other artifacts from it with Hydra interpolation.
- Give `trainer.experiment_name` a descriptive model-task-method name suitable
  for TensorBoard. Avoid generic names such as `test`, `baseline`, or `debug`.
- State task numbering unambiguously in the documentation; simulator task IDs
  are zero-based unless their owning environment contract says otherwise.

## Reproduction Guide Contents

Place the guide under `docs/` with a matching algorithm -> model -> experiment
hierarchy, and link it from the relevant documentation indexes. Reuse an
existing environment setup guide by linking to it instead of copying setup
instructions, while keeping the experiment-specific sequence complete.

The experiment guide should contain, as applicable:

- the experiment goal and the exact initial model and dataset artifacts;
- the benchmark suite, zero-based task ID, task language, and evaluation size;
- the command to launch the checked-in example;
- the division between stable YAML settings and hardware-specific launcher
  overrides;
- the output layout and a separate command for starting TensorBoard;
- the reference training configuration and artifact provenance;
- evaluation success counts and denominators, not only rounded percentages;
- curves or tables backed by retained logs, including the metric definition;
- relevant controls or baselines and known limitations.

Do not present a smoke test as a reproduced result. If full training was not
run, say so explicitly. Documentation links, artifact identifiers, commands,
and reported metrics must describe the checked-in recipe rather than an
untracked local variant.

## Required Validation

Validate a new or changed example in proportion to what it claims:

1. Run `bash -n` on each shell launcher.
2. Compose the real Hydra job through the launcher, normally with
   `run_train.sh --cfg job`, and inspect required paths and key experiment
   fields.
3. Run focused tests for any workflow behavior changed to support the example.
   Do not add tests that merely freeze launcher text or trivial YAML defaults.
4. Run pre-commit checks on the changed files and a strict documentation build
   when the associated guide changes.
5. Run the smallest meaningful end-to-end smoke test when the environment is
   available. A published performance claim requires the documented full
   evaluation, not only successful process startup.

Before finishing a rename or migration, search the repository for the old path,
launcher name, experiment name, and documentation link. Update real consumers
and remove the superseded example rather than leaving two apparent entrypoints.
