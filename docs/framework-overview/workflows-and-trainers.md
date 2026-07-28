# Workflows and Trainers

verl-vla separates **what a user wants to run** from **how each training
algorithm updates a model**. Workflows own the first responsibility, while
trainers own the second.

## One workflow for each entrypoint

Each user-facing entrypoint corresponds to one supported workflow. The
entrypoint selects a Hydra configuration and hands it to the workflow; it does
not construct models, workers, datasets, or trainers itself.

A workflow defines the complete procedure behind that command. Depending on
the task, it can:

- prepare and connect multiple stages;
- create the `TrainCluster` required by each stage;
- select and configure a trainer;
- pass datasets and checkpoints between stages; and
- manage initialization, resumption, and cleanup.

This gives every entrypoint a clear and reproducible execution path while
keeping orchestration out of the trainer and distributed runtime.

## Composable execution

Workflows can compose `TrainCluster` operations and trainers in the order
required by an algorithm. A simple workflow such as SFT creates one
`TrainCluster`, attaches an SFT trainer, runs `fit`, and shuts the cluster down.
A multi-stage workflow can create different clusters for collection,
evaluation, and training, reuse an existing trainer in several stages, and
feed the output of one stage into the next.

The trainer remains focused on algorithm progression. It decides how to
iterate over data, when and how to update parameters, when to evaluate or save
checkpoints, and when training should stop. It uses `TrainCluster` operations
without owning worker placement, simulator lifecycle, or resource allocation.
This boundary makes both sides reusable: the same trainer can participate in
different workflows, and the same cluster operations can support different
algorithms.

## RECAP as a six-stage workflow

RECAP demonstrates how a workflow can assemble a complete post-training
algorithm from reusable components. One RECAP iteration contains six stages:

| Stage | Workflow responsibility | Reused component |
| --- | --- | --- |
| 1. Policy evaluation | Evaluate the configured policy, or the policy produced by the previous iteration. | Evaluation workflow and rollout cluster |
| 2. Data collection | Run the policy in the environment and record a LeRobot dataset. | Environment loop, `TrainCluster`, and recorder |
| 3. Return computation | Compute RECAP returns and merge the collected data into the training dataset. | Dataset utilities |
| 4. Value-model training | Train the RECAP value model on the prepared dataset. | SFT workflow, SFT trainer, and actor cluster |
| 5. Value inference | Infer values, compute advantages and indicators, and write them back to the dataset. | Value model and data pipeline |
| 6. Policy training | Train the advantage-conditioned policy and produce the policy for the next iteration. | SFT workflow, SFT trainer, and actor cluster |

The outer RECAP workflow controls the order of these stages and passes their
artifacts forward. Individual stages remain independently configurable, and a
run can resume from a selected iteration and stage. When multiple iterations
are requested, the policy produced by stage 6 becomes the policy used for
collection and evaluation in the next iteration.

The important architectural point is that RECAP does not introduce a separate
training stack. It composes existing evaluation, collection, dataset, SFT, and
`TrainCluster` capabilities into a new user-facing procedure. Other
multi-stage algorithms can follow the same pattern: define the workflow,
select the appropriate trainers and cluster operations for each stage, and
keep stage-specific data transformations at their owning boundaries.
