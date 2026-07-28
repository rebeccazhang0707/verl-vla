# TrainCluster

`TrainCluster` is the core execution abstraction in verl-vla. It turns model
training, policy rollout, environment interaction, evaluation, recording, and
checkpointing into one consistent API. Workflows and trainers operate on this
API instead of directly managing Ray actors, worker groups, placement groups,
or simulator processes.

## One API over multiple topologies

A `TrainCluster` supports four user-facing cluster compositions:

- **Actor cluster** (`actor_cluster`) contains actor workers for SFT-style and
  other offline training workflows.
- **Environment–actor–rollout cluster** (`env_actor_rollout_cluster`) combines
  environment, actor, and rollout workers for interactive online
  reinforcement learning. Actor and rollout execution can be colocated or
  placed on separate resources.
- **Environment–rollout cluster** (`env_rollout_cluster`) combines environment
  and rollout workers without actor workers. It is used when a workflow needs
  policy interaction but does not update the policy, such as evaluation,
  autonomous collection, or DAgger data collection.
- **Environment cluster** (`env_cluster`) contains only environment workers
  for teleoperation, demonstration recording, and trajectory replay.

Whichever composition a workflow selects, it interacts with the same
`TrainCluster` interface. Resource planning, worker initialization,
environment-loop construction, weight synchronization, and cleanup all remain
behind this boundary.

## Lifecycle

### `start()`

`start` turns a cluster configuration into a running distributed system. It
allocates the configured resources, creates the required worker groups,
initializes models and environments, and connects the environment loop and
checkpoint infrastructure when needed.

### `shutdown()`

`shutdown` terminates workers, removes their Ray placement groups, releases
cluster state, and cancels any pending asynchronous rollout. Workflows should
always pair `start` with `shutdown` so each stage owns a complete resource
lifecycle.

## Training and interaction

### `rollout()`

`rollout` coordinates policy inference and environment execution. It returns:

- trajectories represented as a verl `DataProto`;
- any LeRobot datasets produced during collection; and
- rollout and trajectory metrics.

The same API supports synchronous rollout and pipelined asynchronous rollout.
When actor and rollout workers are separated, `TrainCluster` coordinates the
weight update and schedules the next rollout without exposing that machinery
to the trainer.

### `train(data)`

`train` sends a `DataProto` batch to the actor workers and executes the update
procedure implemented by the selected actor. It supports synchronous and
asynchronous updates, allowing an algorithm to overlap training with other
work when its workflow requires it.

The trainer decides what data to train on and when to update. `TrainCluster`
decides where and how that update is executed.

### `eval()`

`eval` runs the current policy against the configured evaluation benchmark and
returns aggregated metrics. It manages environment resets, rollout execution,
episode accounting, and per-task trajectory statistics behind one call.

This makes evaluation reusable from a standalone evaluation workflow, a
training loop, or one stage of a larger algorithm such as RECAP.

## Data collection

### `record()`

`record` runs environment-side teleoperation and recording through an
environment-only cluster. It can merge the datasets produced by environment
workers into one LeRobot dataset and return its output path. The same method
can run without collecting a dataset when a workflow only needs interactive
teleoperation.

### `replay(episode)`

`replay` executes a recorded episode in an environment and returns the replay
result. It provides workflows with a common validation path without exposing
the environment worker RPC.

## Model state

### `update_weights()`

`update_weights` synchronizes a separately deployed rollout model with the
actor. It is a no-op when training and rollout share the same model workers,
so callers do not need separate code paths for colocated and disaggregated
deployment.

### `load_checkpoint()` and `save_checkpoint()`

The checkpoint API owns the actor training lifecycle. `load_checkpoint`
restores the configured training state and synchronizes rollout weights when
necessary. `save_checkpoint` writes the current global step and can include
additional trainer-owned state, such as dataloader progress.

This keeps checkpoint policy in `TrainCluster` while allowing each trainer to
decide when a checkpoint should be loaded or saved.

## Diagnostics

`start_profiling`, `stop_profiling`, and `dump_memory_snapshot` expose
diagnostics through the same cluster boundary. The cluster forwards these
operations to the relevant workers, so trainers can profile a distributed run
without depending on worker implementation details.

## The trainer-facing boundary

A trainer can express its control loop using only high-level operations:

```text
start cluster
load checkpoint

repeat:
    rollout
    train
    evaluate when scheduled
    save checkpoint when scheduled

shutdown cluster
```

Not every workflow uses every operation. SFT primarily uses `train` and the
checkpoint API, evaluation uses `eval`, and teleoperation uses `record`.
Keeping these operations on one abstraction lets new workflows compose the
capabilities they need without rebuilding the distributed execution layer.
