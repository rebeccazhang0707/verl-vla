# Resource and Configuration

verl-vla separates **what a workflow runs** from **where each distributed role
runs**. Hydra composes the model, environment, trainer, and workflow settings,
while `TrainCluster` converts the selected resource configuration into Ray
resource pools and worker groups.

This design allows the same workflow to run on one workstation or across
specialized nodes without changing the trainer or environment implementation.
For example, actor training, policy rollout, and image-rendering environments
can share a machine, occupy different GPU sets, or be pinned to different
classes of nodes.

## Configuration architecture

The configuration tree mirrors runtime ownership:

```text
workflow
├── cluster
│   ├── resource
│   ├── actor_rollout_ref
│   │   ├── model
│   │   ├── actor
│   │   └── rollout
│   ├── env
│   │   ├── env_loop
│   │   └── env_worker
│   │       ├── simulator
│   │       ├── teleop
│   │       └── recorder
│   └── checkpoint
├── trainer
├── data
└── ray_kwargs
```

Each layer owns only the values it consumes:

- A **workflow config** selects the procedure and composes its cluster,
  trainer, data, and Ray settings.
- A **cluster config** selects the worker composition and assembles the
  resources and component configs required by that composition.
- A **resource config** describes placement and process counts, but contains
  no model or algorithm settings.
- Component configs remain with their owning model, actor, rollout,
  environment, recorder, or trainer.
- `ray_kwargs` configures Ray initialization and the runtime environment
  inherited by distributed workers.

Hydra YAML under `workflows/config/` is the public composition layer. Typed
configuration classes live beside the components that consume them and
validate the fully composed values at the construction boundary.

## Cluster composition

The four cluster configurations select the distributed roles required by a
workflow:

| Cluster config | Roles | Typical use |
| --- | --- | --- |
| `actor_cluster` | actor | Offline SFT-style training |
| `env_actor_rollout_cluster` | environment and actor/rollout | Interactive online reinforcement learning |
| `env_rollout_cluster` | environment and rollout | Evaluation, autonomous collection, and DAgger |
| `env_cluster` | environment | Teleoperation, demonstration recording, and replay |

A workflow chooses one cluster through its Hydra defaults:

```yaml
defaults:
  - /cluster@cluster: env_actor_rollout_cluster
  - /trainer@trainer: sac_trainer
  - /ray@ray_kwargs: default
  - _self_
```

The package expression `/cluster@cluster` selects a config from the
`cluster` group and places the result at the workflow's `cluster` key.
Keeping the package explicit makes the final override path match the runtime
owner, such as `cluster.resource.env.device`.

`_self_` remains last when values in the current file intentionally override
the composed defaults.

## Resource model

`TrainCluster` uses one `ResourceConfig` for each independently placed role:

```yaml
env:
  _target_: verl_vla.train_cluster.config.ResourceConfig
  device: cuda
  resource_label: null
  nnodes: 1
  gpus_per_node: 1
  workers_per_node: 1
```

The fields have the following meanings:

- `device` selects a `cuda` or `cpu` worker pool.
- `nnodes` is the number of nodes participating in the pool.
- `gpus_per_node` is the number of GPU workers created on each node when
  `device: cuda`. Each worker reserves one GPU.
- `workers_per_node` is the number of workers created on each node when
  `device: cpu`.
- `resource_label` optionally constrains the pool to Ray nodes advertising a
  matching custom resource.

For a CUDA pool, the total worker count is
`nnodes * gpus_per_node`. For a CPU pool, it is
`nnodes * workers_per_node`.

`controller_label` belongs to the enclosing resource configuration. When set,
the remote workflow controller is scheduled on a node with that Ray resource
label. This is independent of worker placement.

## How roles map to resources

### Actor-only training

`actor_cluster` creates one model resource pool and places actor workers in
it:

```text
model resource → actor workers
```

The model pool defines the distributed training world size. For example:

```yaml
cluster:
  resource:
    model:
      device: cuda
      nnodes: 2
      gpus_per_node: 8
```

creates 16 actor workers across two nodes.

### Environment-only execution

`env_cluster` creates only the environment resource pool:

```text
environment resource → environment workers
```

This is the smallest composition for teleoperation and recording because no
model worker is constructed.

An environment can use a GPU for rendering:

```yaml
cluster:
  resource:
    env:
      device: cuda
      nnodes: 1
      gpus_per_node: 1
```

or use CPU workers when the backend supports CPU execution:

```yaml
cluster:
  resource:
    env:
      device: cpu
      nnodes: 1
      workers_per_node: 2
```

The selected resource device is also propagated to the environment worker
unless `env.env_worker.device` is set explicitly.

### Environment and model execution

Environment-loop clusters always give environment workers their own resource
pool. Model placement has two modes.

By default, actor and rollout capabilities share one worker group and one
model pool:

```text
environment resource → environment workers
model resource       → actor/rollout workers
```

This composition avoids maintaining two model copies and allows the worker to
switch between training and rollout execution.

Set `separate_rollout_model.enabled: true` when rollout should use independent
workers and resources:

```yaml
cluster:
  resource:
    model:
      device: cuda
      nnodes: 1
      gpus_per_node: 8

    separate_rollout_model:
      enabled: true
      device: cuda
      nnodes: 1
      gpus_per_node: 2
```

The resulting topology is:

```text
environment resource       → environment workers
model resource             → actor workers
separate rollout resource  → rollout workers
```

`TrainCluster` then synchronizes actor weights to the rollout workers through
the shared model checkpoint and weight-update contracts. A separate rollout
pool is valid only when the cluster also contains an actor.

## Place roles on specialized nodes

Ray resource labels allow each role to select a node class. Start Ray nodes
with custom resources that describe their capability, then reference the
corresponding label from the workflow:

```yaml
cluster:
  resource:
    controller_label: controller

    env:
      resource_label: rendering
      device: cuda
      nnodes: 1
      gpus_per_node: 2

    model:
      resource_label: training
      device: cuda
      nnodes: 2
      gpus_per_node: 8
```

This can place image-rendering simulators on rendering nodes, distributed
optimization on training nodes, and the workflow controller on a lightweight
controller node. Labels constrain placement; `nnodes` and the per-node worker
count still determine how many resources the pool requests.

Without labels, Ray chooses any nodes that satisfy the CPU or GPU bundles.
verl-vla checks the total available accelerator count before starting GPU
worker pools and fails early if the cluster cannot satisfy the requested
topology.

## Configure Ray workers

The shared Ray config is composed under `ray_kwargs`:

```yaml
ray_kwargs:
  ray_init:
    num_cpus: null
    logging_level: INFO
    runtime_env:
      env_vars:
        MUJOCO_GL: egl
        VERL_LOGGING_LEVEL: INFO
        HF_TOKEN: ${oc.env:HF_TOKEN,""}
```

`ray_init` values are forwarded to `ray.init`. Values under
`runtime_env.env_vars` are propagated to distributed workers, making this the
appropriate place for renderer selection, logging settings, model-cache
locations, and credentials read from the launch environment.

Use OmegaConf environment interpolation for secrets instead of writing them
into a tracked configuration:

```yaml
HF_TOKEN: ${oc.env:HF_TOKEN,""}
```

The empty fallback keeps the configuration resolvable when access to a gated
artifact is not required.

## Override a workflow

Hydra overrides follow the ownership tree. Common examples include:

```bash
python -m verl_vla.entrypoints.eval \
  cluster.resource.env.gpus_per_node=2 \
  cluster.resource.model.gpus_per_node=4 \
  cluster.env.env_worker.num_envs=8 \
  cluster.env.env_worker.simulator.simulator_type=libero
```

These values control different forms of parallelism:

- `cluster.resource.env.*` controls the number and placement of environment
  worker processes.
- `cluster.env.env_worker.num_envs` controls the number of native environments
  hosted inside each environment worker.
- `cluster.resource.model.*` controls the model worker world size.
- `cluster.env.env_loop.pipeline_stage_num` controls the number of environment
  pipeline stages hosted by each environment worker.

Keep these dimensions explicit. Increasing vectorized environments does not
reserve additional Ray workers, while increasing a resource pool changes the
distributed worker topology.

To select a different config-group option, override the Hydra group rather
than copying its fields. For example, a workflow should select an existing
simulator or cluster variant through its defaults and use field overrides only
for values specific to that run.

## Add a configuration component

When integrating a new model, environment, algorithm, or workflow stage:

1. Define the typed configuration next to the component that consumes it.
2. Add one reusable YAML option in the corresponding
   `workflows/config/<group>/` directory.
3. Give the YAML a `_target_` for the typed configuration.
4. Compose child configs through `defaults` using explicit package paths.
5. Keep `_self_` last when local values should override composed defaults.
6. Instantiate the config once at the owning runtime boundary.

For example:

```yaml
defaults:
  - /model/adapter@adapter: my_policy
  - /model/lora@lora: default
  - _self_

_target_: verl.workers.config.model.HFModelConfig

path: ${oc.env:MODEL_PATH}
```

Avoid copying a full model, environment, or resource block into each workflow.
Reusable config groups preserve one canonical definition and keep command-line
override paths stable.

Do not flatten child settings into a parent merely to shorten an override.
The path communicates ownership: environment rendering belongs under
`cluster.env.env_worker.simulator`, placement belongs under
`cluster.resource`, and algorithm progression belongs under `trainer`.

## Choose a topology

Start with the smallest composition that provides the required roles:

- Use `actor_cluster` for offline supervised training.
- Use `env_cluster` for teleoperation, recording, and replay.
- Use `env_rollout_cluster` for inference-only environment interaction.
- Use `env_actor_rollout_cluster` for interactive online reinforcement
  learning that alternates environment interaction and policy updates.

Keep actor and rollout colocated unless inference requires different hardware,
independent scaling, or isolation from training memory pressure. Separate
environment resources when rendering or robot access imposes its own hardware
and placement requirements.

Because workflows consume the same typed `TrainCluster` interface, changing
this resource topology does not change the algorithm. Configuration selects
the deployment; trainers continue to express the same rollout, training,
evaluation, and recording operations.
