# Model Integration

verl-vla integrates an upstream policy without taking ownership of its
implementation. The native policy remains the source of truth for its model
architecture, preprocessing, and checkpoint format. A thin verl-vla
integration adapts that policy to the common training and rollout contracts
used by workers and `TrainCluster`.

## Integration principles

### Preserve the native policy

The upstream policy is kept as a regular PyTorch module and loaded through its
native API. verl-vla does not copy the model implementation into a new
framework-specific model or require it to register with Transformers
AutoClass.

The integration may add framework-owned components, such as an input adapter
or critic head, but these remain outside the upstream policy. This keeps the
ownership boundary clear and makes upstream updates easier to adopt.

### Adapt at one boundary

Environment observations and actions use the shared verl-vla `DataProto`
contract. The model integration translates that canonical representation into
the tensors, processors, and action format expected by the native policy.

Model-specific assumptions therefore stay inside the integration instead of
spreading into environments, trainers, or workers.

### Implement capabilities explicitly

A model only implements the training capabilities it supports:

- The **SFT contract** exposes the supervised loss used by SFT-style trainers.
- The **SAC-capable contract** exposes action sampling for the current
  environment rollout path together with the actor, critic, state-feature, and
  target-network operations required by off-policy reinforcement learning.

Trainers select models through these contracts rather than calling
model-specific methods.

### Keep native exports usable

verl-vla distinguishes a full training checkpoint from a native policy
export. The full checkpoint contains the wrapper, optimizer, and other state
required to resume training. The native export contains the policy and its
required processors or artifacts in the format expected by the upstream
implementation.

## Integration boundary

| Component | Responsibility |
| --- | --- |
| Native policy | Own the original model architecture and policy behavior. |
| Trainable wrapper | Contain one native policy and expose the supported verl-vla training contracts. |
| Input/output adapter | Translate `DataProto` observations and actions to and from the native policy interface. |
| Model builder | Load the native policy and assemble its wrapper and adapter explicitly. |
| Hydra configuration | Select the model artifact, adapter settings, optional model overrides, and training features such as LoRA. |
| Native export | Save a checkpoint that remains loadable by the upstream implementation. |

## Integrating a new model

### 1. Keep the upstream implementation intact

Add a model integration package under `verl_vla.models` that imports the
upstream policy from its original package. Treat that package's configuration,
processor, and checkpoint as native artifacts rather than redefining them in
verl-vla.

A typical integration keeps the following responsibilities together:

```text
models/my_policy/
├── adapter_config.py
├── trainable_model.py
└── policy/
```

The exact file layout can follow the needs of the policy, but the native
policy, framework configuration, and environment-specific I/O translation
should remain separate concepts.

### 2. Create a trainable wrapper

The wrapper inherits `TrainableVLAModelBase`, which is an `nn.Module` that
contains one native policy. It also opts into the training contracts supported
by the model:

```python
class MyTrainableModel(
    TrainableVLAModelBase,
    SupportSFTTraining,
):
    def __init__(self, policy, adapter_config):
        super().__init__(policy=policy)
        SupportSFTTraining.__init__(self, adapter_config)

    def sft_loss(self, obs, tokenizer, actions, valids, **kwargs):
        ...
```

An SFT-only policy only needs its SFT behavior. Implement the SAC-capable
contract when the policy is intended for the current environment rollout path
or RL post-training.

### 3. Implement canonical I/O translation

Read observations and actions from their canonical `DataProto` keys and
convert them once at the model boundary. The adapter should own operations
such as:

- selecting and ordering camera observations;
- applying native image, language, and state processors;
- normalizing state and action values;
- matching the native action horizon and dimensions; and
- converting sampled actions into a `ModelOutput` whose `to_data_proto`
  method returns the shared rollout representation.

Do not make environments emit model-specific tensors, and do not make
trainers guess which input schema a model expects.

### 4. Add an explicit builder

Extend `build_vla_model` with one explicit branch that:

1. loads the native configuration and policy;
2. applies only the supported native configuration overrides;
3. constructs the model-specific adapter; and
4. returns the trainable wrapper.

Also teach `VLAModelConfig` how to recognize the native checkpoint metadata
and select that builder. Explicit construction makes supported models visible
and avoids global AutoClass registration or implicit model conversion.

### 5. Add composable configuration

Place model-specific runtime settings in an adapter configuration under
`workflows/config/model/adapter/`. Add a model override config only when the
native policy supports intentional construction-time overrides.

The shared model configuration continues to own common choices such as:

- the checkpoint or Hugging Face repository path;
- tokenizer or processor loading;
- adapter selection;
- optional native overrides; and
- LoRA settings.

Keep adapter settings out of the native model config when they exist only to
control verl-vla behavior.

### 6. Preserve checkpoint semantics

Implement `export_policy` when the generic native export is insufficient.
Extract the native policy weights from the wrapper and save all artifacts
required by the upstream loader, such as processors or normalization
statistics. Framework-only state should remain in the full verl checkpoint
unless it is required for native policy inference.

## Integration checklist

A model integration is complete when:

- its native checkpoint can be loaded without converting the upstream model;
- the wrapper implements only the training and rollout contracts it supports;
- all model-specific observation and action translation stays at the model
  boundary;
- `build_vla_model` constructs the integration explicitly;
- Hydra can select the model and its adapter without duplicating a workflow;
- the supported trainer can run through `TrainCluster`; and
- the exported policy can be loaded through the upstream implementation.

Following these boundaries allows a new model to reuse the existing SFT, RL,
rollout, evaluation, and checkpoint infrastructure without creating a
model-specific execution pipeline.
