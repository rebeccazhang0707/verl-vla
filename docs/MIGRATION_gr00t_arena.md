# Migration: GR00T + Isaac Lab Arena + SAC → `verl-vla`

Living document tracking the multi-phase migration of the existing **GR00T (N1.6) +
Isaac Lab Arena + SAC** adaptation from the source veRL fork into the standalone
`verl-vla` package.

> Status: **Phase 6 complete — migration complete (pending docker-only sign-off gates).**
> All code/config/test/doc work is landed and CPU-validated: Phases 1–5, the **Phase 4 P0
> decode-semantics fix** (chunk-level fixed-base decode — §8.J), the **#2 action-horizon
> fail-fast guard** (§8.K), and **Phase 6** = the 4 P1 review items closed (§11), the
> `eval`/`smoke` scripts migrated with a CPU import/argparse test (§11.B), and the
> **docker-only mandatory verification gates** itemised (§11.C). What remains is **execution
> only** (no code left to write): run the docker gates in §11.C — first real `verl_vla`
> `--cfg job` compose, the #1 numerical decode equivalence proof, the end-to-end
> env_worker⇄env_loop⇄trainer trajectory, and FSDP2 forward — none runnable on the CPU host.

---

## 1. Repos & paths

| Role | Path | Import prefix | Layout |
|------|------|---------------|--------|
| **Source** (gr00t work already landed; embedded veRL fork) | `/project/VLA_RL/verl_isaac_project/libero_rl_example/verl/verl/experimental/vla/` | `verl.experimental.vla.*` | in-tree fork |
| **Target** (standalone package, treats veRL as an upstream lib) | `/project/VLA_RL/verl_isaac_project/verl-vla/` | `verl_vla.*` | `src/` layout (`src/verl_vla/`) |

Target repo assets already in place:
- Checkpoint: `checkpoints/gr1_ranch_bottle_into_fridge/checkpoint-5000-export`
- Arena symlink: `_isaaclab_arena -> /project/Arena/Arena_lab_3`
- GR1 joint-space YAMLs: `_isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1/{gr00t_26dof,36dof,54dof}_joint_space.yaml`

Test conventions (`pyproject.toml` → `[tool.pytest.ini_options]`): `testpaths = ["tests"]`,
`pythonpath = ["src"]`. ruff `line-length = 120`.

---

## 2. Source → target file map

Legend: ✅ migrated (Phase 1) · ⏳ pending (later phase) · 🔁 import/path rewrite applied.

### Code

| Source (`verl/.../experimental/vla/`) | Target (`src/verl_vla/`) | Status | Notes |
|---|---|---|---|
| `models/gr00t/utils.py` | `models/gr00t/utils.py` | ✅ | verbatim (no `verl.*` imports; stdlib + numpy only) |
| `models/gr00t/gr00t_policy.py` | `models/gr00t/gr00t_policy.py` | ✅ 🔁 | import `verl.experimental.vla.models.gr00t.utils` → `verl_vla.models.gr00t.utils` |
| `envs/arena_env/embodiment.py` | `envs/arena_env/embodiment.py` | ✅ 🔁 | import + docstring `verl.experimental.vla.models.gr00t.utils` → `verl_vla.models.gr00t.utils` |
| `envs/arena_env/utils.py` | `envs/arena_env/utils.py` | ✅ | verbatim (stdlib only) |
| (new) | `models/gr00t/__init__.py` | ✅ | license-only (no eager imports, keeps helpers light) |
| (new) | `envs/arena_env/__init__.py` | ✅ | license-only |
| `models/gr00t/modeling_gr00t_sac.py` | `models/gr00t/modeling_gr00t_sac.py` | ✅ 🔁 | **Phase 2 core** — `Gr00tN1d6ForSAC`, Flow-SDE, action mask, eagle/transformers-4.51.3/cuDNN patches, `register_gr00t_sac`. Ported to the **new** SAC interface (see §6). Imports rewritten `verl.experimental.vla.sac.base` → `verl_vla.models.base`, gr00t `utils` → `verl_vla.models.gr00t.utils`. |
| `envs/arena_env/arena_env.py` | `envs/arena_env/arena_env.py` | ✅ 🔁 | **Phase 4 + P0 fix** — `IsaacLabArenaEnv`; imports rewritten `verl.experimental.vla.*` → `verl_vla.*` (`action_utils` → `utils.envs.action`). **Scheme Y**: `_wrap_obs` packs the eagle obs (`build_inputs`). **P0 decode fix (§8.J)**: `chunk_step` now decodes the **whole** normalised chunk **once** against a single chunk-start base (`_decode_chunk_to_policy_actions` → `(B, chunk, 26)`); `step` only 26→36 scatters an already-decoded action (no per-step decode, no live-state base). All Isaac/omni imports deferred inside methods; gymnasium / imageio / torchvision imports made CPU-safe. |
| (existing) | `workers/env/env_worker.py` | ✅ | **Phase 4** — added `simulator_type == "arena"` branch (mirrors isaac/libero/lerobot) → `EnvManager(env_cls=IsaacLabArenaEnv)`. The isaac-only `assert len(set(stage_state_ids))==1` is **not** extended to arena (arena uses distinct dummy state ids). |
| `sac/naive_rollout_gr00t.py` | `workers/rollout/naive_rollout_gr00t.py` | ✅ 🔁 | **Phase 3 → simplified in Phase 4 (scheme Y) → #2 guard (Phase 5).** `GR00TRolloutRob(NaiveRolloutRob)`. Emits **only** the ACTION-slot keys `action` (normalised) + `critic_value` + `log_probs`. `register_fsdp_forward_method` guarded by `self.module is not None`. Registered as a **string path**. **#2 fail-fast guard** at `__init__` via the pure helper `assert_action_horizon_invariant(num_action_chunks, critic_action_horizon, action_horizon)` (§8.K). |
| `eval_arena_gr00t.py` | `examples/arena_sac/eval_arena_gr00t.py` | ✅ 🔁 | **Phase 6** — imports rewritten `verl.experimental.vla.*` → `verl_vla.*`; aligned to **scheme Y**: `--actor sac` emits the NORMALISED chunk → `env.chunk_step` (env decodes once, fixed base, = `GR00TRolloutRob`); `--actor gr00tpolicy` outputs ABSOLUTE joints → per-step `env.step`. Eagle patch delegates to `modeling_gr00t_sac._patch_eagle_compat`. CPU-importable (`build_parser` split out; all heavy imports deferred). Anchors: `action_horizon=50`, `env_spacing=10.0`, `num_action_chunks`=`--chunk`. |
| `smoke_test_gr00t_arena.py` (referenced but absent in source) | `examples/arena_sac/smoke_test_gr00t_arena.py` | ✅ (new) | **Phase 6** — authored to the source README "Workflow A" spec (build_inputs → sac_sample_actions → decode_actions_flat → critic value → state features → grad actor → critic → bc_loss + backward; `8/8 PASS`). CPU-importable; docker-only to run. Optional `--compare-gr00t-policy` cross-check. |
| (new) | `scripts/prepare_arena_dataset.py` | ✅ | **Phase 5** — placeholder train/val parquet (single-task `task_ids=0`, `state_ids=0..N-1`); pure `datasets`/`pandas`. |
| `models/register_vla_models.py` (`register_gr00t_sac`) | `models/register_vla_models.py` | ✅ | **Phase 2** — guarded registration (`ImportError` → `logger.warning` + skip) so gr00t deps never pollute pi0/openvla. **Only existing target file modified this phase.** |

### Tests

| Source | Target | Status | Notes |
|---|---|---|---|
| `models/gr00t/utils_test.py` | `tests/models/gr00t/utils_test.py` | ✅ 🔁 | `importorskip` target rewritten to `verl_vla.models.gr00t.utils` |
| `envs/arena_env/embodiment_test.py` | `tests/envs/arena_env/embodiment_test.py` | ✅ 🔁 | `importorskip` → `verl_vla.envs.arena_env.embodiment` |
| `envs/arena_env/arena_env_test.py` | `tests/envs/arena_env/arena_env_test.py` | ✅ 🔁 | **Phase 4** — no longer self-skips. `conftest.py` installs a `verl_vla.models` namespace shim (real `__path__`, no `register_vla_models`) so the pure-numpy gr00t leaf modules import on a minimal CPU host. Covers `_build_args`/`_init_env`, `_extract_image_and_state` (54→26 gather), `_wrap_obs` packing, `step` decode→scatter order, `_calc_step_reward`, `chunk_step` `ever_done`. 13 tests. |
| (new) | `tests/env_loop/test_gr00t_transition_prefix_chain.py` | ✅ | **Phase 4** integration test (reviewer-required): synthetic env-obs + simplified rollout output → `stack_dataproto_with_padding("obs"/"action")` → `add_transition_prefixes`, asserting `t0.obs.images`, `t1.obs.*`, and `t0.action.action` (normalised shape). Locks the env→`obs.*` / rollout→`action.*` provenance chain. 1 test. |
| (new) | `tests/models/gr00t/modeling_gr00t_sac_test.py` | ✅ | **Phase 2** — CPU-only mock shape tests for `Gr00tN1d6ForSAC`. Stubs gr00t/transformers/verl in `sys.modules`, builds the model via `object.__new__` + fake `action_head`/`backbone`, uses **real** `CriticMLP` + `split_nested_dicts_or_tuples`. 13 tests, runs on bare CPU (torch + numpy only). |
| (new) | `tests/models/__init__.py`, `tests/models/gr00t/__init__.py`, `tests/envs/__init__.py`, `tests/envs/arena_env/__init__.py` | ✅ | package markers w/ license headers |
| (new) | `tests/workers/rollout/naive_rollout_gr00t_test.py` | ✅ | **Phase 3** — CPU-only mock tests for `GR00TRolloutRob`. Stubs `verl`/`verl.utils.device`/`NaiveRolloutRob` in `sys.modules`, loads the rollout by path, builds it via `object.__new__` + fake `module`/`adapter`. 10 tests, bare CPU (torch + numpy only). |
| (new) | `tests/workers/engine/sac/training_worker_bc_test.py` | ✅ | **Phase 6 (P1-1)** — CPU-only test of the `_forward_actor` BC branch. Stubs all heavy top-level deps, loads `training_worker.py` by path with **save/restore of `sys.modules`** (no leak), drives `_forward_actor` with a mocked engine module. Asserts: default `bc_loss_coef==0.0` → pure SAC + `bc_loss` NOT called (pi05 unchanged); `bc_loss_coef>0` → `sac_loss + coef*bc_loss`; `td3_enabled` wins. 3 tests. |
| (new) | `tests/examples/arena_sac/eval_smoke_cpu_test.py` | ✅ | **Phase 6 (B)** — CPU import + argparse smoke for the migrated `eval`/`smoke` scripts (loaded by file path). Asserts they import on a gr00t-free host and that `build_parser` defaults match the training anchors (`chunk=16`, `action_horizon=50`, `env_spacing=10.0`). 5 tests. |

### Config / scripts (all ⏳, later phases)

| Source | Target | Notes |
|---|---|---|
| `config/rob_sac_trainer_arena_gr00t.yaml` | (CLI overrides on `trainer/config/rob_sac_trainer.yaml`) | ✅ **Phase 5** — re-expressed as CLI overrides in the run scripts (no new YAML group; the existing `rob_sac_trainer` defaults tree already carries every field — see §9). |
| `run_gr00t_arena_sac.sh` | `examples/arena_sac/run_gr00t_arena_sac.sh` | ✅ **Phase 5** — single-node combined train/rollout + GPU env workers. |
| `run_gr00t_arena_sac_disagg.sh` | `examples/arena_sac/run_gr00t_arena_sac_disagg.sh` | ✅ **Phase 5** — separate env / train-rollout GPU pools (`env.disagg_sim.enable=True`). |
| `README_gr00t_arena.md` | (this doc, §11 + run-script headers) | ✅ **Phase 6** — the source README's workflows are captured here (§11.C docker gates) and in the `examples/arena_sac/run_*.sh` header comments. A standalone user-facing README is left for the reviewer's final pass (key inputs in §11.G). |

---

## 3. SAC model interface diff (three-way)

The decided direction (§4) is to upgrade gr00t's SAC model to the **target pi0** interface
rather than carrying over the old engine.

| Concern | Source pi0 / gr00t (old engine) | Target pi0 (new engine) | gr00t target (planned) |
|---|---|---|---|
| `sac_forward_state_features` | `(s: dict)` — no tokenizer; preprocessing on **rollout** side | `(obs: DataProto, tokenizer)` — preprocessing on **model** side | adopt `(obs, tokenizer)` |
| `sac_forward_actor` | `(state_features, is_first_micro_batch=False)` — no `task_ids` | `(sf, task_ids=None, is_first_micro_batch=False)` | add `task_ids=None` **but ignore** (arena single-task) |
| `sac_forward_critic` | `(a, state_features, ...)` — no `task_ids` | `(a, sf, task_ids=None, *, ...)` | add `task_ids=None`, ignore; keep **inline `CriticMLP`** |
| `bc_loss` | `(state_features, actions, valids)` | `(obs, tokenizer, actions, valids)` | adopt `(obs, tokenizer, actions, valids)` |
| rollout sampling | n/a (rollout owns preprocessing) | `sac_sample_actions(obs, tokenizer, validate)` / `sac_get_critic_value(obs, actions, tokenizer)` | add equivalents |
| critic backend | inline | pluggable `critic_api` (`models/pi0_torch/critic/`, `uses_task_ids`) | **keep inline CriticMLP**, do *not* wire into `critic_api` |
| replay schema | `s0./s1./a0./a1.` prefixes | `t0.obs./t1.obs./t0.action./info.` + `SACReplayPool` (RLPD: online+offline, positive_only, per-`task_ids` pools) | adopt target schema |

---

## 4. Decided technical decisions (unless reviewer overrides)

1. **方案 1**: upgrade gr00t SAC model to the **target's new interface** (unify with target pi0);
   do **not** port the source's older SAC engine.
2. gr00t critic: **keep the existing inline `CriticMLP`**. Add `task_ids=None` params to
   `sac_forward_actor` / `sac_forward_critic` but ignore them internally (arena is single-task).
   Do **not** force-integrate the target's multi-task `critic_api`.
3. gr00t deps live only in the training Docker image ⇒ registration must be **guarded**
   (skip on import failure) so it never pollutes the pi0 / openvla paths.

---

## 5. Phase 1 — completed

### Done
- Branch `migrate/gr00t-arena-sac` created in the target repo.
- Migrated low-risk, new-interface-independent files (see §2): gr00t `utils.py` + `gr00t_policy.py`,
  arena_env `embodiment.py` + `utils.py`, plus new `__init__.py` package markers.
- Migrated the three CPU unit tests with `importorskip` targets rewritten to `verl_vla.*`.
- Direct-import CPU validation of the migrated logic (bypassing the eager package `__init__`):
  GR1 → `action_dim=26`, `embodiment_id=20`; YAML-derived index tables `policy_dim=26`,
  `sim_action_dim=36`, `state_full_dim=54`, all-unique gather/scatter indices.

### Notes / caveats
- **YAML discovery at import time**: `embodiment.py` resolves the GR1 ⇄ sim index tables at
  module import via `_resolve_maps()`. With no installed `isaaclab_arena_gr00t` package, set
  `ARENA_GR1_JOINT_SPACE_DIR=_isaaclab_arena/isaaclab_arena_gr00t/embodiments/gr1` (relative to
  the repo root) so `embodiment_test.py` can import. Otherwise it raises `RuntimeError`
  (which `pytest.importorskip` does **not** catch ⇒ would be a collection error, not a skip).
- **`models/__init__.py` eager import**: importing anything under `verl_vla.models.*` triggers
  `verl_vla/models/__init__.py → register_vla_models`, which imports `transformers` + `verl` +
  the openvla/pi0 submodules. This means the pure-numpy gr00t `utils.py` is **not importable**
  on a bare CPU env without the full model stack (pre-existing behaviour; identical in source).
  The new `models/gr00t/__init__.py` is intentionally license-only to avoid adding to this.

### Deferred to Phase 2+
- `modeling_gr00t_sac.py`, `arena_env.py`, `naive_rollout_gr00t.py`, config, run scripts,
  eval/smoke, and the guarded gr00t registration in `register_vla_models.py` (all coupled to
  the new SAC interface / worker architecture).

---

## 6. Phase 2 — SAC model interface upgrade (complete)

Ported `modeling_gr00t_sac.py` (`Gr00tN1d6ForSAC`) to the package's current
`SupportSACTraining` interface (the one `SACTrainingWorker` drives) and added a
**guarded** `register_gr00t_sac`. CPU-validated only — nothing here needs gr00t /
Isaac / a real checkpoint at test time.

### A. Final interface signatures (what the worker calls)

| Method | Final signature | Notes |
|---|---|---|
| `sac_forward_state_features` | `(self, obs: DataProto, tokenizer=None) -> dict[str, Tensor]` | Unpacks the eagle tensors the rollout packed into `obs.batch` (`images / lang_tokens / lang_masks / states`) via `_obs_to_state_dict`, then runs `_state_features_impl`. **`tokenizer` ignored** (GR00T runs its own processor on the rollout side). |
| `sac_forward_actor` | `(self, state_features, task_ids=None, is_first_micro_batch=False) -> (actions, log_probs, metrics)` | `task_ids` accepted for parity, **ignored** (single-task). `_denoise` logic unchanged. |
| `sac_forward_critic` | `(self, a, state_features, task_ids=None, *, use_target_network=False, method="cat", requires_grad=False) -> Tensor` | `task_ids` ignored. Reads **`a["action"]`** (was `a["full_action"]`). `cat → (B, num_heads)`, `min → (B,)`. Inline `CriticMLP` ensemble kept (not wired into pi0's `critic_api`). |
| `bc_loss` | `(self, obs: DataProto, tokenizer, actions, valids) -> Tensor` | Was `(state_features, actions, valids)`. Now computes state features **inline** via `_state_features_impl` (not the registered `sac_forward_state_features`, to avoid a nested FSDP-forward boundary). Demo read from **`actions["action"]`**. |
| `sac_sample_actions` *(new)* | `(self, obs: DataProto, tokenizer=None, validate=False) -> dict` | Rollout entry. `@torch.no_grad`. Returns a **dict** (see return contract below). |
| `sac_get_critic_value` *(new)* | `(self, obs, actions, tokenizer=None) -> Tensor` | `@torch.no_grad`, `method="min" → (B,)`. Accepts `actions` as the `sac_sample_actions` dict **or** any object exposing `.action`. |
| `sac_init` | registers `[bc_loss, sac_sample_actions, sac_forward_critic, sac_forward_actor, sac_forward_state_features]` (+ `sac_update_target_network`). Pre-FSDP critic-head build in `__init__` (gated by `config.sac_enable`) retained. |

`sac_get_critic_parameters` / `sac_get_named_actor_parameters` / `sac_update_target_network`
unchanged.

### B. gr00t-specific patches preserved verbatim (do not delete)

`_disable_cudnn_sdpa()` (Hopper cuDNN-SDPA off), `_patch_eagle_compat()`
(transformers-4.51.3 `_attn_implementation_autoset` shim + forced FA2), and
`register_gr00t_sac`'s `AutoModel.register(..., exist_ok=True)` with the
old-transformers `TypeError` fallback. The full **Flow-SDE** sampler
(`_denoise` / `_run_flow`: `s = 1 − t` map, β anneal, log-prob) and the
**two-pass action mask** decoupling (`sac_action_train_dims`: explored vs base
trajectory from the same initial noise; `_gaussian_log_prob` mask normalisation)
are unchanged except for the `full_action → action` critic key rename.

### C. `full_action` → `action` key rename

The new replay/worker use a single `action` key. Changed `_critic_input`
(`a["action"]`) and `bc_loss` demo (`actions["action"]`). The legacy standalone
`sample_actions` (which returned `{"full_action": ...}`) was **removed**; rollout
sampling now goes through `sac_sample_actions`.

### D. None-free state-features contract (correctness-critical)

The worker runs `split_nested_dicts_or_tuples(state_features, 2)` on the dict
returned by `sac_forward_state_features`; that util raises `TypeError` on any
`None` value. The Eagle backbone can yield `image_mask = None`, so
`_state_features_impl` now **drops the `image_mask` key entirely** when it is
`None` (rather than storing `None`). `_run_flow` reads it with
`sf.get("image_mask", None)`, so a missing key is treated exactly like the
original `None` — the DiT still gets `image_mask=None` and behaves identically.
A dedicated test asserts `split_nested_dicts_or_tuples(sf, 2)` does not raise for
both the mask-present and mask-absent cases, and that every value is a `Tensor`.

### E. Guarded registration

`register_vla_models.py` (the **only** existing target file modified this phase)
gained `import logging` + a module `logger`, a `_REGISTERED_MODELS["gr00t_sac"]`
idempotency flag, and a new `register_gr00t_sac_model()` appended to
`register_vla_models()`:

```python
try:
    from .gr00t.modeling_gr00t_sac import register_gr00t_sac
    register_gr00t_sac()
    _REGISTERED_MODELS["gr00t_sac"] = True
except ImportError as e:
    logger.warning("gr00t SAC not registered (deps absent): %s", e)
```

The module's top-level `from gr00t... import Gr00tN1d6` raises `ImportError`
wherever gr00t is absent (i.e. everywhere except the training Docker image), and
that is swallowed here — so the pi0 / openvla registrations are never affected.

### F. `sac_sample_actions` / `sac_get_critic_value` return contract (for Phase 3)

Unlike pi0 (which returns a `Pi0Output`), GR00T's `sac_sample_actions` returns a
plain **dict**:

```python
{
    "action":    Tensor (B, action_horizon, max_action_dim),  # RAW, normalised
    "log_probs": Tensor (B,),                                  # zeros when flow-SDE off
}
```

- The `action` is the **raw, model-space (normalised)** action. **Un-normalising /
  decoding to the real 26-DOF action is the rollout's job** (the GR00T processor /
  `GR00TN16Adapter.decode_actions_flat`) — the model does *not* un-normalise (pi0
  does, via `action_unnormalize_transform`; GR00T deliberately does not).
- `sac_get_critic_value` accepts this dict (reads `actions["action"]`) or any
  `.action`-bearing object.

> **Open question for the reviewer / Phase 3:** confirm this dict shape (vs. a
> thin `ModelOutput`/`Pi0Output`-style wrapper) is acceptable for the rollout, and
> that the rollout owns un-normalisation. If a wrapper is preferred, only these two
> methods + the rollout glue change.

## 7. Phase 3 — Rollout worker (complete; partly SUPERSEDED by Phase 4)

Migrated `GR00TRolloutRob` to `workers/rollout/naive_rollout_gr00t.py`, registered via the
`_ROLLOUT_REGISTRY` string-path mechanism, and wired the Phase 2 `sac_sample_actions` /
`sac_get_critic_value` model entry points. CPU-validated only.

> ⚠️ **Superseded by Phase 4 (scheme Y).** Subsections **D** and **E** below describe the
> original rollout that *also* emitted `obs.*` and a decoded env `action` plus the normalised
> `action.action`. Phase 4 moved obs packing + action decoding **into the env** and simplified
> the rollout to emit **only** `action` (normalised) + `critic_value` + `log_probs`. Read §8 for
> the current contract. Subsections A–C (why a dedicated rollout, registration, constructor)
> still hold (minus the now-unused `GR00TN16Adapter` build in the rollout constructor).

### A. Why a dedicated rollout (HFRollout cannot be reused)

`HFRollout.generate_sequences` ends with `ret = output.to_data_proto()`. GR00T's
`sac_sample_actions` returns a **plain dict** (`{"action", "log_probs"}`) with **no
`.to_data_proto()`** → reusing `HFRollout` would raise `AttributeError`. GR00T also needs
rollout-side **pre-packing** (`GR00TN16Adapter.build_inputs` runs the checkpoint processor +
collator) and **post-decoding** (`decode_actions_flat` un-normalises → 26-DOF joints), which
`HFRollout` does not do. So a separate `GR00TRolloutRob` + a new registry key is required.
The model emits the normalised action dict; the **rollout** owns packing / decoding / DataProto
assembly. `info.*` keys (`task_ids` / `positive_sample_mask` / `rewards` / `dones` / `valids`)
are produced by the **trainer** (`_prepare_actor_input`), **not** the rollout.

### B. Rollout selection / registration

Registered as **string paths** in `workers/rollout/base.py::register_vla_rollouts()` (exactly
like `hf`), so `get_rollout_class(name, mode)` imports the module **lazily** — a gr00t-free
host never triggers the import:

```python
("gr00t", "sync"):          "verl_vla.workers.rollout.naive_rollout_gr00t.GR00TRolloutRob",
("gr00t", "async"):         "verl_vla.workers.rollout.naive_rollout_gr00t.GR00TRolloutRob",
("gr00t", "async_envloop"): "verl_vla.workers.rollout.naive_rollout_gr00t.GR00TRolloutRob",
```

`workers/rollout/__init__.py` is **intentionally not** changed to eager-`import GR00TRolloutRob`
(gr00t deps only exist in the image). Selected at runtime via `rollout.name=gr00t` (Phase 5 config).

### C. Constructor signature (adapted to the target call site)

The actual instantiation point is `VLAActorRolloutRefWorker.init_model`
(`workers/engine/engine_workers.py`):

```python
rollout_cls(config=rollout_config, model_config=model_config,
            device_mesh=rollout_device_mesh,
            engine=self.actor.engine if "actor" in self.role else None,
            tokenizer=self.tokenizer)
```

So `GR00TRolloutRob.__init__` mirrors **`HFRollout`** —
`(config, model_config, device_mesh, engine=None, module=None, tokenizer=None, **kwargs)` —
**not** the legacy bare-`module` `PI0RolloutRob(model_config, module, tokenizer)` signature
(`PI0RolloutRob` is stale w.r.t. this call site and is not in the registry). `self.module` is
taken from the shared actor `engine.module` (or an explicit `module`). `NaiveRolloutRob.__init__`
is **not** called (it would load an OpenVLA checkpoint); attributes used by the inherited
`update_weights/release/resume` are set directly. The `GR00TN16Adapter` is imported **inside**
`__init__` (lazy) and built from `model_config.path` + `embodiment_tag` (default `gr1`);
`action_dim` / `num_action_chunks` are read from `model_config` (attr/dict) with embodiment-spec
defaults (`GR1.action_dim=26`, `GR00TDim.ACTION_HORIZON`). Critic registration/compute is gated by
`config.output_critic_value` (default `True`).

### D. Normalised vs decoded action separation + obs-key duality

The single most important correctness rule (⚠️ do not mix):

| Output key | Space | Shape | Consumer |
|---|---|---|---|
| `action` | **decoded** absolute 26-DOF joints | `(B, num_action_chunks, 26)` | env (`env.step`, 26→36 scatter) |
| `action.action` | **normalised** model action | `(B, action_horizon, max_action_dim)` | replay buffer / critic / actor (training space) |

`action` = `adapter.decode_actions_flat(action_norm, raw_state_groups)[:, :num_action_chunks]`;
`action.action` = the raw `sac_sample_actions` output (model does **not** un-normalise).

**obs-key duality** (also do not mix):
- **obs handed to the model** (`sac_sample_actions` / `sac_get_critic_value`) uses
  **un-prefixed** keys `images / lang_tokens / lang_masks / states` — exactly what
  `Gr00tN1d6ForSAC._obs_to_state_dict` reads from `obs.batch`.
- **obs stored for replay** uses **`obs.`-prefixed** keys `obs.images / obs.lang_tokens /
  obs.lang_masks / obs.states` (CPU tensors). The trainer slices these into `t0.obs.*` /
  `t1.obs.*` (`add_transition_prefixes`); the actor then calls
  `get_dataproto_from_prefix(., "t0.obs.")` which strips `t0.obs.` → `images/...`, landing
  back exactly where `_obs_to_state_dict` reads. The eagle `pixel_values` (a per-sample list)
  is stacked to `(B, n_patches, C, H, W)` so the replay buffer can store it;
  `_state_features_impl` restores the per-sample list.

### E. `generate_sequences` output DataProto

`obs.images`, `obs.lang_tokens`, `obs.lang_masks`, `obs.states` (CPU), `action.action`
(normalised), `action` (decoded env chunk), `critic_value (B,)` (when `output_critic_value`),
`log_probs (B,)`. `task_descriptions` are read from `prompts.non_tensor_batch`; `task_ids` are
supplied by the trainer/prompts (Phase 4/5), not emitted here.

### F. CPU tests (`tests/workers/rollout/naive_rollout_gr00t_test.py`, 7 tests)

Stub-load pattern from Phase 2: minimal functional `verl.DataProto`, `get_device_id→"cpu"`, a
stub `NaiveRolloutRob` base; `object.__new__(GR00TRolloutRob)` bypasses `__init__`; fake `module`
(fixed-shape `sac_sample_actions` / `sac_get_critic_value`) + fake `adapter` (`build_inputs`
returns eagle tensors + `raw_state_groups`; `decode_actions_flat` returns a constant-fill
`(B, H, 26)`). Asserts: output keys/shapes (`obs.*`, `action.action`, env `action`,
`critic_value`, `log_probs`); normalised vs decoded **separated** (different shapes + decoded ==
sentinel); model obs un-prefixed vs replay obs `obs.`-prefixed (and same obs object reused for the
critic); `decode_actions_flat` / critic called exactly once; `output_critic_value=False` drops
`critic_value` + skips the critic; `num_action_chunks > horizon` asserts; `register_vla_rollouts()`
registers `("gr00t", *)` as **str** values (no import). Result: **7 passed**; Phase 1/2 gr00t
tests still **22 passed**.

### Phase 3/4 data-schema dependencies (must be honoured downstream)

The new `SACTrainingWorker` (target `workers/engine/sac/training_worker.py`) reads
these **unconditionally**, so the Phase 3 rollout + Phase 4 env worker must emit them:

- **`info.task_ids`** — read on every replay add (`_add_data_to_replay_pool`) and
  every critic/actor forward. Arena is single-task but **must still emit a constant**
  (e.g. all-zeros `int64 (B,)`). The model ignores its *value*, but the worker uses it
  to key the per-task replay pools and passes it into the (ignored) `task_ids` params.
- **`info.positive_sample_mask`** — `bool (B,)`; required by the offline
  `positive_only` RLPD pool and the positive/negative Q-value metrics.
- **`info.rewards` / `info.dones` / `info.valids`** — `(B,)` tensors consumed by the
  critic TD target and all valid-masked means.
- **obs / action prefixes** — obs under `t0.obs.*` / `t1.obs.*` (the eagle tensors
  `images / lang_tokens / lang_masks / states`); actions under `t0.action.action` /
  `t1.action.action` — i.e. the **single `action` key** (no `full_action`). The replay
  `add_transition_prefixes` already supports an `action.action` step key.

## 8. Phase 4 — Env worker integration (complete · **scheme Y**)

Migrated `arena_env.py` and wired `arena` into `env_worker.py`. The key decision — **scheme Y** —
makes GR00T reuse the **exact pi05 data channel**: the **env** owns obs packing + action
decoding, and the rollout is reduced to a thin `sac_sample_actions` wrapper. CPU-validated only
(no Isaac / gr00t / Docker run).

### A. Why scheme Y (the data-flow fact that forced it)

In the target `env_loop` pipeline the prefixes are assigned by *slot*, not by the producer's
key names:
- **`obs.*` always comes from the env.** `env_worker.create_env_batch_dataproto` maps each
  `images_and_states` key → `obs.<key>`; `env_loop._collate_trajectories` then
  `stack_dataproto_with_padding(..., "obs")`; the trainer's `add_transition_prefixes` slices to
  `t0.obs.*` / `t1.obs.*`.
- **The whole rollout output goes into the ACTION slot** and is uniformly `action.`-prefixed
  (`stack_dataproto_with_padding(..., "action")`).

⇒ A rollout that emits its own `obs.*` / `action.action` (the Phase 3 schema) would be
**mis-prefixed** under this pipeline (`action.obs.images`, `action.action.action`), and the env's
own decoded `action` would collide. So obs packing + action decoding must live in the **env**.

### B. The three scheme-Y changes

1. **Env packs obs** (`IsaacLabArenaEnv._wrap_obs`): runs
   `GR00TN16Adapter.build_inputs(full_image, state_26, task_descriptions)` and returns the eagle
   tensors `images / lang_tokens / lang_masks / states` as the `images_and_states` keys (eagle
   `pixel_values` list → stacked `(B, n_patches, C, H, W)`). The pipeline turns these into
   `obs.images/...` → `t0.obs.*`, exactly the slots `Gr00tN1d6ForSAC._obs_to_state_dict` reads
   (**Phase 2 model unchanged**). `full_image` is kept at the **top level** of the obs dict for
   video only (`create_env_batch_dataproto` ignores it).
2. **Env decodes the action** (`IsaacLabArenaEnv.chunk_step`, **corrected in the P0 fix §8.J**):
   at chunk entry, `_decode_chunk_to_policy_actions(chunk_actions)` builds the base
   `raw_state_groups` **once** from the chunk-start state (`_last_state26`) and decodes the **whole**
   normalised chunk in a single `adapter.decode_actions_flat(chunk_actions, base_groups)` →
   `(B, chunk, 26)`. The per-step `step` then only `joint_map.scatter_action`-expands the already
   decoded `decoded[:, i]` 26 → 36 sim joints (no per-step decode, no live-state base). `_wrap_obs`
   still updates `_last_state26` so it serves as the base for the **next** chunk (one inference per
   chunk, matching the env-loop cadence).
3. **Rollout simplified** (`naive_rollout_gr00t.py`): reads the already-packed obs from
   `prompts.batch["images"/"lang_tokens"/"lang_masks"/"states"]`, calls `sac_sample_actions(prompts)`,
   and emits **only** `action` (normalised chunk, `(B, num_action_chunks, max_action_dim)`),
   `critic_value (B,)` (gated by `output_critic_value`), `log_probs (B,)`. **Removed**: the
   `GR00TN16Adapter.build_inputs` / `decode_actions_flat` calls, the `obs.*` emission, and the
   separate `action.action` key. Result: `t0.obs.*` = env-packed, `t0.action.action` = normalised
   (critic/replay work in the normalised space) — all correct **without touching `env_loop` or the
   Phase 2 model**.

### C. obs / action → `t0.*` flow (end-to-end)

```
env._wrap_obs ─► images_and_states{images,lang_tokens,lang_masks,states}
   └► create_env_batch_dataproto ─► obs.images / ...                 (env_worker)
        └► env_loop next_obs = get_dataproto_from_prefix(.,"obs") ─► images / ... (un-prefixed)
             └► rollout.generate_sequences reads prompts.batch["images"/...]  (model obs)
                  └► sac_sample_actions ─► {"action"(norm), "log_probs"}
                       └► rollout emits action / critic_value / log_probs (ACTION slot)
   _collate_trajectories: stack("obs")→obs.* , stack("action")→action.* (+ critic_value/log_probs)
   add_transition_prefixes: obs.* →t0.obs.*/t1.obs.* , action.action →t0.action.action/t1.action.action
```

Locked by `tests/env_loop/test_gr00t_transition_prefix_chain.py`.

### D. Preserved source fixes (verified line-by-line)

| Fix | Status | Note |
|---|---|---|
| `disable_lightwheel_ssl_verify()` (TLS bypass) | ✅ kept | called in `__init__` and re-asserted in `_init_env` (idempotent) |
| `_apply_rl_reward_and_disable_autoreset` | ✅ kept | success→`RewTerm(weight=1/step_dt)` + null every termination term (no auto-reset). Powers the trainer's `complete_any = feedback.terminations.any(-1)` chain (success→reward>0→`chunk_step` `ever_done`→terminations). Isaac imports moved **after** the `terminations.success is None` early-return (behaviour-preserving; lets the guard run on a CPU host). |
| HDF5 `DatasetExportMode.EXPORT_NONE` | ✅ kept | avoids the multi-worker `/tmp` h5 `errno 11` lock |
| Lazy Isaac/omni imports | ✅ kept | every `isaaclab*` / `omni` import is inside a method (`__init__`, `_init_env`, `_apply_rl_reward...`) — module top is CPU-importable |
| `chunk_step` first-done (`ever_done`) | ✅ kept | cumulative OR of `reward>0`; latches done for the rest of the chunk |

### E. CPU-importability adjustments (new, minimal)

The minimal CPU test host lacks `gymnasium` / `imageio` / `torchvision` (the latter two are
imported at the top of `utils/envs/action.py`). To keep the module top-level importable:
- `import gymnasium` is guarded (falls back to `object` as the base class; CPU tests use
  `object.__new__` anyway);
- `to_tensor` is imported from `utils.envs.action` when available, else a local CPU-safe copy is
  used; the video-only helpers (`put_info_on_image` / `tile_images` / `save_rollout_video`) are
  imported **lazily** inside `add_new_frames` / `flush_video`.

These are inert in the Docker image (all deps present) and only matter for the unit tests.

### F. `task_ids` / `state_ids` injection — Phase 5 (dataset layer), NOT here

The trainer reads `non_tensor_batch["task_ids"]` / `["state_ids"]` from **dataset rows**
(`_next_rollout_batch` → `_get_gen_batch` pops all non-tensor keys; `_reset_envs` reads both).
Arena is single-task, so each train-dataset row must carry a **constant `task_ids=0`** and a
**dummy `state_ids`**. This belongs to the **dataset / dataloader** (Phase 5 data prep) and is
deliberately **not** hacked into the env or rollout. The env exposes `get_all_state_ids()` →
`range(num_envs)` (dummy) and `reset_envs_to_state_ids(state_ids, task_ids)` (re-inits Arena,
ignoring the ids), so the interface is ready for the Phase 5 dataset to drive it.

`info.*` (`task_ids` / `positive_sample_mask` / `rewards` / `dones` / `valids`) is derived by the
trainer's `_prepare_actor_input` from `feedback.terminations` — the env/rollout emit **none** of it.

### G. env_worker arena branch

`init_worker` gained an `elif self.cfg.train.simulator_type == "arena"` branch that appends
`EnvManager(..., env_cls=IsaacLabArenaEnv, ...)` per pipeline stage (identical shape to the
isaac/libero/lerobot branches). `reset_envs_to_state_ids`'s isaac-only
`assert len(set(stage_state_ids)) == 1` is intentionally **not** extended to arena (arena's dummy
state ids are distinct by design); arena falls through with no assert.

### H. CPU tests + results

- `tests/envs/arena_env/arena_env_test.py` (**13**) — `conftest` namespace shim lets the real
  module import; `object.__new__` + stub adapter / fake sim env cover the pure logic
  (`_extract_image_and_state` 54→26 gather, `_wrap_obs` eagle-key packing + state caching, `step`
  decode→scatter **order** and 26→36 correctness, `_calc_step_reward` abs/rel, `chunk_step`
  monotonic `ever_done`, `get_all_state_ids`, plus the existing `_build_args` / `_init_env`).
- `tests/env_loop/test_gr00t_transition_prefix_chain.py` (**1**) — the prefix-chain integration
  test (§C).
- `tests/workers/rollout/naive_rollout_gr00t_test.py` (**7**, rewritten) — simplified schema:
  emits only `action` / `critic_value` / `log_probs`, no `obs.*`, no `action.action`; model sees
  the un-prefixed prompts as-is; critic evaluated on the emitted chunk; `output_critic_value=False`
  skips the critic; `num_action_chunks > horizon` asserts; registry still string-path.

Run (collection order) `tests/env_loop tests/envs/arena_env tests/models/gr00t
tests/workers/rollout/naive_rollout_gr00t_test.py` ⇒ **48 passed** (was 29 passed + 2 skipped;
the 2 arena modules now run). `ruff check` clean on all touched files.

### I. Risks / open questions for the reviewer

1. **Per-step decode vs chunk-base-state decode (semantics).** ✅ **RESOLVED — see §8.J (P0 fix).**
   The Phase-4 scheme-Y implementation decoded each step against the **live** raw joint state,
   which for a `use_relative_action=true` checkpoint accumulates offsets (`base+δ[i-1]+δ[i]`) and
   diverges. The fix decodes the **whole chunk once** against the single chunk-start base
   (`base+δ[i]`), matching the source `naive_rollout_gr00t.py`. Numerical equivalence is still
   **Docker-only** (CPU stub decode is constant) → Phase 6.
2. **`num_action_chunks` vs `critic_action_horizon`.** ✅ **RESOLVED — see §8.K (#2 guard).** Now
   enforced at rollout-worker `__init__` via `assert_action_horizon_invariant`:
   `critic_action_horizon ≤ num_action_chunks ≤ model action_horizon`. The run scripts wire all
   three from a single `NUM_ACTION_CHUNKS` anchor (§9).
3. **Docker-only, unverified on CPU:** real Arena env build (`_init_env`, `build_registered`,
   RL-reward patch with a live `terminations.success`, HDF5 EXPORT_NONE), `AppLauncher`, real
   `env.step`, real `GR00TN16Adapter` (eagle processor + `decode_action`), video flush, and the
   end-to-end `env_worker` ⇄ `env_loop` ⇄ trainer loop. Unit tests stub all of these.
4. **`task_ids=0` injection requires Phase 5 dataset work** (§F) — not deliverable purely in
   env/rollout.
5. **`envs/action` path confirmed:** the shared helpers live at `verl_vla.utils.envs.action`
   (not `verl_vla.envs.action`); arena_env imports from there.

## 8.J Phase 4 **P0 fix** — chunk-level fixed-base action decode (correctness)

**Bug.** The Phase-4 scheme-Y env (§8.B.2) decoded the action **per env step** inside `step()`,
rebuilding the relative→absolute base from the **live** `_last_state26` that `_wrap_obs` updates
every step. The checkpoint is `use_relative_action=true`, so a per-step live base makes offsets
**accumulate** (`base + δ[i-1] + δ[i] + …`) and the joint targets diverge — whereas the source
decodes every step relative to a **single** chunk-start base (`base + δ[i]`). Left unfixed this
corrupts the executed trajectory and the policy collapses on the first real training run.

**Source semantics replicated.** `sac/naive_rollout_gr00t.py` builds `raw_state_groups` from the
chunk-start observation (T=1, broadcast over the horizon) and decodes the **entire** normalised
chunk in one `adapter.decode_actions_flat(full_action_norm, raw_state_groups)` call.

**Fix (methods touched in `arena_env.py`).**

| Method | Before (bug) | After (fix) |
|---|---|---|
| `chunk_step` | loop calls `step(chunk_actions[:, i])`; each `step` decodes | decodes the whole chunk **once** at entry: `decoded = _decode_chunk_to_policy_actions(chunk_actions)` → `(B, chunk, 26)`; loop calls `step(decoded[:, i])` |
| `_decode_to_policy_action` *(removed)* → `_decode_chunk_to_policy_actions` *(new)* | `decode_actions_flat(actions[:, None, :], live raw_state_groups)` per step → `(B,1,26)` | builds base `raw_state_groups` **once** from chunk-start `_last_state26`, `decode_actions_flat(chunk_actions, base_groups)` → `(B, chunk, 26)` |
| `step` | `policy_action = _decode_to_policy_action(actions)` then 26→36 scatter | `policy_action = actions` (already-decoded 26-DOF) then 26→36 scatter only |
| `_wrap_obs` | updates `_last_state26` (also used as per-step base) | unchanged — `_last_state26` now serves as the base for the **next** chunk only |

**Stub test changes (`tests/envs/arena_env/arena_env_test.py`).** `_StubAdapter.decode_actions_flat`
dropped the hard `shape[1]==1` assert and now accepts the whole `(B, chunk, max_action_dim)`,
returns a **relative** decode `base + δ` (`(B, chunk, 26)`), and **records each `base`** it was
called with. New/changed tests:
- `test_step_scatters_decoded_action` — `step` receives an already-decoded 26-DOF action and only
  scatters 26→36 (no decode).
- `test_chunk_step_decodes_whole_chunk_against_fixed_base` (the P0 regression lock) — asserts
  `decode_actions_flat` is called **exactly once** per `chunk_step`, the recorded base is the fixed
  chunk-start state (not a drifting live state), and each executed `sim_action == scatter(fixed_base
  + δ_i)` (i ≥ 1 does **not** depend on the live state).

> Numerical correctness with the **real** checkpoint is Docker-only (the CPU stub decode is a
> deterministic `base+δ`); this phase fixes the **semantics** + locks them with the regression test.
> Real-value verification → Phase 6.

## 8.K #2 — action-horizon fail-fast guard

`workers/rollout/naive_rollout_gr00t.py` gained a pure, dependency-free helper
`assert_action_horizon_invariant(num_action_chunks, critic_action_horizon, action_horizon)` called
from `GR00TRolloutRob.__init__`:

```
assert critic_action_horizon <= num_action_chunks <= action_horizon
```

**Placement rationale.** The rollout-worker `__init__` is the earliest point where all three values
are available: `num_action_chunks` from `model_config`, and `critic_action_horizon` /
`action_horizon` from `model_config.override_config` (set by the run script; `action_horizon` falls
back to the checkpoint default `GR00TDim.ACTION_HORIZON`). A too-small `num_action_chunks` would let
the critic head silently truncate / zero-pad its action input (`sac_forward_critic` slices the first
`critic_action_horizon` steps); a too-large one exceeds the model decode horizon. Extracting it to a
helper makes the lower bound CPU-testable without constructing the full worker.

CPU tests (`tests/workers/rollout/naive_rollout_gr00t_test.py`): `test_action_horizon_invariant_ok`
(both bounds inclusive), `_too_small_truncates_critic`, `_too_large_exceeds_model`; plus the
pre-existing `generate_sequences` upper-bound check.

## 9. Phase 5 — Config & run scripts & dataset (complete · CPU-validated)

No new Hydra YAML group was added: the existing `trainer/config/rob_sac_trainer.yaml` defaults tree
(env / actor / rollout / model groups) already carries every field the Arena/GR00T channel needs, so
the source `rob_sac_trainer_arena_gr00t.yaml` is re-expressed as **CLI overrides** in the run
scripts (target convention — same style as `examples/libero_sac/`).

### A. Dataset — `scripts/prepare_arena_dataset.py`

Generates `train.parquet` / `test.parquet` under `<local_save_dir>/<arena_env_name>/`, schema
identical to `prepare_libero_dataset.py` (`data_source / prompt / state_ids / task_ids / ability /
extra_info`). Arena is **single-task** and resets to a **random** layout each episode
(`reset_envs_to_state_ids` ignores the value), so:

| Column | Value | Why |
|---|---|---|
| `task_ids` | `0` for every row | single task; model/critic ignore the value but the trainer keys per-task replay pools by it |
| `state_ids` | `0 .. N-1` (distinct placeholders) | Arena ignores the value, but `_reset_envs`/`reset_envs_to_state_ids` need one id per env slot (`len == num_envs * stage_num`) |
| `prompt` / `extra_info.task_description` | the live Arena task string | drives dataloader length; the live task still comes from the env |

Row-count constraint (matches the run-script batching): `TRAIN_BATCH_SIZE * ROLLOUT_N ==
NUM_ENV_WORKERS * NUM_STAGE * NUM_ENV`. Pure `datasets`/`pandas` — no Isaac/gr00t/torch. Verified:
`--num_train 16 --num_val 4` → two parquet files written.

### B. Run scripts — `examples/arena_sac/run_gr00t_arena_sac.sh` (+ `_disagg.sh`)

Based on `examples/libero_sac/run_pi05_libero_sac.sh`, with **`ENV_DEVICE=cuda`** (Arena runs Isaac
Sim — there is **no** CPU env path, so the LIBERO `MUJOCO_GL`/`osmesa` block is dropped). Disagg
mirrors `run_pi05_libero_sac_disagg.sh` (`env.disagg_sim.enable=True` + `+trainer.n_env_gpus_per_node`
/ `+trainer.n_rollout_gpus_per_node`). **Single anchor:** `NUM_ACTION_CHUNKS` (=16) is defined once
and fed to env, model and critic horizon.

Key Arena/GR00T fields (authority: source `config/rob_sac_trainer_arena_gr00t.yaml` +
`run_gr00t_arena_sac*.sh`):

| Override (CLI) | Value | Notes |
|---|---|---|
| `actor_rollout_ref.rollout.name` | `gr00t` | resolves to `GR00TRolloutRob` (string-path registry) |
| `actor_rollout_ref.rollout.mode` | `async_envloop` | env-loop pipeline |
| `env.train.simulator_type` | `arena` | → `IsaacLabArenaEnv` branch |
| `env.train.device` | `cuda` | **Arena requires GPU env workers** |
| `+env.train.gr00t_model_path` | `$SFT_MODEL_PATH` | GR1 export dir |
| `+env.train.embodiment_tag` | `gr1` | |
| `+env.train.arena_env_name` | `put_item_in_fridge_and_close_door` | |
| `+env.train.arena_object` | `ranch_dressing_hope_robolab` | matches the SFT checkpoint |
| `+env.train.arena_embodiment` | `gr1_joint` | joint-position control (`gr1_pink`=IK is incompatible) |
| `+env.train.kitchen_style` | `2` | |
| `+env.train.rl_success_reward` | `True` | success→reward, autoreset disabled |
| `env.actor.model.num_action_chunks` / `.action_dim` | `$NUM_ACTION_CHUNKS` / `26` | env-facing chunk |
| `actor_rollout_ref.model.path` | `$SFT_MODEL_PATH` | |
| `+actor_rollout_ref.model.embodiment_tag` / `.num_action_chunks` / `.action_dim` | `gr1` / `$NUM_ACTION_CHUNKS` / `26` | rollout reads these |
| `override_config.policy_type` | `gr00t` | |
| `override_config.sac_enable` | `True` | |
| `override_config.critic_head_num` | `10` | |
| `+override_config.critic_action_horizon` | `$NUM_ACTION_CHUNKS` | `≤ num_action_chunks` (#2) |
| `+override_config.action_dim` / `.embodiment_id` | `26` / `20` | |
| `+override_config.sac_action_train_dims` | `[[7,14],[20,26]]` | two-pass action mask |
| `override_config.flow_sde_enable` (+ `flow_sde_*`) | `True` (+ noise/β knobs) | Flow-SDE exploration |
| `actor_rollout_ref.actor.grad_clip` | `1` | verl-base actor field (also used by libero scripts) |
| `actor_rollout_ref.actor.sac.{gamma,tau,initial_alpha,critic/actor_replay_positive_sample_ratio}` | … | target SAC field names |

> **Source→target field gaps.** Two source fields were initially NOT carried over:
> `sac.bc_loss_coef` and an `actor.num_images_in_input` override.
> * `sac.bc_loss_coef` — **RESOLVED in Phase 6 (P1-1, §11.1).** A fixed-coefficient `bc_loss_coef`
>   field (default **0.0**) was added to `SACConfig` (dataclass + `actor.yaml`) and a matching path
>   in `training_worker._forward_actor`; the arena run scripts now set `...sac.bc_loss_coef=0.05`
>   (= source recipe). Default 0.0 keeps pi05/libero byte-for-byte unchanged.
> * `actor.num_images_in_input` — still **dropped**: it is a *model* field hardcoded to 1 for the rob
>   rollout (not an `actor.*` field, and unused for gr00t), so it would be rejected by struct mode.

### C. Hydra compose dry-run

`python -m verl_vla.trainer.main_sac <all overrides> --cfg job` requires the full
torch/ray/verl/hydra stack (Docker-only). On the CPU host this was reproduced with a **compose-only**
driver: a copy of `trainer/config` whose searchpath points (`file://`) at a reachable verl checkout
(`reb_verl`) as a **proxy** for the target upstream verl, then `hydra.compose(config_name=
"rob_sac_trainer", overrides=<captured from the run script>)`.

**Result: COMPOSE OK** for both `run_gr00t_arena_sac.sh` and `run_gr00t_arena_sac_disagg.sh` — every
Phase-5 field parses, `rollout.name=gr00t` resolves, the defaults tree assembles, and the invariant
`critic_action_horizon(16) ≤ num_action_chunks(16) ≤ action_horizon(50)` with
`num_action_chunks == model.num_action_chunks` holds. The only fields the **proxy** forced (they are
genuine target-verl fields the proxy checkout diverges on, used identically by the working libero
scripts) were `actor.grad_clip` (proxy lacks it) and, for disagg, `trainer.n_rollout_gpus_per_node`
/ `trainer.rollout_interval` (proxy pre-defines them so the target's `+`-append collides). These are
**proxy artifacts, not script bugs** — the in-container `--cfg job` against the real verl needs none
of the forcing.

### D. CPU validation summary

- `prepare_arena_dataset.py` → parquet written ✅
- compose dry-run (proxy) → COMPOSE OK, both scripts ✅
- gr00t-related unit tests: `tests/models/gr00t tests/workers/rollout/naive_rollout_gr00t_test.py
  tests/envs/arena_env tests/env_loop/test_gr00t_transition_prefix_chain.py` → **40 passed, 1
  skipped** (was 37+1; +3 new `#2` invariant tests; the P0 regression test included) ✅
- `ruff check --line-length 120` clean on all touched files ✅

### E. CPU vs Docker-only boundary

CPU-verified: config parsing/assembly, dataset generation, all stubbed unit logic (decode
**order/shape/semantics**, #2 guard, prefix chain). Docker-only (Phase 6): real Arena build + step,
real `GR00TN16Adapter` decode **numerics**, FSDP load of the checkpoint, the live
`env_worker ⇄ env_loop ⇄ trainer` loop, and the numerical proof that the §8.J fix matches the source.

## 10. Phase 6 — completed (see §11 for the detailed record)

`eval_arena_gr00t.py` + `smoke_test_gr00t_arena.py` migrated to `examples/arena_sac/`; the 4 P1
review items closed; docker-only verification gates itemised. Full record in §11.

---

## 11. Phase 6 — P1 close-out, eval/smoke, docker gates

### 11.1 P1-1 — `bc_loss_coef` (restore the source BC anchor)

**Problem.** Source `sac/sac_actor.py` uses a **fixed** coefficient
`actor_loss = sac_loss + bc_loss_coef * bc_loss` (arena recipe `bc_loss_coef=0.05`). The target
`training_worker.py` only had the adaptive **TD3+BC** weight (`td3_bc_weight*bc_loss`), gated by
`td3_enabled` (default false, never set in the run scripts) → BC was completely OFF.

**Fix (files changed).**
- `src/verl_vla/workers/config/actor.py` — `SACConfig` gains `bc_loss_coef: float = 0.0`
  (+ a `>= 0` validator). **Default 0.0** is the linchpin: it means the BC anchor is OFF unless
  explicitly turned on, so existing pi05/libero runs are unchanged.
- `src/verl_vla/trainer/config/actor/actor.yaml` — matching `sac.bc_loss_coef: 0.0` field
  (struct-mode safe).
- `src/verl_vla/workers/engine/sac/training_worker.py` — reads
  `self.bc_loss_coef = float(self.sac_config.get("bc_loss_coef", 0.0))`; `_forward_actor` now has a
  third branch. **Precedence (mutually exclusive):** `td3_enabled` → adaptive TD3+BC; **elif**
  `bc_loss_coef > 0` → fixed-coef `sac_loss + bc_loss_coef * model.bc_loss(...)`; **else** pure SAC.
  `bc_loss` is computed **lazily** (only in the branch taken) so the pure-SAC path performs zero
  extra work. Metric `sac/bc_loss_coef` logged.
- `examples/arena_sac/run_gr00t_arena_sac{,_disagg}.sh` — set
  `actor_rollout_ref.actor.sac.bc_loss_coef=0.05` (= source recipe).

**TD3 relationship.** Mutually exclusive, TD3 wins (documented in code + here). Both call the same
`model.bc_loss(obs, tokenizer, actions, valids)` (Phase 2); they only differ in the weight.

**pi05 zero-impact proof.** Default `0.0` ⇒ the `elif bc_loss_coef > 0` branch is never entered, no
`bc_loss` forward is run, `actor_loss == sac_loss` exactly as before. Verified by
`tests/workers/engine/sac/training_worker_bc_test.py::test_default_coef_zero_is_pure_sac_no_bc_call`
(asserts `bc_loss` is **not** called and `actor_loss == sac_loss`). The `>0` and `td3` branches are
covered by two more tests.

### 11.2 P1-2 — action_horizon guard upper bound (was 16, real ckpt = 50)

**Problem.** `assert_action_horizon_invariant` fell back to `GR00TDim.ACTION_HORIZON=16` for the
upper bound, but the checkpoint's real `action_horizon=50`. It passed only by coincidence
(`num_action_chunks=16==16`) and would falsely reject any chunk in `(16, 50]`.

**Fix (files changed).**
- `examples/arena_sac/run_gr00t_arena_sac{,_disagg}.sh` — add
  `+actor_rollout_ref.model.override_config.action_horizon=50` so the guard reads the model's real
  horizon (the guard already prefers `override_config.action_horizon`, falling back to the enum).
- `src/verl_vla/models/gr00t/utils.py` — comment on `GR00TDim.ACTION_HORIZON` clarifying it is an
  **env-side chunk-count fallback, NOT the checkpoint horizon** (which is 50, from `config.json` /
  `override_config.action_horizon`), and that run scripts MUST set the override.

### 11.3 P1-3 — config drift vs source `rob_sac_trainer_arena_gr00t.yaml`

| Field | Source value | Target before | Action |
|---|---|---|---|
| `env.train.step_penalty` | `0.001` (yaml, **as of source `09b0f07`**; was `0.0` at migration time) | env yaml default `0.001` | run scripts now set `env.train.step_penalty=0.001` (declared key, no `+`). **Corrected** — see §12 |
| `env.train.env_spacing` | `10.0` (yaml) | arena default `30.0` | run scripts now set `+env.train.env_spacing=10.0` (undeclared key, `+`) |
| `env.train.max_episode_steps` | yaml `1200`; **source run script overrides to `512`** | `512` | **kept 512** — matches the source *run script* (the actual runtime value); the yaml `1200` is a never-used default. Documented, not changed. |
| `env.rollout.pipeline_stage_num` | yaml `1`; **source run script overrides to `2`** | `2` | **kept 2** — matches the source *run script*; yaml `1` is a never-used default. |
| `action_horizon` guard | ckpt `50` | enum fallback `16` | P1-2 (`+...override_config.action_horizon=50`). |

> Note on 512/2: the task brief cites "源 1200/1" (the **yaml** defaults). The source **run script**
> (`run_gr00t_arena_sac.sh`) overrides them to `MAX_EPISODE_STEPS=512` / `NUM_STAGE=2`, which is what
> actually runs. The target therefore matches the source *runtime*, not the unused yaml defaults — an
> intentional alignment, not a drift.

### 11.4 P1-4 — real compose re-check (docker-only)

The CPU `reb_verl` proxy compose (§9.C, re-run in Phase 6 — see §11.D) validates field
**parsing/assembly** but is a proxy, not the real verl. **Mandatory before the first docker run**
(itemised in both run-script headers and §11.C gate 0):

```
python -m verl_vla.trainer.main_sac --config-name rob_sac_trainer_arena_gr00t \
    <all overrides from the run script> --cfg job
```

Confirm fields **without** a leading `+` already exist in the merged config (no
"Could not override … use +" error) — in particular `actor_rollout_ref.actor.grad_clip=1` and
`actor_rollout_ref.actor.sac.bc_loss_coef=0.05`.

### 11.B eval / smoke migration

- `examples/arena_sac/eval_arena_gr00t.py` — closed-loop Arena eval. `--actor sac` is the verl-vla
  rollout action path (NORMALISED chunk → `env.chunk_step`, env decodes once with a fixed base);
  `--actor gr00tpolicy` is the official-policy reference (ABSOLUTE joints → per-step `env.step`,
  which expects already-decoded 26-DOF). Reuses the training anchors
  (`override_config` / `embodiment_tag` / `num_action_chunks` / `action_horizon=50`).
- `examples/arena_sac/smoke_test_gr00t_arena.py` — in-process Workflow-A smoke (no simulator):
  build_inputs → sac_sample_actions → decode_actions_flat → sac_get_critic_value →
  sac_forward_state_features → sac_forward_actor → sac_forward_critic → bc_loss + backward
  (`8/8 PASS`). `--compare-gr00t-policy` adds the official-policy cross-check.
- Both keep the **eagle / transformers-4.51.3 / cuDNN** patch path: `_patch_transformers_eagle`
  delegates to the Phase-2 `modeling_gr00t_sac._patch_eagle_compat` (or
  `isaaclab_arena_gr00t.utils.eagle_config_compat` inside the image). Every isaac/omni/transformers/
  gr00t import is **function-local**, so the files import on a gr00t-free CPU host.
- CPU test: `tests/examples/arena_sac/eval_smoke_cpu_test.py` (import + argparse anchors).

### 11.C Docker-only verification gates (MANDATORY before first formal training)

> None of these are runnable on the CPU dev host (no real verl / Isaac Sim / gr00t / GPU). Build the
> image first (`_isaaclab_arena/docker/run_docker.sh -g`, image `isaaclab_arena:cuda_gr00t_gn16`),
> prepend `PYTHONPATH=/opt/groot_deps:<verl-vla/src>` (transformers 4.51.3 must win — see source
> README "Mandatory environment").

**Gate 0 — real `--cfg job` compose (P1-4).** As §11.4. Replaces the CPU proxy. Must pass before any
run.

**Gate 1 — #1 numerical decode equivalence (HARD GATE).** With the real checkpoint + real
`GR00TN16Adapter`, for one `(chunk_actions, chunk_start_state)`:
  - assert the env `chunk_step` `decoded_chunk` (`_decode_chunk_to_policy_actions`) equals the
    source-style one-shot `decode_actions_flat(full_chunk, base_groups)` **element-wise**
    (`atol=1e-5`); and
  - perturb `env._last_state26` **after** the chunk-start capture and assert the decoded values are
    **unchanged** (proves the base is fixed at chunk entry — the §8.J P0 fix). The `smoke` script's
    steps 2+3 already exercise sample→decode on real weights; extend with the two asserts above.

**Gate 2 — end-to-end env_worker ⇄ env_loop ⇄ trainer.** Run 1–2 `rollout_interval`s (temporarily
shrink `total_epochs` / `num_envs` / `num_action_chunks`; `trainer.save_freq=-1`,
`trainer.val_before_train=False`) via `run_gr00t_arena_sac.sh`. After one trajectory assert the
replay carries `t0.obs.images`, `t0.action.action` (normalised `(·,16,128)`),
`feedback.terminations`, and correctly-shaped `info.*`.

**Gate 3 — FSDP2 forward.** Confirm `sac_sample_actions` / `sac_get_critic_value` / `bc_loss` run
under FSDP2 without DTensor (`mixed Tensor and DTensor`) errors, and that
`actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap` covers every `_denoise` action-head
submodule: `Qwen3DecoderLayer`, `Siglip2EncoderLayer`, `BasicTransformerBlock`,
`MultiEmbodimentActionEncoder`, `CategorySpecificMLP` (already set in both run scripts).

**Gate 4 — docker smoke.** `smoke_test_gr00t_arena.py --ckpt … --num-envs 2 --denoise-steps 2`
→ expect `8/8 PASS`; then `eval_arena_gr00t.py --reset-only` (scene/asset), then a short
`--actor sac` closed-loop eval.

### 11.D Phase-6 CPU re-validation results

- **Unit tests** (full suite, this host): **62 passed**, plus 2 pre-existing failures + 2 errors that
  are **environment-only** and unrelated to this work — `tests/workers/rollout/test_weight_sync.py`
  needs `pytest-asyncio`/`anyio` plugins and `tests/workers/rollout/test_sample_transfer.py` needs
  `tensordict` (neither installed on the CPU host). The 54-test gr00t/arena baseline is unchanged;
  +8 new (3 BC + 5 eval/smoke). No pi05/libero behavior change.
- **Hydra compose dry-run** (proxy, both run scripts): **COMPOSE OK**. The new fields resolve to
  `actor.sac.bc_loss_coef=0.05`, `env.train.step_penalty=0.0`, `env.train.env_spacing=10.0`,
  `override.action_horizon=50`; invariant `16 ≤ 16 ≤ 50` holds. Only proxy-artifact forcings
  (`actor.grad_clip`; disagg `trainer.n_rollout_gpus_per_node` / `trainer.rollout_interval`) —
  identical to Phase 5, **not** script bugs; the real `--cfg job` (Gate 0) needs none.
- **ruff** (`line-length=120`): clean on all touched/new files.

### 11.E CPU vs docker-only test matrix

| Concern | CPU (here) | Docker-only (gate) |
|---|---|---|
| Config parse/assemble (incl. new P1 fields) | ✅ proxy compose | Gate 0 (`--cfg job`) |
| `bc_loss_coef` branch selection / pi05 zero-impact | ✅ mocked unit test | covered by training |
| action_horizon guard bounds | ✅ pure-helper tests | — |
| eval/smoke import + argparse anchors | ✅ | — |
| eval/smoke real model forward (`8/8 PASS`) | ✗ | Gate 4 |
| #1 decode numerical equivalence + fixed base | ✗ | **Gate 1 (hard)** |
| env_worker⇄env_loop⇄trainer trajectory shapes | ✗ | Gate 2 |
| FSDP2 forward / wrap-policy coverage | ✗ | Gate 3 |

### 11.F Differences vs source recipe (final)

| Item | Source | Target | Status |
|---|---|---|---|
| BC anchor | fixed `bc_loss_coef=0.05` | fixed `bc_loss_coef` (default 0.0; run scripts 0.05) + TD3+BC available | **aligned** (P1-1) |
| `step_penalty` | 0.001 (source `09b0f07`) | 0.001 (run scripts) | **corrected** (§12) — was 0.0 vs pre-`09b0f07` source |
| `env_spacing` | 10.0 | 10.0 (run scripts) | aligned (P1-3) |
| `max_episode_steps` | run script 512 (yaml 1200) | 512 | aligned to run script (P1-3) |
| `pipeline_stage_num` | run script 2 (yaml 1) | 2 | aligned to run script (P1-3) |
| action_horizon guard | ckpt 50 | `+override_config.action_horizon=50` | aligned (P1-2) |

### 11.G Key inputs for the reviewer's final README

- **Target repo / branch:** `/project/VLA_RL/verl_isaac_project/verl-vla/` · `migrate/gr00t-arena-sac`
  (package `src/verl_vla/`, prefix `verl_vla.*`).
- **Source:** `/project/VLA_RL/verl_isaac_project/libero_rl_example/verl/verl/experimental/vla/`.
- **Checkpoint:** `checkpoints/gr1_ranch_bottle_into_fridge/checkpoint-5000-export`
  (`action_horizon=50`, `use_relative_action=true`, `embodiment_id=20`).
- **Image build entry:** `_isaaclab_arena/docker/run_docker.sh -g` → `isaaclab_arena:cuda_gr00t_gn16`;
  `PYTHONPATH=/opt/groot_deps:<verl-vla/src>` (transformers 4.51.3 must win).
- **Run scripts:** `examples/arena_sac/run_gr00t_arena_sac.sh` (single-node) /
  `run_gr00t_arena_sac_disagg.sh` (separate sim/train GPU pools).
- **Dataset:** `python scripts/prepare_arena_dataset.py --num_train <BATCH_SIZE>` → train/val parquet.
- **Eval / smoke:** `examples/arena_sac/eval_arena_gr00t.py`, `examples/arena_sac/smoke_test_gr00t_arena.py`.
- **Mandatory gates before first formal training:** §11.C Gates 0–4 (Gate 1 numerical equivalence is
  the hard gate).
- **Entry module:** `python -m verl_vla.trainer.main_sac`.

## 12. Source commit `09b0f07` sync (post-migration regression points)

> **Why this section exists.** Phase 2 migrated `modeling_gr00t_sac.py` *before* source commit
> `09b0f07e053b5a47196e6a212dd6543263ea9521` ("freeze the vision tower; adds cross_atten_pooling
> for critic heads"). That commit fixed GR00T policy bugs **after** the migration snapshot, so the
> migrated model carried the same bugs. This is the back-port. Every model change is **config-gated
> with a legacy default**, so `pi05` / `libero` / the prior gr00t path are bit-unchanged; the Arena
> recipe opts in via the run-script `override_config` overrides.

### 12.1 `modeling_gr00t_sac.py` (`src/verl_vla/models/gr00t/`)

| Fix | Source `09b0f07` | Target back-port | Gate / default |
|---|---|---|---|
| **Freeze vision tower** | `freeze_vision_tower()` freezes `backbone.eagle_model.vision_model` (SiglipVisionModel) + `mlp1` connector (`requires_grad_(False)` + `.eval()`), called in `__init__` pre-FSDP, default `True` | identical method + pre-FSDP call. **Model default `False`** (legacy: vision follows backbone `tune_visual`) to protect existing paths; Arena run scripts set `freeze_vision_tower=True` | config-gated; defensive no-op if `eagle_model` absent |
| **Critic cross-attention pool** | `critic_pooling="attn"`: learnable `critic_state_token` query + `critic_prefix_cross_attn` (`nn.MultiheadAttention`) over VL tokens, frozen `target_*` copies, Polyak-tracked | same, behind `critic_pooling`. **Default `mean`** = the param-free masked mean-pool already in `sac_forward_state_features`. Output dim == `backbone_feature_dim` ⇒ `critic_input_dim` unchanged | config-gated |
| **Encoded-state critic input** | `critic_use_encoded_state=True`: critic uses the actor's dense `state_features` (state_encoder output) instead of raw padded `state`; changes `critic_input_dim` | same, behind `critic_use_encoded_state`. **Default `False`** (raw state). `critic_input_dim` recomputed in `__init__` from `_state_feature_dim` (=`action_head.input_embedding_dim`) | config-gated; **incompatible ckpt** when toggled |
| **`_init_weights` MHA in-proj** | xavier_uniform_ on `in_proj_weight`, zeros_ on `in_proj_bias` for `nn.MultiheadAttention` (avoids NaN under meta-device fast-init) | identical, appended to the existing Linear branch | always on (only matches MHA modules, i.e. only when `attn` pool exists) |
| **Param/Polyak/grad plumbing** | `sac_get_critic_parameters` (+cross_attn +state_token), `sac_update_target_network` (Polyak `lerp_` on attn + token), `sac_forward_critic` (toggle attn `requires_grad`, pass `use_target_network` to `_critic_input`), `_cross_attention_pool`, `_critic_input(use_target_network)` | all back-ported, **keeping the target interface**: action key `a["action"]` (not source's `a["full_action"]`), already-aligned `sac_forward_critic(self, a, state_features, task_ids=None, *, use_target_network=False, method="cat", requires_grad=False)`, None-free `state_features` contract | gated by `critic_pooling=="attn"` |

**`sac_forward_state_features` already None-free + complete.** The migrated state-features dict already
emits `pooled` + `backbone_features` + `backbone_attention_mask` + `state_features` + `state`, so the
cross-attn pool (needs `backbone_features`/`backbone_attention_mask`) and encoded-state path (needs
`state_features`) work without changing the producer — no contract change was required.

**FSDP2 wrap-policy (analysis, docker-only to confirm — Gate 3).** The new `critic_prefix_cross_attn`
(`nn.MultiheadAttention`) and `critic_state_token` are **root-level** params, exactly like the existing
`critic_heads` (`CriticMLP` ensemble) — none are listed in
`actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap`. Under FSDP2 they are sharded with the
root module and all-gathered when the registered `sac_forward_critic` runs, so **no wrap-policy entry is
needed** (the critic heads already prove this pattern). Residual risk: `nn.MultiheadAttention`'s
`F.multi_head_attention_forward` reads `in_proj_weight` directly — if it were ever left as a DTensor at
forward time it would raise "mixed Tensor and DTensor"; this cannot be exercised on the CPU host and is
folded into **Gate 3** (FSDP2 forward / DTensor check).

### 12.2 `use_critic = False` — already present (no change needed)

Source `09b0f07` added `self.use_critic = False` to `RobRaySACTrainer.init_workers` (SAC is a combined
actor-critic; the Q-head lives in the actor worker and is saved/loaded with the actor ckpt, so the
base-class `critic_wg` save/load/profile paths must be disabled). **The verl-vla migration already does
this** — both `src/verl_vla/trainer/sac/sac_ray_trainer.py` and `sac_separate_trainer.py` set
`self.use_critic = False` in `__init__` (alongside `use_reference_policy=False` / `use_rm=False`). This is
the same effect, set earlier and in a cleaner location, so **no port was required** (N/A).

### 12.3 Run scripts (`examples/arena_sac/`)

| Knob | `run_gr00t_arena_sac.sh` (single-node) | `run_gr00t_arena_sac_disagg.sh` |
|---|---|---|
| `env.train.step_penalty` | `0.0 → 0.001` | `0.0 → 0.001` |
| `+override_config.freeze_vision_tower` | added `=True` | added `=True` |
| `+override_config.critic_pooling` | added `=attn` | added `=attn` |
| `+override_config.critic_use_encoded_state` | added `=True` | added `=True` |
| `override_config.critic_prefix_attn_heads` | added `=8` (no `+`; key pre-exists in `rob_sac_trainer.yaml`) | same |
| `MICRO_BATCH_SIZE` | **unchanged** `16` (source left the single-node script alone) | `16 → 32` |
| `FLOW_SDE_NOISE_LEVEL` | **unchanged** `0.065` | `0.0... → 0.02` |

Override-prefix correctness was proxy-checked against `rob_sac_trainer.yaml`'s `override_config` (the
default `main_sac` config): `critic_pooling` / `critic_use_encoded_state` / `freeze_vision_tower` are new
keys ⇒ `+`; `critic_prefix_attn_heads` already exists in the default `override_config` ⇒ no `+`. The real
`--cfg job` compose remains **Gate 0** (docker-only).

### 12.4 step_penalty correction (supersedes §11.3 / §11.F)

Earlier migration notes recorded `step_penalty` as source `0.0` → target `0.0` (P1-3), based on the
pre-`09b0f07` source. **Source `09b0f07` changed the yaml `0.0 → 0.001`** (per-step time penalty giving
the critic a dense signal, à la LIBERO). The target is updated to **`0.001`** in both run scripts and the
§11.3 / §11.F tables are corrected accordingly. The prior `0.0` entries are obsolete.

### 12.5 SAC tuning hyperparameters — synced to source (user decision)

Source `09b0f07` also retuned a bundle of SAC hyperparameters in
`rob_sac_trainer_arena_gr00t.yaml`. These were initially recorded as out-of-scope tuning, but the
**user has ruled to fully align them to the latest source Arena recipe**. All 7 are now set in both
verl-vla run scripts (`run_gr00t_arena_sac{,_disagg}.sh`) as `actor.*` overrides — every key is a
declared dataclass field (`SACConfig` / `ActorConfig`) present in `config/actor/actor.yaml`, so **none
need a `+` prefix**. Source carries them in the shared yaml, so the single-node and disagg values are
identical (no per-script divergence for these 7).

| Param | Override path | verl-vla before | **Now (= source)** | Semantic effect |
|---|---|---|---|---|
| `tau` | `actor.sac.tau` | 0.005 | **1.0** | Target-network update coeff. `1.0` = **hard copy** of the online critic into the target every update (no target lag). NB: `update_target_network` is called every `update_policy`, so this reproduces the old hard-copy behaviour; raises Q-overestimation/divergence risk vs the 0.005 soft default — intended by the source recipe. |
| `bc_loss_coef` | `actor.sac.bc_loss_coef` | 0.05 | **0.0** | Fixed-coefficient BC anchor weight. `0.0` = **BC anchor OFF** → pure SAC actor loss (`actor_loss == sac_loss`). The Phase-6 P1 BC mechanism (field + 3-branch logic in `training_worker`) is **retained**; only the Arena value reverts to 0.0. With `0.0` the `elif bc_loss_coef > 0` branch is never entered and `model.bc_loss` is **not called** (covered by `training_worker_bc_test.py`). |
| `initial_alpha` | `actor.sac.initial_alpha` | 0.05 | **0.01** | Initial (fixed) entropy temperature for max-ent SAC. Lower ⇒ less entropy pressure on the actor / softer target. |
| `actor_replay_positive_sample_ratio` | `actor.sac.actor_replay_positive_sample_ratio` | 0.5 | **0.8** | Positive-sample ratio when drawing replay for **actor** updates. `0.8` trains the policy on a more success-heavy state distribution than the critic (which stays `critic_replay_positive_sample_ratio=0.5`). |
| `alpha_type` | `actor.sac.alpha_type` | softplus | **exp** | Parameterisation of the entropy temperature `alpha`. `exp` ⇒ `alpha = raw_alpha.exp()`. **Confirmed supported** by the target (`SACConfig` default is already `exp`; validator allows `{exp, softplus}`; `training_worker` implements both). |
| `replay_pool_save_interval` | `actor.replay_pool_save_interval` | (newly set) | **500** | Persist replay-buffer snapshots every 500 steps (target dataclass default is already 500; now set explicitly). |
| `replay_pool_single_size` | `actor.replay_pool_single_size` | 6000 | **2000** | Per-(task, pos/neg) rank-local replay capacity. `2000` makes the buffer **shallower ≈ near on-policy** (less SAC/RLPD sample reuse) vs the deeper 6000. RAM-bound; source's chosen depth. |

> `critic_replay_positive_sample_ratio` stays **0.5** (source unchanged).

**Why this matters together.** `tau=1.0` (hard target copy) + `bc_loss_coef=0.0` (no BC anchor) +
`replay_pool_single_size=2000` (near on-policy) make the Arena recipe a leaner, more on-policy SAC than
the earlier verl-vla defaults; combined with the §12.1 critic-representation fix (attn pool + encoded
state) this is the source's converged Arena configuration.

### 12.6 Tests + validation

- `tests/models/gr00t/modeling_gr00t_sac_test.py` extended with 9 CPU mock tests:
  `attn` pool output shape + target init parity, `sac_get_critic_parameters` includes attn + state_token,
  Polyak updates attn + token (incl. `tau=1.0` hard copy), `sac_forward_critic` target-pool path,
  `critic_use_encoded_state` input-width change, `_init_weights` MHA in-proj no-NaN, `freeze_vision_tower`
  sets `requires_grad=False` + eval (mock eagle) and is a no-op when absent, and a **legacy-default
  regression** test (default `mean` ⇒ no attn params, mean-pool input width).
- Full suite (CPU host): **71 passed** (prior 62 baseline + 9 new). The only failures are the documented
  environment-only ones in `tests/workers/rollout/` (`tensordict` / `pytest-asyncio`/`anyio` /
  `verl` not installed on the CPU host) — unrelated to this work. No `pi05` / `libero` behaviour change.
- Override-prefix proxy compose: **OK** for both run scripts. **ruff (line-length=120): clean.**

### 12.7 cross-attn NaN guard + single-node FLOW_SDE drift (source follow-on)

Two more items synced from the source `09b0f07` work — the first is a **source working-tree fix not yet
committed** (forward-ported here proactively), the second a leftover migration drift.

**(a) cross-attention pool NaN guard (`modeling_gr00t_sac.py`).** The `_cross_attention_pool` added in
§12.1 passed the raw `vl_embeds` as MHA key/value. Padded VL token positions can carry `NaN`/`inf`;
`key_padding_mask` zeros their *softmax weight*, but `nn.MultiheadAttention` still computes the **value
projection** of every position, and `0 * NaN = NaN` propagates into the pooled output → **NaN critic**.
Fix (mirrors the masked mean-pool guard already in `sac_forward_state_features`): zero the padded
positions **before** the MHA call —

```python
mask_b = attn_mask.unsqueeze(-1).to(torch.bool)
vl_safe = torch.where(mask_b, vl_embeds, torch.zeros_like(vl_embeds))
pooled, _ = cross_attn(query=query, key=vl_safe, value=vl_safe,
                       key_padding_mask=key_padding_mask, need_weights=False)
```

Plus an **env-gated** NaN debug hook in `_critic_input` (`os.environ.get("GR00T_SAC_NAN_DEBUG")`): when
set, it `logger.error`s the per-tensor non-finite count for `pooled` / `state` / `action` with context
(`pooling` / `encoded_state` / `target` / `state_token_finite`). `import os` added. **Both are
attn-path-only** — the guard runs inside `_cross_attention_pool` (only built when `critic_pooling=="attn"`)
and the debug block is a no-op when the env var is unset (zero overhead). The default mean-pool path is
byte-unchanged. (Target keeps the `a["action"]` key: the debug block references the existing local `act`,
not the source's `full_action`.)

**(b) single-node `FLOW_SDE_NOISE_LEVEL` drift (`run_gr00t_arena_sac.sh`).** Source single-node has
`FLOW_SDE_NOISE_LEVEL=0.02` (since `09b0f07`); the verl-vla single-node script still carried a migration
leftover `0.065`. Corrected to **`0.02`** (the disagg script was already `0.02`, unchanged).

**Tests/validation (this round).** +2 CPU regression tests:
`test_cross_attention_pool_nan_guard_on_padding` (padded NaN/inf VL tokens → pooled finite; and the valid
positions alone reproduce the pooled vector, i.e. masked positions are truly excluded) and
`test_attn_critic_input_finite_with_padded_nan` (end-to-end `sac_forward_critic` → finite Q). The guard's
necessity is confirmed independently: a bare `nn.MultiheadAttention` with a padded-NaN key +
`key_padding_mask` yields a **non-finite** output without the `vl_safe` zeroing and a **finite** one with
it, so the test fails if the guard is removed. `tests/models/gr00t/`: **33 passed** (31 prior + 2).
ruff clean; single-node `bash -n` OK; FLOW_SDE proxy resolves `0.02`.
