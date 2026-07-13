# Tests

Unit tests mirror `src/verl_vla/...`. Prefer pure-Python cases that do **not** need
Isaac Sim / `gr00t` / a checkpoint, so they run on a laptop or in CI.

## Layout

| Path | What it covers |
|------|----------------|
| `tests/envs/arena/` | Embodiment adapters, camera/state extract, RL reward cfg patch |
| `tests/models/` | GR00T utils, Arena policy IO, SAC config helpers, critic action track |
| `tests/workers/` | Rollout / weight-sync (needs async fixtures; may fail without `pytest-asyncio`) |
| `tests/special_sanity/` | Import / license / structure checks |

### Arena env (`tests/envs/arena/`)

- **`test_embodiment.py`** — `ArenaJointMapping` gather/scatter + YAML index maps;
  G1 identity joint-space; GR1 lazy map; task-space (`eef_pose`) passthrough;
  `use_policy_action` / `policy_action_dim` from cfg.
- **`test_g1_parity.py`** — G1 camera uint8 conversion + state/action identity vs
  small reference helpers. Cameras are **strict**: missing configured names raise
  (no fallback / zero-image placeholder).
- **`test_rl_reward.py`** — `apply_arena_rl_reward` with fake `isaaclab` modules
  (no Isaac Sim).

### GR00T models (`tests/models/`)

- **`test_gr00t_utils.py`** — `GR1` / `EMBODIMENTS`, `split_flat_state_to_groups`,
  `load_embodiment_id` (checkpoint JSON vs fallback).
- **`test_gr00t_arena_policy.py`** — `ArenaGr00tInput` / `ArenaGr00tOutput`:
  image BHWC uint8; `from_env_obs` passes every `observation.images.*` through as a
  dict (no head/wrist name heuristics); action chunk / `full_action` / log_prob.
- **`test_gr00t_config.py`** — `cfg_get` / SAC override defaults (no gr00t package).
- **`test_gr00t_critic_action.py`** — normalised `full_action` survives replay
  plumbing (decoded env `action` stays separate).

## How to run

### Host (minimal deps)

```bash
# from repo root; needs torch + verl on PYTHONPATH
PYTHONPATH=src python -m pytest tests/envs/arena tests/models -q
```

Some `tests/models/*` import `verl_vla.models`, which currently pulls OpenVLA
registration (`timm` / matching `transformers`). Prefer the Docker path below if
the host stack is incomplete.

### Docker (recommended for this branch)

Use the running GR00T Arena image (repo mounted at `/eval`):

```bash
docker exec -w /eval isaaclab_arena-cuda_gr00t_gn16 bash -lc \
  'export PYTHONPATH=/eval/src; /isaac-sim/python.sh -m pytest tests/envs/arena tests/models -q'
```

Full suite:

```bash
docker exec -w /eval isaaclab_arena-cuda_gr00t_gn16 bash -lc \
  'export PYTHONPATH=/eval/src; /isaac-sim/python.sh -m pytest tests/ -q'
```

Expected for the Arena / GR00T unit set: all green. Unrelated
`tests/workers/rollout/test_weight_sync.py` may fail without `pytest-asyncio` /
fixtures — ignore for embodiment/policy changes.

## Design contracts the tests lock in

1. **Cameras** — env `camera_names` order ↔ policy `images` dict order ↔ checkpoint
   `video_keys` by **position** (not by `wrist` / `head` name heuristics).
2. **Joint-space** — G1 identity passthrough; GR1 mapping from YAML dir
   (`arena_joint_space_dir` only, no env/package discovery).
3. **Task-space** — action/state passthrough from concatenated `obs["policy"]`
   (checkpoint layout must match sim).
4. **SAC action double-track** — critic sees normalised `full_action`; env steps
   decoded `action`.
