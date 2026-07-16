# Episodic Replay Collection for SAC

`trainer.episodic_replay=True` switches the SAC data-collection path from
per-rollout transition masking to a streaming, per-env **episode collector**.
It fixes a systematic data-distribution problem — **near-termination bias** —
that arises from the interaction between auto-reset rollout windows and this
implementation's outcome-labelled dual replay pool (section 2.3): the pool
over-represents transitions close to episode termination and silently drops
the early and middle parts of episodes that straddle rollout windows.

```bash
# Requires auto-reset rollouts (episodes continue across rollout windows).
vvla-train-sac ... \
  cluster.env.env_worker.auto_reset=true \
  trainer.episodic_replay=True
```

The feature is off by default; with `episodic_replay=False` (or
`auto_reset=False`) the legacy masking path is used unchanged.

---

## 1. Background: rollout windows vs. episodes

Terminology used throughout (see also the chunk-level MDP notes):

| Symbol | Meaning |
| --- | --- |
| `S` | `env_loop.max_interactions` — macro-steps (action chunks) per rollout window |
| `C` | `action_chunk_steps` — primitive env steps per chunk |
| `S·C` | rollout-window horizon in primitive steps |
| `L` | episode length in primitive steps (up to `max_episode_steps`) |
| `B` | number of parallel env lanes after collation |

Training alternates between **rollout windows** (collect `[B, S]` macro-steps)
and update steps. With `auto_reset=True` the envs are *not* reset between
windows: `reset()` returns the cached observation of the previous step
(`envs/base.py`), so per batch row the windows form one physically continuous
stream, and episodes freely straddle window boundaries. IsaacLab performs the
per-episode auto reset internally whenever an episode terminates or times out.

Where the episode horizon `L` comes from differs per simulator — a frequent
source of confusion:

- the **native LIBERO** simulator (`simulator_type: libero`) reads
  `simulator.max_episode_steps` from the verl-vla config
  (`workflows/config/env/simulator/libero.yaml`);
- the **Arena** simulators (`arena_gr1`, `arena_libero`) *ignore* that key:
  IsaacLab owns the horizon via the Arena task's native `episode_length_s`
  (sim `time_out`). For Arena LIBERO that is
  `episode_length_s_for_suite` in
  `IsaacLab-Arena/isaaclab_arena_examples/external_environments/libero/franka_libero_rl_env_cfg.py`
  (10 s default, 26 s for `libero_10`) at 20 Hz control
  (`sim.dt = 1/60`, `decimation = 3`).

Whether episodes straddle windows is then a matter of arithmetic. Three
production examples:

- **Native LIBERO** (`examples/libero_recap/run_pi05_libero_recap.sh`,
  `auto_reset=true`): window = `8` chunks ≈ 80 primitive steps (10-step
  chunks) vs `max_episode_steps = 200` (script default; the yaml default is
  512). An episode spans 2–3 windows.
- **Arena GR1** (`examples/gr00t_arena_sac/run_gr00t_arena_sac.sh`):
  `S·C = 32 × 16 = 512` vs an episode horizon of ≈500 steps (10 s at 50 Hz).
  Window and episode have nearly equal length, so almost every episode
  crosses a boundary and loses its pre-boundary prefix.
- **Arena LIBERO** (same script, `ARENA_TASK=libero`):
  `S·C = 10 × 16 = 160` vs an episode horizon of ~120–200 control steps for
  the default suites (`episode_length_s = 10 s`; timeouts around ~120 steps
  are typical in practice). `L ≈ S·C`, so most episodes straddle one window
  boundary; for `libero_10` (26 s ≈ 520 steps) episodes span 3–4 windows and
  lose their early/middle parts entirely.

## 2. The problem: near-termination bias

### 2.1 Legacy collection path

The legacy SAC path (`prepare_sac_actor_input` in
`trainer/sac/sac_ray_trainer.py`) processes every rollout window
independently:

1. `_build_sac_transition_masks` (`utils/data.py`) splits each row into
   episode segments at `done = terminated | truncated`. Only transitions
   inside a complete `[segment_start, done]` segment get `valid=1`. The
   **residual after the last `done`** — and entire rows containing no `done`
   — get `valid=0`.
2. `add_transition_prefixes` + `flatten_trajectories` turn the `[B, S]` slots
   into flat `t0.*/t1.*` transitions.
3. `SACReplayPool.add_batch` (`utils/replay_pool.py`) drops every `valid=0`
   row at insertion.

This yields a simple inclusion criterion:

> **A transition enters the replay pool if and only if a `done` follows it
> within the same rollout window.**

### 2.2 Why the dropped data never comes back

Under `auto_reset=True` the next window continues from where the previous one
stopped. The residual transitions that were dropped are **not replayed** — the
env has already moved past them. For an episode spanning `k` windows:

```
episode timeline (primitive steps):  |============ L ============|
rollout windows (S·C each):          [ w1 ][ w2 ][ w3 (done here) ]
done inside this window?               no    no    yes
legacy valid mask:                    drop  drop   keep (start..done)
```

Only the final window's segment survives. The result is a replay pool that is
**systematically biased towards near-termination states**; episode-early
states may have no coverage at all. Consequences:

- The critic sees few or no samples of early-episode states, so Q-estimates
  at episode start (exactly where the policy must commit to a strategy) are
  unreliable. The longer the task horizon, the worse the effect.
- Success/failure classification for the dual (positive/negative) pool is
  only known when a `done` is observed, which the legacy path conflates with
  the window boundary.
- Observable symptom: `data/valid_ratio` (logged by the SAC training worker)
  is well below 1 — that fraction of collected experience is thrown away.

Note this is *more* general than `S·C < L`: even with `S·C ≥ L`, any episode
that merely *straddles* a boundary loses its pre-boundary part. Only episodes
that fall entirely inside a single window survive intact.

### 2.3 Root cause: pool admission depends on the episode outcome

It is worth being precise about *why* the legacy path drops data, because
auto-reset streaming itself is not the culprit. Ingesting transitions out of
auto-reset rollouts is standard off-policy practice, and for a **vanilla
single-pool SAC buffer** it is essentially lossless: a transition
`(s, a, r, s')` is admissible the moment its next observation is known. The
only window artifact would be the final macro-step of each lane — its true
`s'` arrives with the next window — which needs a one-slot carry (or a
one-transition drop), not a segment-wide mask. No near-termination bias would
result.

What makes whole residual segments inadmissible is this implementation's
**RLPD-style dual replay pool**. `SACReplayPool` routes every transition into
a positive or negative pool by the episode-level `info.success_mask`, and the
critic/actor updates sample the two pools at a fixed
`critic/actor_positive_sample_ratio` so that sparse successful experience
stays visible under sparse binary rewards. That routing label is a property
of the *whole episode* and is only defined once the episode's outcome is
known — i.e., once its `done` has been observed. The label is used purely for
pool routing and sampling ratios, never in the Bellman target, but every
transition still needs it *before* insertion.

The legacy vectorized masking resolves this dependency by construction: it
admits only transitions whose episode completed within the same window —
outcome known — and marks everything else `valid=0`. "Outcome not yet known"
is thereby silently converted into "never enters the pool", which is exactly
the bias of section 2.2.

The episodic collector keeps the same admission rule — no transition enters
the pool without its episode's outcome label — but satisfies it by
**buffering** the undecided residual until the outcome arrives (a later
`done`, or a forced truncation that assigns a conservative label), instead of
dropping it.

### 2.4 A second, silent continuity break: evaluation

Evaluation rollouts (`cluster.eval()`) reuse the training env workers. Eval
steps overwrite the cached `_latest_obs`, and eval resets are real resets. The
first training window after an eval therefore does **not** continue the
pre-eval episodes. The legacy path never noticed (it treats every window
independently); the episodic collector must treat eval as a hard flush
boundary (section 3.3).

## 3. The solution: streaming `EpisodeCollector`

`verl_vla/utils/episode_collector.py` implements the classic episodic-replay
collector shape: the collated `[B, S]` rollout output is consumed as `B`
continuous per-lane streams, one lane per physical env.

### 3.1 Data flow

```
env_loop (unchanged)          trainer driver                     actor workers (unchanged)
[B, S] collated rollout  ->   EpisodeCollector.ingest()     ->   replay pool add + SAC update
                              per lane:
                                append S slots to open buffer
                                flush on done / overflow
                              emit flat t0./t1./info.* batch
                              (all rows valid=1, padded to
                               actor world_size)
```

The collector lives on the trainer driver, inside `_prepare_actor_input`
(`trainer/sac/sac_ray_trainer.py`). It deliberately does **not** live in
`env_loop`, which is shared by PPO/eval and must stay stateless. The emitted
batch uses exactly the schema the SAC training worker already consumes
(`t0.obs.*`, `t1.obs.*`, `t0.action.*`, `t1.action.*`, `info.rewards`,
`info.terminateds`, `info.valids`, `info.success_mask`), so **the replay pool,
the training worker, and the env stack are untouched**.

### 3.2 Key insight: episodes flush the moment they finish

By the existing `add_transition_prefixes` boundary convention, the terminal
transition of an episode uses a **self-copied** `t1` observation (the real
next observation belongs to the next episode after auto reset and must not be
consumed). Therefore a complete episode needs *nothing from the future*: it
can be flushed into the replay pool at the exact macro-step its `done`
arrives. Only the residual segment after the last `done` must wait — and it
simply stays in the open buffer until later ingests deliver its next
observations. Across a window boundary, `t1.obs` of the last slot of window
`w` is the first slot of window `w+1` — physically continuous, no correction
step needed.

### 3.3 Flush rules

| Trigger | What is flushed | Terminal `t1` | `info.terminateds` at tail | Success label |
| --- | --- | --- | --- | --- |
| `done` slot ingested | whole open segment incl. the done slot | self-copy (boundary rule) | the slot's own `terminated` flag (truncation keeps bootstrapping) | `any(success)` over the segment |
| open buffer reaches `episodic_max_open_len` | all but the newest slot | the **retained** newest slot — a real next observation | `0` (treated as truncation; bootstrapping continues) | `any(success)` seen so far |
| continuity break (`force_flush_all`, called after every eval) | all but the newest slot, per lane; the newest slot is dropped | the dropped slot's observation | `0` | `any(success)` seen so far |

Design notes on the forced flushes:

- Flushing *all but the newest* slot guarantees every emitted transition
  bootstraps from a **real** next observation; only real episode ends use the
  self-copy approximation (same as the legacy path).
- On a continuity break the newest slot is dropped, not kept: its true next
  observation will never arrive, and chaining it to post-break data would
  fabricate a transition that never happened. Cost: at most one macro-step per
  lane per eval.
- A forced cut can mislabel the flushed prefix of an episode that later
  succeeds (it enters the negative pool). This only affects positive/negative
  pool routing, never the Bellman target, and is rare (overflow or eval
  boundaries only).

### 3.4 Contracts and failure modes

- **Lane stability.** Cross-window stitching relies on batch row `b` mapping
  to the same physical env in every window. The env-loop restructure/collate
  (`train_cluster/env_loop.py`) is deterministic given a fixed
  `world_size / stage_num / num_envs` topology, so this holds within a run.
  `ingest` raises if the lane count ever changes rather than silently
  splicing unrelated streams.
- **Checkpoint restart.** Open buffers are intentionally *not* persisted.
  After a restart the simulator starts fresh, so the buffered episodes'
  futures never arrive; an empty collector is the correct state. At most one
  open segment per lane is lost.
- **Async rollout.** With `trainer.async_rollout=True` the prefetched window
  may physically predate an eval. The eval-time flush is then conservative —
  it truncates a still-continuous episode early — which is safe (the emitted
  transitions are all real); it never corrupts data.
- **Variable batch size.** The number of flushed transitions varies per
  window, so the trainer pads the batch to a multiple of the actor world size
  with `valid=0` padding rows (`pad_dataproto_to_divisor_with_valid_mask`),
  the same mechanism the RLPD prefill path uses.

### 3.5 Why mixing policies inside one episode is fine (for SAC)

An episode spanning `k` windows contains actions from up to `k` policy
versions. This is a physical fact of auto-reset training, not something the
collector introduces — the legacy path generated the same mixed-policy
episodes and merely *deleted* the older segments. SAC consumes transitions
independently: the critic target `r + γ·Q(s', a'~π_current)` and the actor
loss (fresh actions sampled at pooled states) never reference the behavior
policy, and the pool already mixes transitions across thousands of updates
plus RLPD offline data. The only episode-level quantity is the success label
used for dual-pool routing (see 3.3).

This reasoning does **not** transfer to PPO, which needs behavior-policy
log-probs for importance ratios. The PPO workflow keeps `auto_reset=False`
and does not use the collector.

### 3.6 Memory

Open buffers hold at most `episodic_max_open_len` macro-steps per lane, and in
practice at most one episode (`≈ L/C` chunks) because flushing happens at
`done`. With the default GR00T Arena topology (8 lanes, one 480×640×3 camera
≈ 0.9 MB per macro-step) the worst case is
`8 × 128 × ~1 MB ≈ 1 GB` of driver CPU RAM, typically far less — the same
order as holding one extra rollout output, and roughly a tenth of a single
replay pool (capacity 2000, obs stored twice per transition). Observations are
stored once per slot; `t1` views are only materialized at flush time. Scale
`episodic_max_open_len` down if you scale `num_envs` up by orders of
magnitude.

## 4. Configuration reference

| Key | Default | Meaning |
| --- | --- | --- |
| `trainer.episodic_replay` | `False` | Enable the streaming collector. Requires `cluster.env.env_worker.auto_reset=true` (validated at startup). |
| `trainer.episodic_max_open_len` | `128` | Per-lane open-buffer cap in macro-steps before a forced truncation flush. Should comfortably exceed the episode horizon in chunks (`L / C`). |

Legacy behavior is fully preserved when the flag is off, and
`auto_reset=False` runs are unaffected (each window starts from a real reset,
so the per-window masking is already correct there).

## 5. Monitoring

Collector metrics are attached to the actor input and logged with the
training metrics:

| Metric | Healthy signal |
| --- | --- |
| `data/valid_ratio` | ≈ 1 (only world-size padding rows are invalid). Well below 1 means you are on the legacy path or something is wrong. |
| `data/collector_episodes_flushed` | Cumulative completed episodes; should grow every rollout window. |
| `data/collector_success_episodes_flushed` | Cumulative successful episodes (positive-pool routing). |
| `data/collector_forced_flushes` | Should stay near the eval count. Rapid growth means `episodic_max_open_len` is too small for the episode horizon. |
| `data/collector_transitions_emitted` | Cumulative transitions entering the pool. |
| `data/collector_slots_dropped` | Slots dropped at continuity breaks; grows by ≤ `B` per eval. |
| `data/collector_open_len_max` / `_mean` | Should stay below one episode length in chunks (`L / C`). |

### 5.1 Relationship to the legacy `collect_*` diagnostics

The legacy path emits `data/collect_valid_ratio`,
`data/collect_dropped_ratio`, `data/collect_rows_without_done_ratio`, and
`data/collect_{valid,total,dropped}_chunks` from `prepare_sac_actor_input`.
These measure structural loss *within a single `[B, S]` window*: the residual
after the last `done` and entire no-`done` rows are dropped, so the ratio is
meaningfully below 1 and quantifies the near-termination bias.

The collector deliberately does **not** expose analogous ratios, and this is
not an omission. Its whole point is that these transitions are never
dropped — the residual after the last `done` stays buffered and is emitted on
a later ingest once its next observation arrives. A window-local "valid
ratio" would therefore be ≈ 1 by construction and carry no signal. The
collector's true, rare loss (one newest slot per lane at a continuity break)
is already captured by `data/collector_slots_dropped`.

So there is intentionally no forced metric alignment between the two paths.
The conceptual correspondence is:

| Legacy diagnostic | Collector equivalent |
| --- | --- |
| `collect_dropped_ratio` (structural, per-window) | `collector_slots_dropped` — only continuity-break drops (≤ `B` per eval), otherwise ~0 |
| `collect_rows_without_done_ratio` | `collector_open_len_mean` / `_max` — no-`done` rows are *buffered*, not dropped, so they show up as buffer depth |
| `collect_valid_chunks` / `collect_total_chunks` | `collector_transitions_emitted` (cumulative) |

## 6. Implementation map and tests

| Piece | Location |
| --- | --- |
| Collector (all core logic) | `src/verl_vla/utils/episode_collector.py` |
| Trainer wiring, eval flush, padding | `src/verl_vla/trainer/sac/sac_ray_trainer.py` (`_prepare_actor_input`, `_flush_episode_collector`) |
| Config fields | `src/verl_vla/trainer/sac/config.py`, `src/verl_vla/workflows/config/trainer/sac_trainer.yaml` |
| Legacy path (kept as fallback) | `_build_sac_transition_masks` / `add_transition_prefixes` / `flatten_trajectories` in `src/verl_vla/utils/data.py` |
| Unit tests (CPU-only, no simulator) | `tests/utils/test_episode_collector.py` |

The unit tests pin down the two properties that matter most: for episodes
completing inside one window the collector's output is **numerically
identical** to the legacy path, and for cross-window episodes every
transition is recovered with correct `t1` continuity across the boundary —
exactly the data the legacy path drops.

```bash
python -m pytest tests/utils/test_episode_collector.py
```

## 7. Limitations and future work

- Open buffers are not checkpointed (deliberate, see 3.4); at most one open
  segment per lane is lost on restart.
- The eval-time flush is conservative under `async_rollout` (see 3.4). Exact
  boundary tracking would require the cluster to report whether the prefetched
  window predates the eval.
- The per-episode flush point is a natural place to add n-step returns,
  reward relabeling, or HER-style goal relabeling — the whole episode is
  available in one place before it enters the pool.
