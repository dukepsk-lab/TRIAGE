# Phase 0 Notes

What was built this session, the ambiguities I hit in
`docs/ARCHITECTURE.md`, and decisions I'm flagging for you before
Phase 1.

## What was built

Phase 0 scope only — the harness correctness layer. Nothing downstream
of the gate exists (rule engine, feature service, meta-model, sizer,
execution adapter are all empty stubs with docstrings pointing at the
architecture section they'll implement).

### `src/triage/labeling/` — triple-barrier resolver (§4)

- `resolve(event, m1_bars)` follows the §4 pseudocode **line-for-line**,
  including the named same-M1-bar → SL tie-break (a deliberately
  pessimistic, deterministic branch — `same_bar_ambiguity_long` pins it).
- Frozen barrier constants: `TP_MULT=1.5`, `SL_MULT=1.0`,
  `VERTICAL_BARS=24`, `ATR_PERIOD=14`. Hardcoded with a TODO pointing
  at `configs/rules/<rule>.yaml` (§3) for Phase 1; they stay
  pre-registered there, not free parameters.
- Label policy per §4: `label = 1` iff outcome = TP; SL and vertical → 0.
- `r_realized` (signed R-multiple) computed on every event regardless
  of label. TP → +1.5R, SL → −1.0R, vertical mark-to-market at the
  `t_vert` bar's close (so the §4/§8 vertical-as-zero ablation can be
  revisited later without losing the actual mark).

### `src/triage/validation/` — purge + embargo + CPCV (§7)

- `purged_train_idx(events, test_start, test_end, embargo_bars=48)`
  matches the §7 sketch exactly: keep iff `t1 < test_start` OR
  `t0 > test_end + emb`. Default embargo = 48 H4 bars = 2× vertical.
- `cpcv_splits(n_groups=6, k_test=2)` → exactly 15 splits. Each split
  purges + embargoes training events against the union span of its
  held-out test groups.
- `backtest_paths(n=6, k=2)` → exactly **5** backtest paths via a
  1-factorization of K₆ (each path is a perfect matching: 3 disjoint
  pairs partitioning {0..5}; all 15 pairs covered exactly once).
  This matches §7's "5 synthetic backtest paths".
- Uniqueness weights are **not** computed here yet — see flag #2 below.

### `src/triage/features/` — no-lookahead audit (§4/§5/§7)

- `no_lookahead(snapshot, t0)` is the real, assertable enforcement of
  the no-look-ahead invariant. It scans every field ending in
  `*_source_close_time` plus the aggregate `max_source_close_time`,
  and raises `LookaheadError` (subclass of `AssertionError`) if any
  source timestamp exceeds `t0`. Snapshots with no auditable timestamp
  field fail closed — the invariant cannot be verified → drop.
- `build_snapshot` / `compute_features` are NotImplementedError stubs
  (Phase 1, §5).

### `src/triage/registry/` — trial registry (§0/§2)

- SQLite store with the schema from the build prompt
  (`run_id, timestamp, config_hash, config_json, gate_results_json`)
  plus `status` / `error` so a fail/error trial is recorded with its
  reason rather than dropped.
- `@logged_trial(config)` decorator + `trial_context` context manager.
  Both write a row in a `finally` block, so pass/fail/error all log.
- `config_hash` is order-independent (sorted JSON); two runs with the
  same config hash collide, which is what makes a "trial" countable
  for the DSR/PBO gates (§0, §7).

### `tests/synthetic/paths.py` — synthetic M1 generators (§4 battery)

Six generators, each returning the path **and** its hand-derived
ground-truth outcome:

| Generator | Side | Expected outcome | label | r_realized |
|---|---|---|---|---|
| `linear_tp_long` | +1 | tp @ bar 3 | 1 | +1.5 |
| `linear_sl_long` | +1 | sl @ bar 2 | 0 | −1.0 |
| `flat_timeout` | +1 | vert @ t_vert | 0 | 0.0 |
| `same_bar_ambiguity_long` | +1 | sl @ bar 1 (tie-break) | 0 | −1.0 |
| `linear_tp_short` | −1 | tp @ bar 2 | 1 | +1.5 |
| `late_tp_long` | +1 | tp @ bar 5 | 1 | +1.5 |

The short-side case is the one that exercises the `side == -1` branch
of `resolve()` (TP via `low <= tp_px`, SL via `high >= sl_px`).

### `tests/unit/shuffled_label_harness.py` — leakage harness (§4)

Reusable `shuffled_label_auc(features, labels, ...)` and
`assert_no_leakage(...)`. Trains a trivial plain LightGBM on permuted
labels with a time-ordered split and asserts OOS AUC ≈ 0.50 within a
stated tolerance. Negative-control test confirms the machinery is
alive (a label-correlated feature gives AUC > 0.85 on real labels).

## Test results

```
60 passed in 1.27s
```

Covers: barrier constants, exact-match resolution on all 6 synthetic
paths, label policy, r_realized, same-bar tie-break (isolated),
determinism, hand-enumerated purge/embargo removal set, embargo
boundary strictness, CPCV 15-splits/5-paths, CPCV overlap-purge,
feature audit (clean + leaky + per-component-leak + fail-closed),
shuffled-label AUC≈0.5 (single + multi-seed), and registry
pass/fail/error logging.

## Ambiguities / things I had to resolve

These all have a defensible reading of the doc, but I want them on the
record before Phase 1.

1. **`flat_timeout` vertical mark-to-market.** §8 says "vertical
   conservatively valued at 0" for the EV formula, but §4 says
   `r_realized` is "mark-to-market for verticals" and stored on every
   event. I read these as compatible: the *event store* keeps the
   actual mark (so the §4 ablation "vertical → sign(r_realized)
   instead of 0" can reuse it), while the *EV/sizing* layer applies the
   conservative 0. My `flat_timeout` fixture happens to land r=0.0
   because its mark is exactly the entry; a non-flat timeout path
   would store a nonzero `r_realized` and size from 0. **Confirm this
   split is what you intended.**

2. **Uniqueness weights — per-fold, not global.** §4 is explicit
   ("weights are computed per fold on training events only") and the
   prompt calls this out as a leakage vector. I did **not** implement
   uniqueness weighting this session — it isn't a Phase 0 deliverable
   in the build prompt (it appears with the training pipeline in
   Phase 1), and shipping it without the training loop to consume it
   would be untested code. Flagging because the invariant
   ("never over the full event set") needs to be enforced *at the
   training call site*, not just by convention — I'll wire it as a
   function that takes `(train_events, train_idx)` and refuses the
   full index in Phase 1.

3. **`t1 == NaT` handling in purging.** §4's event schema has `t1` as
   nullable (events can be unresolved). §7's purge sketch
   (`events.t1 < test_start`) is silent on NaT. I chose **fail
   conservative**: an unresolved event cannot be proven to end before
   the test span, so it is *kept out* of training unless its `t0`
   clears the post-test embargo. pandas comparison `NaT < x` is
   `False`, which gives this behavior naturally. **Confirm**: in
   production, unresolved events at training time should be rare
   (backfill labels everything), but in live they'll exist — and
   dropping them from training is the safe call.

4. **`purge` boundary strictness.** §7 writes `t1 < test_start` (strict)
   and `t0 > test_end + emb` (strict). I implemented both strict and
   added a test (`test_embargo_boundary_is_strictly_greater`) pinning
   that an event with `t0 == test_end + emb` is purged. The doc's
   pseudocode is consistent with this; just calling it out because
   off-by-one at fold boundaries is the classic silent leak.

5. **CPCV "5 backtest paths" construction.** §7 says "N=6, k=2 → 15
   splits, 5 synthetic backtest paths" but doesn't give the path
   construction. The López de Prado reference (AFML, CPCV) makes the 5
   paths a **1-factorization of K₆** — 5 edge-disjoint perfect
   matchings covering all 15 pairs exactly once. I implemented the
   standard round-robin 1-factorization and asserted it covers every
   pair once. **This is the one place I went outside the doc** for the
   construction algorithm; the *counts* (15 splits, 5 paths) match the
   doc exactly. If you intended a different path-grouping (e.g.
   López de Prado's φ = (k/N)·C(N,k) greedy rotation), the test will
   need to relax — but a 1-factorization is the canonical reading and
   is what the PBO literature expects.

6. **Registry `run_id` uniqueness.** I made `run_id = hash[:12]-ts-pid`
   so two runs of the same config get distinct ids but the same
   `config_hash` (which is what counts the trial). The schema uses
   `INSERT OR REPLACE` on `run_id` PK; if you'd rather append on
   collision than replace, say so. Current behavior: same `run_id`
   re-run overwrites; distinct runs (the normal case) append.

## Decisions I'd like from you before Phase 1

1. **Vertical `r_realized` policy** (ambiguity #1 above) — confirm the
   store-the-mark, size-from-zero split.

2. **Where do barrier constants live in Phase 1?** The §3 YAML example
   puts them under `barriers:` per-rule, but §0 says they're
   pre-registered constants, not per-rule tunables. My read: they live
   in the rule YAML for *attribution* (so the trial registry records
   which barriers a run used) but are **read-only** — any value other
   than 1.5/1.0/24 still counts as a trial. Confirm before I build the
   config loader.

3. **Real EURUSD data source for Phase 1.** The doc assumes TimescaleDB
   is already populated (§9 "Existing AQR component: TimescaleDB →
   Bars as hypertables"). For a standalone Phase 1 I need to know:
   is there an existing DB to point at, or do I need an MT5/CSV ingest
   path? `docker-compose.yml` is staged but the service isn't stood up
   this session per the prompt.

4. **ATR definition.** §3 says `atr: {period: 14, source: closed_bars_only}`
   but doesn't specify Wilder's vs. simple vs. exponential. I assumed
   Wilder's (the MT5/fx convention). Phase 1's feature service needs
   this pinned — flag now.

5. **Entry price timing.** §4 says "entry at next H4 bar open." For
   synthetic Phase 0 paths I anchor `entry_px` as a given constant. In
   Phase 1, is "next H4 open" the open of the *next H4 bar after t0*
   (i.e. ~4 hours later), or the open of the M1 bar immediately
   following the signal? The former is the doc's plain reading; the
   latter would change event density materially. Confirm.

## Not done (correctly out of scope)

- Rule engine beyond `trivial_long_rule` (only exists to give the
  labeler an event to resolve in tests).
- Feature service / Tier 1–3 features / feature selection (§5).
- Meta-model, calibration, threshold selection (§6).
- Sizing ladder, EV-in-CV-loop, hard overlays (§8).
- MT5 / execution / live barrier watcher (§9).
- TimescaleDB connection — synthetic data only, per prompt.
- Backfill/replay parity test (§3) — needs the real rule engine;
  stubbed for Phase 1.
