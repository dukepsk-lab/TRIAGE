# Prompt for GLM-5.2 — TRIAGE Full Build (Phases 1-4)

Before starting: place `ARCHITECTURE.md` at `docs/ARCHITECTURE.md` in the
repo. Read it completely before writing any code in any phase — it is the
source of truth throughout this build.

This is a multi-week, multi-session build. **You will hit a hard stop at
the end of every phase and wait for explicit sign-off before starting the
next one**, even if everything passes cleanly. This is not a formality —
each phase's gates depend on the trial count and design decisions from every
prior phase, and compounding an error silently across phases is exactly the
failure mode this project is designed to avoid.

**At the start of any new session**, first check `docs/` for which phase
reports already exist (`PHASE1_RESULTS.md`, `PHASE2_RESULTS.md`, etc.) to
determine where the project actually stands. Do not assume — verify against
what's on disk and in the trial registry.

---

## Global non-negotiables (apply to every phase, no exceptions)

- **No look-ahead.** Every feature/ATR value comes only from bars closed
  ≤ t0. Enforce this in code (a timestamp audit utility), not just in
  review.
- **Barriers are frozen:** TP = 1.5×ATR, SL = 1.0×ATR, vertical = 24 H4
  bars. These are not tunable parameters at any phase covered here.
- **Same-M1-bar TP+SL ambiguity → resolves to SL.** Named, tested branch.
- **Timeout label policy:** label = 1 iff TP hit first. SL and vertical
  timeout both → 0. Always store `r_realized` regardless of label.
- **Every experiment run — including failed/rejected ones — writes to the
  trial registry.** No exceptions, no "just a quick test."
- **Uniqueness weights computed per training fold only**, never globally.
- **Purge + embargo on every CV split**, embargo = 48 H4 bars.
- **The primary rule stays loose.** If a phase's classifier isn't
  separating, the fix is usually the rule or the cost/threshold geometry —
  not more features, not a bigger model. Re-read §0 of the architecture if
  tempted to add complexity to force a result.
- **Retired direction-prediction models are not recycled as features at
  any point.** They failed the promotion gates; their outputs are fitted
  noise, not signal.

---

## Phase 1 — EURUSD × Donchian-20, single symbol end-to-end

This is the starting phase — no prerequisite. It combines the correctness
harness with the first real pipeline run, so the harness gets validated
against real data pressure immediately instead of only synthetic fixtures.

**Step 0, before anything else:** audit existing TimescaleDB EURUSD H4/M1
coverage — date range, gaps, quality issues. Write
`docs/PHASE1_DATA_AUDIT.md`. If coverage is under ~5 years or gaps are
severe, **stop and report — do not fetch fresh data from MT5 yourself**,
that decision isn't yours to make.

**Step 1 — build and prove the correctness harness** (still required, just
not a standalone phase):
- Synthetic M1 path generator with analytically known first-touch outcomes
  (ramp-to-barrier, flat-timeout, same-bar-ambiguity cases) — used purely
  as unit-test fixtures for the labeler, not as a phase deliverable in
  itself.
- Triple-barrier labeler (`resolve()` per §4), tested exactly against those
  synthetic outcomes before it ever touches real M1 data.
- Purged K-fold + embargo + CPCV (N=6, k=2) per §7, tested against a
  hand-enumerated synthetic overlap case — not just "fewer events than
  before."
- Shuffled-label leakage test harness (permute labels, train trivial
  LightGBM, assert OOS AUC ≈ 0.50 ± 0.05). Build as a reusable function —
  every later phase re-runs it.
- Feature-timestamp audit utility, tested against both a clean and a
  deliberately-leaky fixture, then wired into the real feature service
  itself (not just left as a standalone test).
- Trial registry (SQLite is fine here): `run_id, timestamp, config_hash,
  config_json, gate_results_json`, plus a `@logged_trial` decorator.

Do not proceed to Step 2 until all of the above passes `pytest`. This is
still a blocking gate — it's just not a separate calendar phase anymore.

**Step 2 — real pipeline:**
- Real Donchian-20 + D1 EMA-50 rule engine (§3 YAML spec), both vectorized
  backfill and incremental live-eval paths. **Bar-replay parity test is a
  gate**, not optional: replay bar-by-bar through the live path, assert
  identical signals to the vectorized run.
- Tier 1+2+3 feature service (~14-16 columns, §5), every snapshot checked
  by the timestamp auditor from Step 1.
- Real labeling run over full EURUSD history, persisted per the `events`
  schema in §4, uniqueness-weighted per fold.
- One pre-registered label ablation (vertical→0 vs vertical→sign(r)),
  logged as two separate trial-registry entries. Report the comparison,
  don't pick a winner yourself.
- LightGBM meta-model per §6 reference config, purged CV + CPCV, Platt
  calibration (n will be small at single-symbol), Brier + reliability
  curve reported, ≤50 Optuna trials on fixed folds.
- Baseline ladder (take-all → +meta filter → +EV sizing) and gates G1-G5
  per §7, EURUSD cost model (~0.03-0.05R).
- Sizing module: discrete ladder from §8, τ0/τ1 selected on training folds
  only.

**STOP condition — this is a decision, not a pass/fail.** Report which of
three outcomes you landed on (G1 passes → recommend Phase 2; separates but
no uplift → show EV-vs-τ curve and cost breakdown; flat AUC → recommend
swapping the rule, explicitly do not force separation via more features).
Write `docs/PHASE1_RESULTS.md` with the gate table, EV-vs-τ curve, ablation
comparison, and recommendation. **Wait for sign-off before Phase 2.**

---

## Phase 2 — Pool EURUSD / GBPUSD / USDJPY

Prerequisite: explicit go-ahead on Phase 1 results.

- Extend the rule engine and feature service to run identically across the
  three majors (should be near-zero new code if Phase 1's abstractions are
  clean — if it isn't, that's a signal Phase 1 was too single-symbol-
  specific and needs refactoring before proceeding).
- Add `symbol` as a native categorical in the LightGBM model. Pool events
  across all three into one training set.
- Full gate stack (G1-G5) now binding — this is the first phase where the
  gates are meant to be taken as a real verdict, not a sanity check. Trial
  counts must be honest cumulative counts from Phase 1-2, pulled from the
  registry, not reset.
- Re-run the shuffled-label harness on the pooled set.

**STOP condition:** `docs/PHASE2_RESULTS.md` with pooled gate results,
per-symbol breakdown (is one symbol carrying the pooled result? say so
explicitly if so), and honest trial-count accounting. **Wait for sign-off
before Phase 3.**

---

## Phase 3 — Full basket + second rule family

Prerequisite: explicit go-ahead on Phase 2 results.

- Add XAUUSD, US30, BTCUSD with their own cost-model tables (BTC also
  needs a weekend-gap adjustment — flag this explicitly in the rule
  engine, don't let it silently misprice barriers over weekend gaps).
- Introduce exactly one second rule family (HTF-aligned pullback, per
  §"Phase 3" in the architecture) as a new `rule_id` — do not invent
  additional rules beyond this.
- Add Tier 4 cross-sectional features, gated by MDA evidence — run MDA
  first, only add features that clear it, per §5's selection method (drop
  |ρ|>0.85 pairs first, then MDA under purged CV, not MDI/gain).
- Decide pooled-vs-split by symbol class using gate results and MDA
  evidence, not a preference.

**STOP condition:** `docs/PHASE3_RESULTS.md` — basket-level gate results,
attribution showing no single symbol/rule dominating, MDA results for Tier
4 features (including ones that got cut). **Wait for sign-off before
Phase 4.**

---

## Phase 4 — Paper trading → small live

Prerequisite: explicit go-ahead on Phase 3 results, and this phase touches
real order flow eventually — treat every step here as higher-stakes than
the previous phases.

- Wire the MT5 execution adapter to consume (symbol, side, size) with
  barrier prices attached, per the integration table in §9.
- Build the production monitors before going live, not after:
  - **Calibration drift CUSUM** on rolling realized hit-rate vs. mean
    predicted p̄ — this is the earliest structural-break alarm the design
    has, per §10.
  - PSR degradation of live vs. CPCV expectation.
  - Realized vs. assumed slippage/cost tracking.
- **Write kill-switch criteria to `docs/PHASE4_KILLSWITCH.md` and get my
  explicit sign-off on the numbers before any live capital is at risk.**
  This is not something to infer from the architecture doc — bring me
  concrete thresholds to approve.
- Start in paper mode. Do not size up to live capital without a separate,
  explicit go-ahead from me after reviewing paper results.

**STOP condition:** paper trading running with all monitors live and
logging. Report cadence and content of what you'll surface to me (don't
let this run silently — a live system with no reporting loop back to me
defeats the point of the monitors).

---

If anything in the architecture is ambiguous at any phase, or if a phase's
results don't cleanly land in one of its stated decision branches, stop and
ask rather than guessing. Guessing is fine for a prototype; it is not fine
for this system.
