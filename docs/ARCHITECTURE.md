# TRIAGE — Triple-Barrier Meta-Label Architecture (v1)

**Scope:** Replaces the direction/magnitude prediction models in AQR's H4 slot with a rule-based primary layer + ML trade filter. Covers the 7 requested design areas, preceded by two preliminaries that end up driving most decisions: the sample budget and the design stance.

---

## 0. Design stance — three commitments up front

**1. The meta-model is a precision dial, not an alpha source.** If the primary rule has uniformly zero conditional edge — P(win | features) flat across feature space — no classifier rescues it. Meta-labeling works when the rule has *heterogeneous* edge: profitable in some regimes, bleeding in others. Corollary: the primary rule should be deliberately **recall-heavy and loose**. Its job is candidate generation; precision is the meta-layer's job. A tightly filtered primary starves the model of both training data and room to add value.

**2. Every tuned knob is a trial.** Barrier multipliers (1.5/1.0/24), the probability threshold τ, rule parameters, label-policy variants, Optuna trials — all of it inflates the DSR/PBO trial count. All configurations attempted get logged to a central trial registry, and the promotion gates are computed against the honest count. Barriers and label policy are **pre-registered constants** in Phase 1, not free parameters.

**3. Pooling across symbols is the plan of record.** Single-symbol Phase 1 exists to validate plumbing, not to render a final verdict. The sample budget (next section) makes this non-negotiable.

---

## 1. Sample budget — the constraint that shapes everything

H4 forex ≈ 1,560–1,700 bars/year (6/day × ~5.3 trading days × 52). A reasonably loose primary rule fires ~1–3×/week → **50–150 events/year/symbol**. Ten years of history:

| Scope | Events (approx.) | Verdict |
|---|---|---|
| 1 symbol, 1 rule | 500–1,500 | Plumbing test only. Too thin for reliable OOS conclusions. |
| 3 majors pooled | 1,500–4,500 | Minimum viable for a 10–15 feature GBM. |
| 6 symbols pooled | 3,000–9,000 | Target operating point. |

Consequences, stated flatly:

- **DL is disqualified by n, not by taste.** At n ≈ 10³–10⁴ tabular samples with 10–15 features, LightGBM/CatBoost dominate. Revisit DL only if pooled n exceeds ~50k *and* the GBM demonstrably plateaus. Neither will happen soon.
- Model capacity must be tiny: depth ≤ 4, `num_leaves` ≤ 15, high `min_child_samples`, heavy regularization (Section 6).
- Aggressive HPO is itself overfitting at this n. Optuna budget capped at ~50 trials over a coarse grid.

**Base-rate geometry (calibrates expectations):** for a near-driftless path with TP = 1.5×ATR and SL = 1.0×ATR, P(TP first) ≈ SL/(TP+SL) = 40% ignoring the vertical barrier; with 24-bar timeouts labeled 0, the unconditional positive rate lands around **32–38%**. Breakeven win rate at 1.5R is 1/(1+1.5) = 40% before costs; with cost c in R units, p\* = (1 + c)/2.5 ≈ **0.42–0.44** for majors. So the entire job of the meta-model is: find feature-space pockets where calibrated P(win) ≥ ~0.47, and size zero everywhere else. That is a much fairer fight than predicting next-bar sign, but it is not a free lunch — it requires the primary to have real conditional structure.

*(Side note: the old MeanReversion EA's 17% win rate against a ~40% breakeven confirms the primary rule matters more than anything downstream. That rule family is not the Phase 1 candidate.)*

---

## 2. System overview

```
                     ┌─────────────────────────────────────────────────┐
                     │                  DATA LAYER                     │
                     │  TimescaleDB: H4 bars, M1 bars, per-symbol      │
                     └──────┬─────────────────────────────┬────────────┘
                            │                             │
                  ┌─────────▼──────────┐        ┌─────────▼──────────┐
                  │  FEATURE SERVICE   │        │    RULE ENGINE     │
                  │  point-in-time,    │        │  config-driven,    │
                  │  closed bars only  │        │  versioned, no ML  │
                  └─────────┬──────────┘        └─────────┬──────────┘
                            │      candidate signal (t0, side, entry) │
                            └──────────────┬──────────────────────────┘
                                           ▼
                                 ┌──────────────────┐
                                 │   EVENT STORE    │  features snapshotted at t0
                                 └───┬──────────┬───┘
                        (backfill)   │          │   (live)
              ┌──────────────────────▼──┐   ┌───▼─────────────────────────┐
              │ LABELER (M1 first-touch)│   │ META-MODEL INFERENCE        │
              │ → labels, r_realized    │   │ calibrated P(win)           │
              └──────────┬──────────────┘   └───┬─────────────────────────┘
                         ▼                      ▼
              ┌─────────────────────┐   ┌─────────────────────┐
              │ TRAINING PIPELINE   │   │ THRESHOLD + SIZER   │
              │ uniqueness weights  │   │ EV ladder, overlays │
              │ purged CPCV         │   └───┬─────────────────┘
              │ LightGBM + calib    │       ▼
              │ → MODEL REGISTRY    │   ┌─────────────────────┐
              └─────────┬───────────┘   │ MT5 EXEC ADAPTER    │──► outcome logging
                        │               └─────────────────────┘    closes the loop
                        ▼
              ┌─────────────────────┐
              │ TRIAL REGISTRY      │  every config ever run → DSR/PBO trial counts
              └─────────────────────┘
```

The M1 layer's path-to-close monitor becomes the **barrier resolution engine** in both backfill (vectorized first-touch) and live (TP/SL/vertical watcher). The H4 model slot becomes rule engine + meta-inference at bar close. Nothing else in the AQR event loop changes.

---

## 3. Primary signal module (ask #2)

**Contract.** A rule is a pure function `(bar_window, params) → Optional[Signal(side, entry_type)]` with a declared max lookback and zero hidden state. Registered in a rule registry, versioned by **content hash of code + config** — every event row carries `rule_version`, because a label is meaningless without the exact rule that produced it.

```yaml
# rules/donchian_break_v1.yaml
rule_id: donchian_break
universe: [EURUSD]
timeframe: H4
params:
  channel_bars: 20
  trend_filter: { tf: D1, ema_period: 50, mode: slope_align }
  session_mask: [london, ny, overlap]
signal:
  entry: next_bar_open          # kills same-bar look-ahead by construction
barriers:                        # pre-registered, NOT tuned per run
  tp_atr: 1.5
  sl_atr: 1.0
  vertical_bars: 24
  atr: { period: 14, source: closed_bars_only }
```

**Two code paths, one truth.** Vectorized backfill (vectorbt/numpy) for history; incremental evaluation for live. These must pass a **bar-replay parity test**: replay history bar-by-bar through the live path and assert byte-identical signal sets versus the vectorized run. This test is non-optional — backfill/live divergence is the classic silent killer of exactly this architecture.

**Phase 1 rule commit:** Donchian-20 H4 breakout, direction-aligned with D1 EMA-50 slope, London/NY sessions only. Rationale: breakouts have a natural invalidation level (clean SL geometry), positive skew that matches asymmetric 1.5R/1R barriers, most false-breakout failures are *conditionally* predictable (vol regime, level proximity, HTF alignment — exactly the Tier 1+2 features), and the basket contains trend-prone instruments (XAU, US30, BTC) for later phases. Keep the trend filter as the *only* gate on the rule — resist adding more filters; that is the model's job.

---

## 4. Labeling pipeline (ask #1)

**Entry convention.** Signal evaluated at H4 bar close t0; entry at next H4 bar open. ATR and all features computed from bars ≤ t0's close — the still-forming bar never enters any computation.

**First-touch resolution on M1** (reusing path-to-close infra):

```python
def resolve(event, m1_bars):                      # m1_bars ∈ (entry_time, t_vert]
    for bar in m1_bars:
        tp_hit = bar.high >= event.tp_px if event.side == +1 else bar.low  <= event.tp_px
        sl_hit = bar.low  <= event.sl_px if event.side == +1 else bar.high >= event.sl_px
        if tp_hit and sl_hit:
            return Outcome("sl", bar.time)        # same-M1-bar ambiguity → conservative
        if sl_hit: return Outcome("sl", bar.time)
        if tp_hit: return Outcome("tp", bar.time)
    return Outcome("vert", event.t_vert)
```

The same-bar tie-break to SL is deliberate pessimism; at M1 granularity it is rare, but the rule must be deterministic and documented.

**Timeout policy (committed):** `label = 1` iff outcome = TP; SL and vertical both → 0. Store `r_realized` (signed R-multiple at exit, mark-to-market for verticals) on every event regardless — it feeds EV analysis and later weighting experiments. Run exactly **one pre-registered ablation** in Phase 1 (vertical → sign(r_realized) instead of 0), logged as 2 trials in the registry, then freeze the policy.

**Sample uniqueness weights (Lopez de Prado ch. 4).** Labels spanning [t0, t1] overlap when signals fire while a prior event is unresolved. Per H4 bar t, concurrency c_t = number of events whose span contains t; average uniqueness of event i:

```
u_i = mean over t ∈ [t0_i, t1_i] of (1 / c_t)        →  sample_weight_i = u_i
```

Two implementation rules: (a) weights are computed **per fold on training events only** — computing concurrency over the full event set leaks test-period event timing into training; (b) skip sequential bootstrap in Phase 1 — with sparse H4 signals, uniqueness weighting alone captures nearly all the benefit at a fraction of the complexity. **Label all triggers**, including ones that fire while a position is open (they are valid P(win | setup) samples and data is the scarce resource); live-side position gating is an execution concern, handled in the sizer.

**Event store schema:**

```sql
CREATE TABLE events (
  event_id      bigserial PRIMARY KEY,
  symbol        text NOT NULL,
  rule_id       text NOT NULL,
  rule_version  text NOT NULL,          -- content hash
  t0            timestamptz NOT NULL,   -- signal bar close
  side          smallint NOT NULL,      -- +1 / -1
  entry_px      numeric NOT NULL,
  atr0          numeric NOT NULL,
  tp_px         numeric NOT NULL,
  sl_px         numeric NOT NULL,
  t_vert        timestamptz NOT NULL,
  t1            timestamptz,            -- first-touch time
  outcome       text,                   -- tp | sl | vert
  label         smallint,               -- 1 iff tp
  r_realized    numeric,
  features      jsonb NOT NULL,         -- point-in-time snapshot
  feature_ver   text NOT NULL,
  created_at    timestamptz DEFAULT now()
);
```

**Correctness battery (Phase 0, blocking):** synthetic M1 paths with analytically known outcomes; shuffled-label test (train on permuted labels → OOS AUC must ≈ 0.50, else the harness leaks); feature-timestamp audit (assert max source-bar timestamp ≤ t0 for every snapshot); backfill/replay parity (Section 3); purge-overlap assertion inside every CV split.

---

## 5. Feature service

**Point-in-time snapshotting is the anti-leakage architecture.** Features are computed once, at signal time, from closed bars only, and persisted into `events.features` with a `feature_ver` hash. They are never recomputed later from a full-history frame — recomputation is where off-by-one windows and revised data silently poison labels.

Phase 1 ships **Tier 1 + Tier 2 + Tier 3** (Tier 3 is free) — roughly 14–16 columns as specified in the brief, no additions. Tier 4 cross-sectional features enter at Phase 3 only, gated by MDA evidence.

**Feature selection:** correlation-prune first (drop one of any pair with |ρ| > 0.85 — this addresses the substitution effect directly), then **MDA under purged CV**: permute each surviving feature within OOS folds, measure the drop in strategy-relevant metric (mean OOS trade EV, not raw AUC). Features whose MDA confidence interval spans zero after two phases get cut. MDI/gain importances are diagnostics only, never selection criteria.

---

## 6. Meta-label model (ask #3)

**Committed: LightGBM binary classifier.** Symbol, rule_id, and session enter as native categoricals (this is what makes pooling clean). CatBoost is the sanctioned fallback if categorical handling misbehaves at Phase 2 pooling — not run in parallel, to keep the trial count honest.

Reference configuration (starting point, coarse Optuna grid around it):

```python
params = dict(
  objective="binary", metric="binary_logloss",
  max_depth=4, num_leaves=15,
  min_child_samples=max(30, int(0.03 * n_train)),
  learning_rate=0.03, n_estimators=2000,        # early stopping decides
  reg_lambda=10.0, feature_fraction=0.7, bagging_fraction=0.8,
)
# fit(..., sample_weight=uniqueness_w, eval_set=purged_val, early_stopping_rounds=100)
```

**No class rebalancing.** The ~35% base rate is the true prior and the sizer consumes absolute probabilities; rebalancing would destroy calibration for zero benefit.

**Calibration is a first-class stage, not a nicety** — the sizing layer consumes P(win) as a literal probability. Platt/sigmoid calibration below ~2k training events, isotonic above, fit on a purged calibration fold. **Brier score and the reliability curve are reported alongside AUC on every run**; an uncalibrated 0.58-AUC model is worse for this system than a calibrated 0.55 one.

**Optuna/Ray integration:** ≤ 50 trials, fixed purged fold definitions shared across all trials (re-randomizing folds per trial is a leak into the search). Ray parallelizes trials × folds × symbols — trivially fits the two-node cluster. The actually compute-heavy job is the **M1 barrier-resolution backfill**, which is embarrassingly parallel per (symbol, year) chunk; vectorize the first-touch scan rather than looping bars in Python.

**Threshold τ lives inside the CV loop.** It is selected on training folds via the EV rule (Section 8) and evaluated frozen on test folds. Every τ grid point tried is a trial in the registry.

---

## 7. Validation framework (ask #4)

**Purged K-fold + embargo.** An event's information lives over [t0, t1]. For each test fold, drop every training event whose span overlaps the test span, then embargo training events starting within h bars after test end:

```python
def purged_train_idx(events, test_start, test_end, embargo_bars=48):
    emb = embargo_bars * H4
    ok  = (events.t1 < test_start) | (events.t0 > test_end + emb)
    return events.index[ok]
```

Embargo = **48 H4 bars (2× the vertical horizon)** — covers label overlap plus feature serial correlation with margin.

**CPCV for PBO.** Combinatorial purged CV with N = 6 groups, k = 2 test groups → 15 splits, 5 synthetic backtest paths. PBO computed across the logged configuration set on these paths. This is the machinery that makes the existing PBO ≤ 0.20 gate meaningful rather than decorative.

**Promotion gates.** Existing gates survive unchanged in value but change in *object*: they now apply to the **filtered + sized strategy returns, after conservative costs**, over CPCV paths, with trial counts from the registry. Two new gates are added:

| Gate | Criterion | Rationale |
|---|---|---|
| **G1 — Uplift (new)** | Filtered strategy beats take-every-signal primary: block-bootstrap of trade-level PnL, ΔSharpe > 0 at 95% confidence | If the meta-layer adds no uplift over the naked rule, reject the model even if absolute numbers pass — otherwise you ship complexity for nothing |
| G2 — DSR | ≥ 0.95, honest trial count | unchanged |
| G3 — PBO | ≤ 0.20 via CPCV | unchanged |
| G4 — t-stat | ≥ 3.0 on OOS mean trade R | unchanged |
| **G5 — Calibration (new)** | Brier < climatology Brier; reliability slope ∈ [0.8, 1.2] on OOS | The sizer consumes probabilities; a miscalibrated model breaks sizing even with good ranking |

**Baseline ladder** — each rung must justify itself against the previous one: (1) primary take-all → (2) + meta filter at τ → (3) + probability-laddered sizing. G1 tests rung 2; a second uplift check tests rung 3.

**Cost model:** per-symbol-class, conservative (≈1.2× typical spread + slippage haircut). For majors, cost ≈ 0.03–0.05R against a 1×ATR H4 stop; XAU and BTC get their own spread-regime tables, BTC additionally a weekend-gap adjustment. Marginal trades near τ are precisely the cost-sensitive ones, so gating on pre-cost numbers is self-deception.

**Leakage checklist (asserted in code, not prose):** features from closed bars ≤ t0; ATR excludes the entry bar; uniqueness weights per training fold only; calibrator/scalers/fracdiff-d fit on train only; purge-overlap assertion on every split; shuffled-label harness test green.

---

## 8. Position sizing (ask #5)

**Division of labor, committed:** regime information belongs in the *features* (the model turns it into p); the sizer consumes **calibrated p and risk budget only**, plus hard risk overlays. Feeding regime into both model and sizer double-counts the same information and makes attribution impossible.

**EV in R units** (vertical conservatively valued at 0):

```
EV(p) = 1.5·p − 1.0·(1 − p) − c   =   2.5·p − 1 − c
p*    = (1 + c) / 2.5                          # breakeven
τ0    = p* + 0.05                              # entry threshold with margin
```

With c ≈ 0.05R → p\* ≈ 0.42, τ0 ≈ 0.47.

**Discrete sizing ladder (Phase 1–3):**

```
p < τ0          → no trade
τ0 ≤ p < τ1     → 0.5 × base_risk        (τ1 ≈ 0.55, set on training folds)
p ≥ τ1          → 1.0 × base_risk
```

`base_risk` = 0.5–1.0% equity per trade; the ATR-based stop means fixed-fractional risk is *already* volatility-normalized (position size auto-shrinks when ATR expands). Continuous Kelly — f\* = p − (1−p)/1.5, quarter-fractional — is the theoretical ceiling and the derivation behind the ladder, but a continuous function of a noisy p̂ is fragile; the ladder is robust, auditable, and loses little. Revisit continuous sizing only after G5 has passed for two consecutive quarters of live paper data.

**Hard overlays (risk, not alpha):** ATR percentile > 95 → halve size; aggregate open risk across symbols ≤ 3 × base_risk; daily equity stop; per-symbol single open position in live execution (the labeler still labels all triggers — train/serve difference is by design, per Section 4).

---

## 9. Integration plan (ask #6)

| Existing AQR component | Role in TRIAGE |
|---|---|
| M1 path-to-close monitor | Barrier resolution engine — backfill (vectorized) and live (TP/SL/vertical watcher). Direct reuse. |
| H4 model slot | Rule engine + meta-model inference at H4 close. Same event loop, new occupant. |
| Ray + Optuna | HPO and CPCV parallelization, capped budget. Unchanged infra. |
| vectorbt | Primary-rule backtests and the take-all baseline for G1. |
| TimescaleDB | Bars as hypertables; `events` as a regular table. |
| LightGBM/CatBoost stack | Meta-model layer. |
| LLM hypothesis agent | Proposes rule configs into a queue; each config auto-runs the full gate pipeline; **every attempt logs to the trial registry** — this is what keeps DSR/PBO honest at agent-generated scale. |
| MT5 execution + Telegram | Unchanged; consumes (symbol, side, size) with barrier prices attached. |
| Failed direction models (10+) | **Retired outright. Not recycled as features.** They failed the gates; importing their scores imports fitted noise plus a dependency, and contaminates attribution of the new system. |

Net-new components are small: rule engine (~300 lines + configs), labeler (~200 lines + tests), trial registry (a table + a decorator), sizing module (~100 lines). Everything heavy already exists.

---

## 10. Phased rollout (ask #7)

**Phase 0 — Harness correctness (≈1 week).** Labeler + purged-CV machinery + parity tests, all against synthetic data with known answers. *Exit: full correctness battery green. No market data conclusions drawn.*

**Phase 1 — EURUSD × Donchian-20 (2–3 weeks).** Full pipeline end-to-end, baseline ladder, all gates computed, the one pre-registered label ablation. *Exit is a decision, not a pass/fail:*

- **G1 uplift passes** (regardless of marginal G2–G4 at this n) → proceed to Phase 2; single-symbol n is too small to demand the full stack.
- **Classifier separates (AUC 0.53–0.56) but no strategy uplift** → inspect the EV-vs-τ curve and cost interaction before touching the model; this failure is usually threshold/cost geometry.
- **p̂ distribution flat, AUC ≈ 0.50** → the primary has no conditional edge on these features. **Swap the rule, not the model.** Do not add features to force separation.

**Phase 2 — Pool EURUSD/GBPUSD/USDJPY (2 weeks).** Symbol as categorical; the data-scarcity fix. Full gate stack now binding, honest trial counts from Phases 0–2. *Exit: G1–G5 pass on pooled CPCV.*

**Phase 3 — Full basket + second rule family (3–4 weeks).** Add XAUUSD/US30/BTCUSD with per-class cost models — test pooled-with-categorical versus split models, prefer pooled unless MDA/gates say otherwise. Introduce one second rule family (e.g., HTF-aligned pullback) as a new `rule_id`, and Tier 4 features gated by MDA. *Exit: gates pass at basket level; per-symbol attribution shows no single symbol carrying the book.*

**Phase 4 — Paper → small live.** Production monitors: **calibration drift** (CUSUM on rolling realized hit-rate minus mean predicted p̄ — the earliest structural-break alarm this design admits), PSR degradation of live versus CPCV expectation, and cost-model realism (realized vs. assumed slippage). Kill-switch criteria written down *before* going live.

---

## 11. What falsifies this design

- **Flat p̂ everywhere across multiple sensible primaries** → H4 setups in this basket lack ML-recoverable conditional structure at these horizons; the meta-labeling reframe itself is rejected, not just a model.
- **G1 uplift repeatedly fails while classifier metrics pass** → the edge pockets exist but sit below cost; the barrier geometry or the instrument set is wrong.
- **CPCV passes, paper trading dies with calibration CUSUM firing early** → regime nonstationarity dominates at this horizon or a backfill/live parity defect survived Phase 0; the parity tests get audited first, then the design is shelved rather than patched.
- **Uniqueness weights ≈ 1.0 everywhere** → signals so sparse the weighting is a no-op. Harmless; keep the machinery, it costs nothing.

The system is designed so each of these produces a *specific, cheap-to-read* signature rather than a vague underperformance — which is the whole point of rejection rigor.
