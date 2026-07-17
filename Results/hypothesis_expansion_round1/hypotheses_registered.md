# TRIAGE — Additional Trial Round: Hypothesis Pre-Registration

**Round scope:** Steps 1-2 only (hypothesis generation + pre-registration), per explicit user
instruction. **No backtesting, training, or registry writes happened in this round** — see
"Why no execution" below. Everything in this file is frozen *before* any results are seen,
exactly as Step 2 requires.

## Step 0 recap — infrastructure reused (not rebuilt)

| Piece | Location | Reused as-is? |
|---|---|---|
| Trial registry | `src/triage/registry/__init__.py` (`TrialRegistry`, `logged_trial`) | Yes — schema unchanged |
| CPCV harness | `src/triage/validation/__init__.py` (purged splits, 48-bar embargo, N=6/k=2) | Yes |
| Triple-barrier labeler | `src/triage/labeling/__init__.py` (`resolve()`, M1 first-touch) | Yes |
| Feature service (Tier 1-3) | `src/triage/features/service.py` (`FEATURE_COLUMNS`) | Yes |
| Tier 4 cross-sectional + MDA gate | `src/triage/features/tier4.py` | Yes; new signal-quality features below follow the same MDA-gate discipline (called "Tier 5" candidates) |
| Cost model | `src/triage/gates.py` (`COST_BY_SYMBOL`) | Yes |
| G1-G5 gate formulas | `src/triage/gates.py` (`block_bootstrap_uplift`, `dsr`, `pbo_from_cpcv`, `compute_gates`) | Yes, exact formulas |
| Existing rule families | `src/triage/rules/donchian.py`, `pullback.py` | Reused as the "already-registered" baseline that new hypotheses must be orthogonal to |

**Explicitly missing (flagged per Step 0, not reconstructed):** no `dsr-pbo-math` or
`five-sins-audit` skill files, and no `critic.md`/`hypothesis.md` agent prompts exist anywhere
in this repo or filesystem. Their content is not silently invented — the gate math above comes
straight from `gates.py`, and the Five Sins checklist (when a future session runs Step 5) should
use the exact five categories listed in the source prompt (look-ahead, survivorship,
narrative-without-mechanism, proxy-objective, unrealistic costs), nothing added or removed.

Also flagged: **Optuna is not a project dependency** (`pyproject.toml` lists `lightgbm` only).
`docs/AUTONOMOUS_GOAL.md` hit the same gap and fell back to a small manual coarse grid on fixed
purged folds. The HPO spaces below are written in Optuna-style bounds but should run as that same
manual coarse-grid fallback unless Optuna gets installed first.

## Why no execution this round

`.triage/` (bar store + `registry.sqlite`) is gitignored and does not exist in this checkout — no
market data, no persisted trial history. Every prior phase hit this identically
(`docs/PHASE3_RESULTS.md`'s "Honest trial count" section) and the data source has always been a
deliberate user decision (`docs/PHASE1_DATA_AUDIT.md`: "that decision isn't yours to make"). The
user chose, for this round, to stop at pre-registration and defer execution to a session that has
real data. **The cumulative honest trial count carried by the last committed results is 22**
(`docs/MAJORS_DONCHIAN_RESULTS.md`) — whoever runs Steps 3-6 against these hypotheses must seed
DSR/PBO from that number (or higher, if Phase 1/2's original 8+10 history is meant to carry
forward too — see the honesty note in `PHASE3_RESULTS.md`), not from zero.

---

## Orthogonality map

Existing registered hypotheses (baseline, not re-registered): **Donchian-20 breakout** (trigger =
channel breakout, regime = D1 trend filter, horizon ≈ 24 H4 bars) and **HTF pullback** (trigger =
retrace-to-EMA, regime = same trend filter, horizon ≈ 24 bars). Everything below is checked
against these two on all three axes before being accepted as genuinely new.

| ID | Instrument(s) | Trigger mechanism | Regime conditioning | Horizon |
|---|---|---|---|---|
| H1 | XAUUSD (+ FX majors arm) | Liquidity-sweep reversal | Range/low-trend only | Intraday |
| H2 | EURUSD, GBPUSD | Scheduled-event drift continuation | Post-decision window only | Multi-day |
| H3 | USDJPY | Intervention-zone momentum exhaustion | Proximity to known intervention level | Multi-day |
| H4 | US30 | Overnight gap fade | Elevated overnight-gap / roll window | Intraday |
| H5 | US30 | Vol-percentile breakout (dealer gamma) | Vol-compression→expansion | Intraday–short multi-day |
| H6 | BTCUSD | Funding-rate crowding reversion | Funding-extreme only | Intraday–2 day |

---

## H1 — Liquidity-Sweep Reversal (ICT-style), XAUUSD primary

*Priority per Step 1.4: same trigger event as Donchian breakout, opposite directional read —
tested first, before any unrelated new family.*

- **Instrument(s) / timeframe:** XAUUSD primary arm. Secondary, explicitly-scoped arm: pooled
  with EURUSD/GBPUSD to test whether the stop-cluster mechanic generalizes beyond gold (this is
  a scope sweep of *this same hypothesis*, not a separate registration — see "Folded, not new"
  below). H4 trigger, M1 barrier resolution, D1 for the regime filter — same convention as
  `donchian.py`.
- **Trigger rule (point-in-time, closed-bar):** reuse the existing rolling 20-H4-bar Donchian
  channel (`CHANNEL_BARS=20`, computed from bars with `close_time <= t0` only, per
  `features/audit.no_lookahead`). A **short** sweep-reversal fires at bar close `t0` when the
  bar's high pierces `donchian_high_20` by ≥ `SWEEP_DEPTH_MULT × ATR14` **and** the same bar's
  close is back below `donchian_high_20` (closed back inside the prior range — a sweep-and-reject
  bar). Long fires symmetrically on the low side. `SWEEP_DEPTH_MULT = 0.25` is the pre-registered
  constant (same magnitude convention as `pullback.py`'s `pullback_atr_threshold=0.25`, chosen for
  consistency, not tuned). Direction traded is **opposite** the sweep (fade). Entry at next H4
  bar open. Session mask: London/NY (reuse `session_mask`).
- **Regime gate (the deliberate inversion of Donchian's filter):** D1 EMA-50 slope must **not**
  be strongly trending in the sweep direction — this rule explicitly fires in range/exhaustion
  conditions, the opposite regime from where Donchian is allowed to fire. Concretely: reject the
  signal if `|D1 EMA-50 slope|` is above its own rolling-52-week 70th percentile in the sweep's
  direction (i.e., don't fade a genuine trending breakout).
- **Named counterparty:** retail trend/breakout traders and the stop-loss orders clustered just
  beyond the prior 20-bar swing high/low — a well-documented clustering pattern in gold and FX
  majors (obvious, round-channel levels are exactly where retail places stops). Bank/market-maker
  dealing desks need to source liquidity to fill larger opposing size and are structurally
  incentivized to push price through those obvious levels to trigger the cluster before reversing
  into their real directional position. The retail cluster is the repeatable loser because its
  stop placement is public, inferable information the counterparty trades against — this is
  standard, not speculative, market-microstructure behavior in gold specifically (concentrated
  retail participation, wide daily ranges relative to spread).
- **Triple-barrier scheme:** unchanged frozen constants — TP 1.5×ATR / SL 1.0×ATR / vertical 24
  H4 bars / ATR-14. No deviation; sweep-reversals are expected to resolve fast, so the existing
  vertical barrier should rarely bind (measure, don't assume — report median bars-to-resolution).
- **Feature set:** Tier 1-3 unchanged, reused as-is. Two new "Tier 5" signal-quality candidates,
  MDA-gated exactly like Tier 4 before being trusted: `sweep_depth_atr` (how many ATR the wick
  pierced beyond the level) and `rejection_close_position` (where in the bar's range the close
  landed, 0=low/1=high — a proxy for rejection strength).
- **HPO search space (bounded, stated up front):** `max_depth ∈ {3,4}`, `num_leaves ∈ {7,15}`,
  `min_child_samples = max(30, 0.03·n_train)`, `reg_lambda ∈ {5,10,20}`,
  `feature_fraction ∈ {0.6,0.7,0.8}`, `bagging_fraction ∈ {0.7,0.8,0.9}`, `learning_rate=0.03`
  fixed, `n_estimators=2000` with early stopping — the same capacity neighborhood as
  `meta.py::DEFAULT_PARAMS`, ≤ 40 configs, fixed purged folds shared across all trials (no
  re-randomizing per §6's leak warning).

## H2 — Scheduled-Event Drift Continuation, EURUSD & GBPUSD

- **Instrument(s) / timeframe:** EURUSD, GBPUSD, H4 trigger / M1 barriers.
- **Trigger rule:** fires only inside a pre-defined post-event window (next N H4 bars after a
  scheduled ECB or BoE rate decision, from a macro calendar — **new data dependency**, not
  currently ingested; flagged explicitly, not silently assumed available) when the H4 close breaks
  the pre-event 20-bar Donchian channel in the direction consistent with the surprise (actual vs.
  expected rate path). Direction = continuation (same direction as the breakout), not a fade.
  Entry at next H4 bar open. No session mask beyond the event window itself.
- **Regime gate:** binary — inside vs. outside the post-decision window. This replaces Donchian's
  always-on D1-trend filter with an event-conditioned gate; that swap is the orthogonal axis here
  (event-driven regime vs. price-derived trend regime).
- **Named counterparty:** real-money asset managers, pension funds, and corporate FX hedgers who
  rebalance or hedge on fixed schedules (month-end, quarter-end, mandate-driven) regardless of
  short-term price level — a well-documented, non-discretionary flow that persists in the
  direction the new rate differential implies. Retail traders who fade the initial post-decision
  move (a common, well-documented behavior) provide the repeatable losing side, run over as the
  scheduled flow keeps arriving over the following days.
- **Triple-barrier scheme:** **deviates from the frozen Phase 1-3 constants, deliberately and
  pre-registered here** (per §0's "any change is a new trial" rule — this is a new constant set
  for this hypothesis, frozen at spec time, not a post-hoc tune): TP 2.0×ATR / SL 1.0×ATR /
  **vertical 60 H4 bars (~10 trading days)**. Justification: the mechanism is explicitly multi-day
  drift, and the existing 24-bar window would truncate the horizon the hypothesis is about before
  it can play out.
- **Feature set:** Tier 1-3 reused, plus one new Tier-5 candidate: `days_since_event` (decay
  proxy for the flow's persistence), MDA-gated before trusting it.
- **HPO search space:** same bounded grid as H1.

## H3 — Intervention-Zone Momentum Exhaustion, USDJPY

- **Instrument(s) / timeframe:** USDJPY, H4 trigger / M1 barriers.
- **Trigger rule:** fires when (a) an extended one-directional run exists — N=6 consecutive H4
  bars with D1 close displaced more than 2×D1-ATR from the D1 EMA-50 — **and** (b) price is within
  a pre-registered proximity band (0.5×ATR) of a documented round-number intervention level
  (e.g. 150.00/152.00/160.00 — the specific historically-actioned levels must be pre-registered as
  a static list before backtesting, not chosen after seeing which levels "worked"). Entry on the
  first H4 reversal candle (close back inside the prior bar's range) after both conditions hold.
  Direction = counter-trend (fade the extended run).
- **Regime gate:** proximity-to-intervention-level, as above — a structurally different regime
  signal from both Donchian's trend filter and H1's range filter.
- **Named counterparty:** momentum/trend-following CTAs and carry-trade longs positioned for
  continued yen weakness — a specific, identifiable institutional actor (the Ministry of
  Finance/BoJ, via verbal or actual spot intervention) has a documented history of acting against
  exactly this crowd at exactly these levels. This is a real, repeatable, event-driven counterparty
  distinct from H1's generic stop-cluster story.
- **Triple-barrier scheme:** deviates from frozen constants, pre-registered: TP 1.5×ATR / SL
  1.2×ATR (wider stop — intervention reversals can be volatile and two-sided before resolving) /
  **vertical 48 H4 bars (~8 days)**, since intervention effects are documented to play out over
  days, not hours.
- **Feature set:** Tier 1-3 reused, plus new Tier-5 candidate `distance_to_intervention_level_atr`
  and `consecutive_trend_bars`, both MDA-gated.
- **HPO search space:** same bounded grid as H1.

## H4 — Overnight Gap Fade at Session Open, US30

- **Instrument(s) / timeframe:** US30, H4 (first bar of NY session) trigger / M1 barriers.
- **Trigger rule:** fires when the NY-session-open H4 bar opens beyond `GAP_MULT × ATR14` from the
  prior day's close (`GAP_MULT = 0.5`, pre-registered). Direction = fade (partial reversion toward
  prior close) within the session. Entry at the open of that same triggering bar (a deliberate,
  documented exception to "entry at next bar open," since the trigger *is* the open print itself —
  flagged explicitly so a reviewer can veto it rather than discover it silently).
- **Regime gate:** elevated overnight-gap regime, additionally conditioned on proximity to
  quarterly futures-roll/expiry dates (a static, pre-registered calendar of roll dates — **new
  data dependency**, flagged).
- **Named counterparty:** index-arbitrage and basis-trading desks that must correct overnight
  order imbalance accumulated during Asia/Europe hours and around quarterly futures roll dates;
  retail traders who chase the gap in the open direction provide the losing flow that the
  arb/basis correction runs against.
- **Triple-barrier scheme:** unchanged frozen constants (TP 1.5×ATR / SL 1.0×ATR / vertical 24 H4
  bars) — the mechanism is explicitly intraday, well inside the existing window.
- **Feature set:** Tier 1-3 reused, plus new Tier-5 candidate `overnight_gap_atr` and
  `bars_to_next_roll_date`, MDA-gated.
- **HPO search space:** same bounded grid as H1.

## H5 — Vol-Percentile-Conditioned Breakout (dealer gamma), US30

*Flagged as a borderline case, not silently resolved: this shares its trigger event (20-bar
channel breakout) with the existing Donchian rule. It is registered separately, not folded in,
because it replaces the regime-conditioning mechanism entirely (vol-compression→expansion instead
of trend-filter) and the counterparty story is structurally different (options-dealer hedging
flow, not trend-following continuation). A reviewer may reasonably decide this belongs in
Donchian's own sweep instead — surfaced here rather than decided unilaterally.*

- **Instrument(s) / timeframe:** US30, H4 trigger / M1 barriers.
- **Trigger rule:** same 20-bar Donchian breakout trigger as the existing rule, but the D1-trend
  filter is **replaced**, not combined, with: ATR-percentile-vs-52wk (already a Tier 1 feature)
  was below its own 20th percentile (compression) within the prior 10 H4 bars, and the breakout
  bar itself pushes ATR-percentile back above the 40th percentile (expansion) — i.e., only trade
  breakouts out of a genuine vol squeeze.
- **Regime gate:** vol-percentile-transition, as above — the "volatility-percentile" axis named
  explicitly in Step 1.3.
- **Named counterparty:** index-options dealers running gamma-hedging programs. In a
  vol-compression regime dealers are typically net long gamma and their hedging flow dampens
  moves (suppressing realized vol, favoring mean reversion) — a documented options-market
  mechanic, not specific to any one dealer. Once realized vol expands past a threshold, dealer
  gamma exposure flips (or existing long-gamma positions roll off), and their hedging flow
  reverses to *chase* price in the breakout direction, amplifying it. This is the mechanism this
  hypothesis is betting on, distinct from Donchian's generic trend-continuation story.
- **Triple-barrier scheme:** unchanged frozen constants (TP 1.5×ATR / SL 1.0×ATR / vertical 24 H4
  bars).
- **Feature set:** Tier 1-3 reused (ATR-percentile already present), no new Tier-5 features
  needed — the regime gate is built entirely from an existing feature, which is itself evidence
  this is a genuinely orthogonal *conditioning*, not a cosmetic parameter tweak.
- **HPO search space:** same bounded grid as H1.

## H6 — Funding-Rate Crowded-Positioning Mean-Reversion, BTCUSD

- **Instrument(s) / timeframe:** BTCUSD, H4 trigger / M1 barriers.
- **Trigger rule:** fires when perpetual-futures funding rate (**new data dependency** — not
  currently ingested; flagged explicitly) is beyond a pre-registered extreme
  (`|funding_rate| > FUNDING_EXTREME`, threshold to be set from the empirical funding-rate
  distribution *before* seeing trade outcomes) **and** price shows H4 momentum extension in the
  same direction as the crowd implied by funding sign (positive funding = crowded long, negative =
  crowded short). Direction = counter-trend (fade the crowd).
- **Regime gate:** funding-extreme-only — this itself is the regime filter, replacing both
  Donchian's trend filter and H1's range filter with a positioning-based signal, structurally
  different from either.
- **Named counterparty:** retail leveraged perpetual-futures traders paying funding to hold a
  crowded, over-levered directional position. When funding is extreme they are structurally the
  side that gets liquidated/squeezed on any reversion; market-making and basis-arbitrage desks are
  structurally positioned against the crowd's excess exposure and are compensated by the funding
  payment itself, independent of any view on direction.
- **Triple-barrier scheme:** deviates from frozen constants, pre-registered: TP 1.5×ATR / SL
  1.0×ATR / **vertical 12 H4 bars (~2 days)**, shorter than the standard 24 — funding resets every
  8h, so the crowding signal itself decays fast and a longer window would hold through funding
  resets that could invalidate the setup.
- **Feature set:** Tier 1-3 reused, plus new Tier-5 candidate `funding_rate_zscore` (vs. its own
  trailing distribution) and `funding_rate_bars_since_flip`, MDA-gated.
- **HPO search space:** same bounded grid as H1.

---

## Rejected at Step 1.2 (no identifiable counterparty — killed before any backtest)

Per Step 1.2, a hypothesis with no identifiable counterparty is rejected here, not after seeing
results. Reporting these explicitly rather than only listing survivors:

- **Round-number psychological bounce, all FX pairs.** ("Price bounces at round pip levels like
  1.1000.") No distinct counterparty could be named beyond "retail places orders at round
  numbers" — that's a description of the pattern, not a reason a specific participant keeps
  losing. Contrast H3, which *does* name a specific institutional actor (BoJ) acting at specific
  levels for a specific reason. Rejected.
- **Day-of-week seasonality (Monday reversal / Friday-close drift), all pairs.** Classic
  calendar-anomaly overlay. No structural participant story distinguishes it from a dataset-mining
  artifact — contrast H2's month/quarter-end rebalancing, which has a named, scheduled, real-money
  flow behind it. A generic "day of week" effect has no equivalent story. Rejected.

## Folded as near-duplicates, not registered separately (Step 1.3)

- **Sweep-reversal on GBPUSD/EURUSD individually.** Folded into H1's own instrument-scope arm
  (pooled majors + XAU), not a separate hypothesis — same trigger, same counterparty mechanism,
  only the instrument set varies.
- **Real-yield / central-bank-reserve-demand regime feature for gold.** This does not introduce a
  new trigger — it's a candidate *regime-conditioning feature* (e.g., "US 10y real yield falling")
  for the **existing, already-registered Donchian-XAU** hypothesis, not a new hypothesis in its own
  right. Recommended as a new Tier-5 candidate on the existing Donchian sweep, MDA-gated, rather
  than registered here.

---

## Definition of Done — status for this round

- [x] Located and reused existing trial registry, CPCV harness, labeler, cost model, gate
      definitions; explicitly flagged what's missing (skill/agent prompt files, Optuna, market
      data + persisted registry)
- [x] Every registered hypothesis (H1-H6) has a named counterparty, written before any
      backtesting
- [x] No direct direction/magnitude prediction proposed — all six are trigger-rule +
      P(win|triggered) meta-labeling problems
- [x] No near-duplicate hypotheses registered as separately "new" (see "Folded" section); one
      borderline case (H5) surfaced explicitly rather than resolved unilaterally
- [ ] Search space frozen before seeing OOS results — **N/A this round**, no execution happened;
      spaces above are frozen for whoever runs Step 4
- [ ] Cumulative N used for DSR/PBO — **N/A this round**; the number to seed from (22) is stated
      above under "Why no execution"
- [ ] Five Sins audit — **N/A this round**, nothing was backtested
- [ ] Trial registry updated — **deliberately not touched**; writing placeholder/un-executed rows
      to `.triage/registry.sqlite` would misrepresent the honest trial count the registry exists
      to protect. This document is the durable pre-registration artifact; the executing session
      should log each hypothesis's actual trials via `logged_trial` when it runs them.
- [x] Plain-language summary given, kills reported honestly (see next section)

---

## Plain-language summary

Six new hypotheses were pre-registered, each with a named counterparty and checked for
orthogonality against the two already-running rule families (Donchian breakout, HTF pullback) on
trigger mechanism, regime conditioning, and holding horizon:

1. **H1 — Liquidity-sweep reversal (XAUUSD, + FX-majors arm).** Fades the exact same breakout
   event Donchian trades, on the theory that it's a stop-hunt, not a real breakout, in range
   regimes. Prioritized first per the round's instructions.
2. **H2 — Post-rate-decision drift continuation (EURUSD, GBPUSD).** Multi-day, event-conditioned,
   riding real-money hedging/rebalancing flow after ECB/BoE decisions.
3. **H3 — Intervention-zone exhaustion fade (USDJPY).** Multi-day, fades extended yen moves near
   documented BoJ intervention levels.
4. **H4 — Overnight gap fade (US30).** Intraday, fades session-open gaps tied to overnight
   imbalance and futures-roll dates.
5. **H5 — Vol-percentile breakout via dealer gamma (US30).** Same breakout event as Donchian but
   conditioned on a vol-compression→expansion regime instead of trend; flagged as a borderline
   case for human judgment rather than silently folded in or silently kept separate.
6. **H6 — Funding-rate crowding reversion (BTCUSD).** Fades momentum when perpetual-funding
   extremes signal an over-levered crowd.

Two candidates were **killed before any backtesting** for lacking an identifiable counterparty
(generic round-number bounce; generic day-of-week seasonality) — both are patterns without a
named structural loser, exactly the trap Step 1.2 exists to catch. Two more candidate ideas were
**folded into existing hypotheses** rather than registered as new, to avoid inflating the trial
count with correlated near-duplicates.

**Nothing was backtested, trained, or gated this round** — this session has no market data and no
persisted trial registry (both are gitignored and were never committed, exactly the same
situation every prior phase in this repo documented hitting). The user explicitly chose to stop
at pre-registration. H2, H4, and H6 additionally need new data feeds (macro calendar, futures-roll
dates, perpetual funding rate) that aren't part of the current ingest pipeline — whoever runs
Steps 3-6 needs to source those first, or narrow scope to H1/H3/H5 which only need the OHLC data
the pipeline already ingests.
