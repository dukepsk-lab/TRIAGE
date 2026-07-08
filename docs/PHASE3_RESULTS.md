# Phase 3 Results — Full Basket + Second Rule Family

**Status: STOP — REJECT at the full-basket level (§11 falsification).**

This is the Phase 3 decision document per `docs/AUTONOMOUS_GOAL.md`. The full
pipeline ran end-to-end on real MT5 history for all six symbols, both rule
families, M1 first-touch labeled. **PRIMARY success was not reached**: on the
mandated full 6-symbol basket with holdout calibration, **G1 (uplift) fails
with flat p̂**. The bounded §6 decision tree was followed to exhaustion and the
§11 falsification signature was observed. Per §0/§8, an honest red is the
correct outcome here — not a manufactured green.

The single most useful finding: **the Phase 2 majors-donchian edge is real and
replicated (G1 p=0.001), but it does not survive the Phase 3 expansion.** The
second rule family (htf_pullback) has no recoverable edge, and XAU/US30/BTC do
not share the majors' conditional structure. Pooling them dilutes p̂ to chance.

---

## Data

- Bars re-ingested from the live MT5 terminal (IUX Markets Live, broker suffix
  `.`), D1 + H4 + M1, UTC, 2015→2026. **The bar store and trial registry did
  not exist in this checkout** (both are gitignored; see "Honest trial count").
- M1 first-touch labeling; frozen barriers 1.5/1.0/24; label=1 iff TP.

| Symbol | Event span (t0) | Events | Base rate |
|---|---|---|---|
| EURUSD | 2015-01-09 → 2026-07-06 | 1914 | 0.373 |
| GBPUSD | 2015-01-13 → 2026-06-26 | 1946 | 0.371 |
| USDJPY | 2015-01-09 → 2026-06-30 | 1964 | 0.381 |
| XAUUSD | 2017-03-08 → 2026-06-30 | 1559 | 0.447 |
| US30   | 2019-07-23 → 2026-07-02 | 1160 | 0.406 |
| BTCUSD | 2018-02-15 → 2026-06-30 | 1689 | 0.430 |

- **Pooled events: 10,232** (Donchian-20 + htf_pullback, both families).
  Per-rule: donchian 5431, pullback 4801.
- **Pooled base rate: 0.398** (§1 predicts 0.32–0.38 for majors; gold/BTC/US30
  run hotter, consistent with their higher realized hit rates).
- Outcomes: {sl: 5953, tp: 4076, vert: 203}.
- **Plumbing validation:** EURUSD (1059), GBPUSD (1112), USDJPY (1076) donchian
  counts reproduce Phase 1/2 **exactly** — the labeler is deterministic and the
  re-ingest is faithful.

## Leakage battery (§4 — blocking, PASSED)

- **Shuffled-label OOS AUC** on the pooled real feature matrix:
  **[0.5064, 0.5017, 0.5002]** — all within [0.45, 0.55]. No leak.
- **Feature-timestamp audit** (`no_lookahead`): enforced at snapshot time on
  every one of the 10,232 events during backfill (would have raised otherwise).
- **Donchian backfill/live parity** (`tests/unit/test_donchian_parity.py`):
  PASS on the populated bar store (previously skipped when no DB existed).
- Full suite: `pytest tests/ -q` green throughout.

## Feature selection (§5 — corr-prune + MDA on realized trade EV)

Correlation-prune (|ρ|>0.85) then **MDA under purged CV using the EV metric**
(mean OOS realized trade EV). ⚠️ The known `tier4.mda_selection` bug (the `"ev"`
branch returned *mean predicted probability*, not realized trade EV) was
**fixed first** (see commit history + `tests/unit/test_tier4_mda.py`), so this
selects on genuine `r_realized`-based captured EV per §5 — no deviation.

Result: **only `cs_rsi_rank` clears** (MDA CI barely above zero,
mean=+0.0022). Every other candidate — including base Tier 1–3 features — has an
MDA CI spanning zero, and several (ema_slope_h4, atr_pct, d1_roc_5, hour_utc)
have **negative** mean MDA (permuting them *improves* EV). That is itself a
low-edge signature: the features carry almost no strategy-relevant information
on the pooled basket. Final feature set = Tier 1–3 (15) + `cs_rsi_rank`.

## Headline meta-model (§6 — full6, holdout calibration)

LightGBM, purged 70/30 split, holdout (dedicated purged calibration fold),
uniqueness-weighted (training fold only), symbol/rule_id native categoricals.

| Metric | Value |
|---|---|
| AUC (OOS) | **0.5116** |
| Brier (OOS) | 0.2424 |
| Climatology Brier | 0.2417 |
| Pooled base rate | 0.398 |
| n_train / n_test | 5725 / 3070 |

AUC ≈ 0.51 is **essentially chance** — a collapse from Phase 2's 0.559 (3 majors,
donchian only). Brier is *above* climatology (the calibrator has nothing to
calibrate).

## EV-vs-τ (per-symbol cost model)

`mean_r` = mean realized R per taken trade **after per-symbol cost**.

| τ | n_trades | mean R (after cost) |
|---|---|---|
| 0.300 | 3028 | **−0.0423** |
| 0.375 | 2780 | −0.0474 |
| 0.400 | 2231 | −0.0350 |
| 0.425 |  485 | −0.0294 |
| 0.450 |  151 | −0.1086 |
| 0.475 |   71 | −0.0434 |
| ≥0.500 |   0 | — (no trades: calibrated p never exceeds 0.5) |

**Negative at every threshold.** Unlike Phase 1 (rising +0.07→+0.17), there is
no τ where filtering turns a profit. The meta-layer cannot rescue a primary
whose post-cost EV is negative everywhere.

## Gates G1–G5 (§7 — full CPCV, per-path refits)

Real PBO from CPCV N=6, k=2 → 15 splits, 5 paths, **per-path model refits**.
DSR at the honest registry count.

```
Gate       Value                                      Result
------------------------------------------------------------
G1 Uplift  ΔSharpe=0.264, p=0.227                     FAIL
G2 DSR     0.000 (n_trials=11)                        FAIL
G3 PBO     0.000                                      PASS*
G4 t-stat  -1.029                                     FAIL
G5 Calib   Brier=0.2424 vs clim=0.2417, slope=0.148   FAIL
```

\* **G3 PASS is vacuous here.** PBO=0 means the filtered strategy never
underperforms the primary out-of-sample — but *both are unprofitable*
(CPCV path sharpes: filtered ≈ −0.09 to −0.20, primary ≈ −0.54 to −0.74).
The filter loses *less* by trading less; that is not uplift. G1 (the gate that
matters, §7) and G4 both fail: the taken trades have a **negative** mean R
(t=−1.03).

## Per-symbol attribution (§10 — is one symbol carrying the book?)

**No symbol carries the book — none has recoverable edge.**

| symbol | n | AUC | mean_r | slope | base_rate |
|---|---|---|---|---|---|
| BTCUSD | 604 | 0.524 | −0.130 | 0.732 | 0.424 |
| USDJPY | 494 | 0.521 | +0.006 | 1.413 | 0.405 |
| US30   | 497 | 0.510 | +0.010 | 1.704 | 0.416 |
| XAUUSD | 512 | 0.508 | +0.167 | 0.074 | 0.500 |
| EURUSD | 469 | 0.501 | −0.143 | −0.007 | 0.356 |
| GBPUSD | 494 | 0.470 | −0.161 | −1.240 | 0.336 |

- AUC spread 0.470–0.524; the *best* symbol (BTC 0.524) still barely separates.
- **GBPUSD is an anti-signal** (AUC 0.470, slope −1.24) — an independent outlier
  by attribution. But dropping it does **not** rescue the gate (the remainder is
  still AUC ≈ 0.51), so there is no principled single-drop that reaches PRIMARY.
  Per §0/§10 the run of record stays full-6; curating is not permitted to
  manufacture a pass, and here it wouldn't even work.
- XAU's +0.167 mean_r pairs with AUC 0.508 (no *ranking* skill) — that is gold's
  raw drift in the test window, not model edge.
- Reliability slopes scatter −1.24 → 1.70: the miscalibration is **systemic**,
  not a lone outlier propping up / dragging down the pass.

## Decision-tree diagnostic (§6 — swap the rule, check capacity)

Before invoking §11, the §6 tree mandates swapping the *rule* (not the model)
and permits a coarse capacity grid on fixed folds. All on the full basket,
holdout calibration, one fixed purged split; every cell logged as an honest
trial. Majors-donchian included as the Phase 2 anchor.

| cell | capacity | n | AUC | G1 ΔSh | G1 p | G1 | G5 |
|---|---|---|---|---|---|---|---|
| **majors3 / donchian** (Phase 2 anchor) | d4·15 | 3247 | **0.543** | 1.531 | **0.001** | **PASS** | . |
| full6 / donchian | d4·15 | 5431 | 0.532 | 0.201 | 0.299 | FAIL | . |
| full6 / donchian | d6·31 | 5431 | 0.530 | 0.201 | 0.299 | FAIL | . |
| full6 / pullback | d4·15 | 4801 | 0.507 | 0.688 | 0.075 | FAIL | . |
| full6 / both | d4·15 | 10232 | 0.502 | 0.383 | 0.150 | FAIL | . |
| full6 / both | d6·31 | 10232 | 0.506 | 0.435 | 0.121 | FAIL | . |

Per-symbol AUC, full6/donchian: EURUSD 0.546, USDJPY 0.545, XAU 0.536,
BTC 0.527, GBPUSD 0.489, **US30 0.478**. Per-symbol AUC, full6/pullback: all
0.45–0.55 (no structure).

**Reading (this is the whole verdict):**
1. **Majors-donchian edge is real and replicated** — G1 p=0.001, AUC 0.543,
   matching Phase 2. The architecture's core claim holds *in that slice*.
2. **The htf_pullback family has no recoverable edge** — AUC 0.507, flat p̂.
   Swapping to it (per §6) makes things *worse*, not better.
3. **XAU/US30/BTC do not share the majors' structure** on donchian — pooling
   them drops AUC to 0.532 and kills G1 (US30 is sub-chance at 0.478).
4. **Both families pooled across all six → AUC ≈ 0.50, slope ≈ 0.04** — the
   textbook §11 "flat p̂ everywhere" signature.
5. **Capacity is not the lever** — d4·15 (the §1/§6 tiny prior) and d6·31 (the
   §5 upper bound) are indistinguishable (0.502 vs 0.506). Adding features is
   forbidden (§6) and MDA already shows the features carry no edge. **The tree
   is exhausted.**

## Per-stream G1 screen — does any addition clear G1 alone?

The recommendation below is "make each added stream clear G1 *on its own* before
pooling." Screening the additions individually (headline holdout, d4·15, one
fixed split each; `scripts/_stream_screen.py`) makes that concrete:

| stream | n | AUC | slope | G1 ΔSh | G1 p | G1 |
|---|---|---|---|---|---|---|
| **majors3 × donchian** (anchor) | 3247 | **0.543** | 0.556 | 1.043 | **0.016** | **PASS** |
| XAUUSD × donchian | 834 | **0.448** | −3.27 | −3.540 | 1.000 | FAIL |
| US30 × donchian | 671 | 0.512 | 1.330 | −0.430 | 0.646 | FAIL |
| BTCUSD × donchian | 679 | **0.417** | −1.923 | −0.406 | 0.670 | FAIL |
| majors3 × pullback | 2577 | **0.474** | −0.565 | 1.622 | 0.022 | PASS\* |

- **Every added instrument on donchian is edgeless.** US30 ~chance (0.512);
  **XAU (0.448) and BTC (0.417) are actively anti-signal** — the filter *hurts*
  (XAU ΔSharpe −3.5, p=1.000). These streams have nothing to pool.
- **\* majors × pullback's G1 "pass" is spurious.** AUC 0.474 is *below chance* —
  there is **no ranking edge**. The positive ΔSharpe comes from the degenerate
  case §0 warns about: filtering a *losing* primary (pullback take-all bleeds
  post-cost) cuts exposure and mechanically lifts the variance-scaled return,
  without any conditional skill. G1 read **without** a ranking check (AUC/G5) is
  fooled here. Contrast the anchor: majors × donchian passes G1 *and* has genuine
  ranking edge (AUC 0.543). **The pullback family is not promotable.**
- Takeaway: the edge is **majors × donchian specifically**. Pooling any of the
  additions in is what collapsed the pooled p̂ to chance. This is the per-stream
  substantiation of the §11 rejection.

## Honest trial count

**11 trials at the time the full-basket gate table above was computed** (its
G2 DSR used n_trials=11). The running session total after the follow-up work
(majors-donchian scoped run of record, the G5 calibration sweep, and this
per-stream screen) is **22** — see `MAJORS_DONCHIAN_RESULTS.md`. All from
`.triage/registry.sqlite`, monotonically accumulating; nothing was reset.

> **Important honesty note.** `.triage/` is gitignored and **did not exist in
> this checkout** — no bar store, no registry. The Phase 1/2 documents cite
> honest counts of 8 and 10, but that registry was never committed and is not
> present here. This run therefore started the registry from zero; **11 reflects
> only this session's trials up to the full-basket run of record** (4 basket-
> comparison cells + 6 decision-tree diagnostic cells + 1 run-of-record). The
> registry was **not** deleted or reset to lower the count (§4) — it was
> genuinely absent. DSR/PBO are computed against this honest count; if the intent
> is to carry Phase 1/2 history, the count should be ≥ 10 higher and G2 becomes
> *harder*, not easier.

## Recommendation / Verdict

**Reject the full-basket Phase 3 configuration. Do not promote.** This is a
§11 falsification outcome, read cleanly:

- **PRIMARY not reached.** On the mandated full-6 / holdout evaluation, G1 fails
  (p=0.227) with flat p̂ (slope 0.15, AUC 0.51) and negative post-cost EV at
  every τ. G5 also fails. The meta-layer adds no uplift over take-all — and
  take-all itself loses money on this basket post-cost.
- **§11 signature observed:** "flat p̂ everywhere across multiple sensible
  primaries → H4 setups in *this basket* lack ML-recoverable conditional
  structure at these horizons; the meta-labeling reframe is rejected, not just
  a model." Confirmed for the pooled basket, with the crucial qualifier that
  the **majors-donchian slice is the exception** — it retains real edge (G1
  p=0.001) but is exactly the Phase 2 result, not a Phase 3 advance.
- **Not a plumbing failure.** Leakage battery green, parity green, labeler
  reproduces Phase 1/2 counts. The red is a property of the market/rule/basket,
  which is precisely what the rejection-rigor architecture is built to surface.

### The single most useful decision for the morning

The edge is **instrument- and rule-specific, not basket-wide.** The honest next
move is **not** more search on this basket (that only degrades DSR). It is a
scoping decision a human should make:

> **Narrow the run of record to where edge demonstrably exists (majors ×
> donchian), and treat XAU/US30/BTC and the pullback family as separate
> hypotheses that must each clear G1 *on their own* before pooling** — rather
> than expanding the basket first and hoping pooling rescues them. The Phase 2
> majors-donchian result (G1 p=0.001, replicated tonight) is the only
> promotable signal in hand; everything added in Phase 3 dilutes it.

Alternatively, if the basket is fixed by mandate, the §11 conclusion stands: the
H4 triple-barrier meta-labeling reframe does not recover edge across this
6-instrument, 2-rule universe, and the design should be shelved for this scope
rather than patched.

## Notes / caveats

- **Cost model unchanged.** No `spread` column was present in the MT5 ingest, so
  `gates.COST_BY_SYMBOL` was left at its pre-registered per-symbol values (the
  one sanctioned cost update was not triggered). Even at these conservative
  costs the primary is unprofitable; a *lower* cost would not flip a negative-
  AUC ranking into uplift.
- **CPCV under heterogeneous history.** US30 (from 2019) and BTC (from 2018)
  concentrate in later time groups, so early CPCV folds train on majors and see
  unseen-category codes for the late instruments. This is an honest limitation
  of pooling instruments with different histories; it makes early-fold
  prediction on the late instruments weaker, but it is not the cause of the flat
  p̂ (the single-split headline, which does not have this issue, is equally
  flat).
- **`cs_rsi_rank` kept by MDA** but its contribution is marginal (mean
  +0.0022); it does not change the verdict.
- Reproduce: `scripts/_ingest_m1_all.py` → `scripts/_backfill_events_all.py` →
  `scripts/run_basket_comparison.py` → `scripts/_diag.py` →
  `scripts/_run_of_record.py` (writes `.triage/run_of_record_summary.json`).
