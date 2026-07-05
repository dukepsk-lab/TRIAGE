# Audit Report — Phase Results Validation

**Date:** 2026-07-05
**Scope:** Independent verification of every quantitative claim in
`PHASE0_NOTES.md`, `PHASE1_DATA_AUDIT.md`, `PHASE1_RESULTS.md`, and
`PHASE2_RESULTS.md`.

**Method:** Each claim was re-derived from ground truth (the bar DB, the
events DB, the test suite, and fresh model training) — *not* by re-running
the orchestration scripts. Numbers that reproduce only via the orchestration
but not independently are flagged.

---

## Summary verdict

**The phase results are substantively accurate.** Every load-bearing claim
that drove the sign-off decisions reproduces exactly. Two documentation
defects were found (neither affects the decisions):

| # | Finding | Severity | Affects decision? |
|---|---|---|---|
| 1 | `n_train` over-reported (pre-carve count, not actual fit rows) | Minor doc defect | No — AUC/gates re-derive exactly |
| 2 | EV-vs-τ curve described as "monotonic" but has local drops | Characterization error | No — overall trend still rises |

No fabricated numbers, no silent failures, no leakage. The G1 p-values
(the decisions' hinge) reproduce to 3 decimals.

---

## Verified accurate (re-derived independently)

### Phase 0 (`PHASE0_NOTES.md`)

| Claim | Verification | Result |
|---|---|---|
| 81/81 tests green | `uv run pytest` | ✓ 81 passed |
| Barriers frozen 1.5/1.0/24 | `from triage.labeling import TP_MULT, SL_MULT, VERTICAL_BARS` | ✓ exact |
| CPCV: 15 splits, 5 paths | `cpcv_splits()` + `backtest_paths(6,2)` | ✓ 15 / 5 |
| Same-bar→SL tie-break tested | `test_same_bar_ambiguity_resolves_to_sl` | ✓ passing |

### Phase 1 data (`PHASE1_DATA_AUDIT.md`)

| Claim | Verification (direct DB query) | Result |
|---|---|---|
| EURUSD D1=3547, H4=18450, M1=4,259,180 | `SELECT COUNT(*) FROM bars` | ✓ exact |
| Span 2015-01-02 → 2026-07-03 (11.5y) | `MIN/MAX(open_time)` | ✓ exact |
| Timestamps are UTC | H4 opens at 04:00/08:00/12:00/... UTC | ✓ verified |

### Phase 1 results (`PHASE1_RESULTS.md`)

| Claim | Independent re-derivation | Result |
|---|---|---|
| 1059 events, base rate 0.374 | events DB query | ✓ exact |
| Purged train/test = 740 / 318 | reproduced split | ✓ exact |
| AUC = 0.5593 | via `train_meta_model` | ✓ exact match |
| Brier = 0.2838 | via `train_meta_model` | ✓ exact match |
| **G1: ΔSharpe=1.232, p=0.059** | fresh `block_bootstrap_uplift` | ✓ **exact** |
| Shuffled-label AUCs [0.499, 0.517, 0.507] | fresh `shuffled_label_auc` | ✓ exact |
| Label policy: 0 violations | `label=1 iff outcome='tp'` query | ✓ 0 violations |
| r_realized: TP=+1.5, SL=-1.0 | aggregate query | ✓ exact |
| §3 parity gate passes | `test_real_data_parity` | ✓ passing |

### Phase 2 results (`PHASE2_RESULTS.md`)

| Claim | Independent re-derivation | Result |
|---|---|---|
| 3247 pooled events, base 0.379 | events DB query | ✓ exact |
| Per-symbol: EURUSD 1059, GBPUSD 1112, USDJPY 1076 | DB query | ✓ exact |
| Purged train/test = 2270 / 975 | reproduced split | ✓ exact |
| AUC = 0.5591, Brier = 0.2829 | via `train_meta_model` | ✓ exact |
| **G1: ΔSharpe=1.182, p=0.004** | fresh `block_bootstrap_uplift` | ✓ **exact** |
| Per-symbol AUC: EURUSD 0.5643, GBPUSD 0.5923, USDJPY 0.5363 | fresh per-symbol `roc_auc_score` | ✓ exact |
| AUC spread 0.056 → no symbol dominates | `0.5923 - 0.5363` | ✓ exact |
| Shuffled-label AUCs [0.494, 0.510, 0.493] | fresh harness | ✓ exact |
| Cumulative trial count = 10 | `len(TrialRegistry)` | ✓ exact (8 Phase 1 + 2 Phase 2) |

---

## Discrepancies found

### Discrepancy 1 — `n_train` over-reported (both Phase 1 & Phase 2 docs)

**The claim:** Phase 1 reports `n_train=740`; Phase 2 reports `n_train=2270`.

**The reality:** `train_meta_model` carves out a validation fold (last 25%
of the training set) before fitting. The model actually trains on:
- Phase 1: 740 − 185 (val) = **555 fit rows**, not 740.
- Phase 2: 2270 − 567 (val) = **1703 fit rows**, not 2270.

The reported `n_train` is the *pre-carve-out* count. The overstatement is
~33% in both phases.

**Impact:** None on the decisions. AUC, Brier, and all gates re-derive
*exactly* via `train_meta_model` regardless of how n_train is labeled —
the model genuinely saw 555/1703 rows and produced the reported metrics
from that. But the documentation misstates training size, which matters
for anyone assessing whether n is adequate (e.g., the §6 "~2k events"
calibration threshold is being evaluated against the wrong denominator).

**Fix:** Report both numbers — `n_train_fit` (actual model rows) and
`n_train_pre_val` (pre-carve). The `MetaModelResult.n_train` field should
return the fit-subset size.

### Discrepancy 2 — EV-vs-τ curve is NOT monotonic (Phase 1)

**The claim:** `PHASE1_RESULTS.md` says the EV-vs-τ curve is
"monotonically increasing."

**The reality:** Independent re-derivation shows local drops:

| τ | n_trades | mean R (after cost) |
|---|---|---|
| 0.30 | 165 | +0.0735 |
| 0.375 | 148 | +0.0324 ← drop |
| 0.45 | 128 | +0.0633 |
| 0.525 | 103 | +0.0398 ← drop |
| 0.60 | 92 | +0.0874 |
| 0.675 | 74 | +0.1512 |
| 0.70 | 64 | +0.1709 |

The *overall trend* rises (0.07 → 0.17), which is the economically
meaningful signal. But "monotonic" is wrong — the curve is noisy at
small per-bin n (148, 103 trades). The economic content (higher-τ
trades pay more on average) survives; the precise characterization does
not.

**Impact:** None on the Phase 1 decision (the recommendation hinged on
G1, not on monotonicity). But the doc overstates the cleanliness of the
signal.

**Fix:** Reword to "rising overall trend (noisy at small per-bin n)"
and show the actual curve with the drops visible (the table already
shows them — only the prose word "monotonic" is wrong).

---

## Things explicitly checked and found clean

- **No look-ahead leakage.** Every event's feature snapshot carries
  `max_source_close_time ≤ t0` (0 violations across 3247 events). The
  shuffled-label harness returns ≈0.50 on both phases.
- **Label-policy integrity (§4).** `label = 1 iff outcome = 'tp'` —
  zero violations. SL and vertical both → 0.
- **r_realized geometry.** TP exits = +1.5R exactly, SL exits = -1.0R
  exactly (frozen barriers honored). Verticals are mark-to-market
  (range -0.699 to +1.388), not forced to 0.
- **CPCV topology (§7).** 15 splits, 5 backtest paths (1-factorization
  of K₆ verified).
- **§3 parity gate.** Vectorized backfill ≡ bar-by-bar replay on real
  EURUSD — passing.
- **Trial registry honesty.** 10 cumulative trials (8 + 2), matching
  the docs. No silent resets.
- **Per-symbol attribution.** AUC spread 0.056 — no single symbol
  carries the pooled result. The "pooling is justified" claim holds.

---

## Recommendation

Both discrepancies are documentation defects, not result defects. The
underlying numbers are sound and the sign-off decisions stand on
verified ground. Both fixes have been applied in this audit pass:

1. **Fixed `MetaModelResult.n_train`** to report the actual fit-subset
   size (post val carve-out), and added an `n_val` field. Verified:
   Phase 1 now reports `n_train=555, n_val=185` (was `n_train=740`);
   AUC/Brier unchanged at 0.5593 / 0.2838.
2. **Softened "monotonic"** to "rising overall trend (noisy at small
   per-bin n)" in `PHASE1_RESULTS.md`, with the local drops noted.

Neither fix changed any metric, gate result, or decision. 81/81 tests
still green.

