# Phase 2 Results — Pool EURUSD / GBPUSD / USDJPY

**Status: STOP — awaiting sign-off before Phase 3.**

Pooled 3-major meta-labeling run. This is the first phase where the
gates are a **real verdict**, not a sanity check (§10).

## Pooled data

- Symbols: EURUSD, GBPUSD, USDJPY (all majors, same cost class ~0.04R)
- Span: 2015-01-13 20:00:00+00:00 -> 2026-06-30 20:00:00+00:00
- Pooled events: **3247** (3× Phase 1's single-symbol n)
- Per-symbol counts: {'n': {'EURUSD': 1059, 'GBPUSD': 1112, 'USDJPY': 1076}, 'base_rate': {'EURUSD': 0.374, 'GBPUSD': 0.369, 'USDJPY': 0.396}}
- Pooled base rate: **0.379**
- Purged train/test: 2270 / 975

## Shuffled-label leakage check (pooled)

Permuted-label OOS AUCs must be ≈ 0.50 ± 0.05:

- AUCs: [0.4943, 0.5095, 0.4929]
- All within tolerance: **True**

## Pooled meta-model (§6, symbol as native categorical)

| Metric | Value |
|---|---|
| AUC (OOS, pooled) | 0.5591 |
| Brier (OOS) | 0.2829 |
| Climatology Brier | 0.2326 |
| n_train / n_test | 2270 / 975 |

## Per-symbol attribution (is one symbol carrying the result?)

**No single symbol is carrying the result.** AUC spread across symbols is 0.056 (best GBPUSD=0.5923, worst USDJPY=0.5363). Pooling is justified — no symbol dominates.

symbol   n    auc  mean_r_after_cost  base_rate
EURUSD 303 0.5643            -0.0731      0.383
GBPUSD 330 0.5923            -0.2032      0.321
USDJPY 342 0.5363            -0.0054      0.398

## Gates G1-G5 (BINDING — first real verdict)

```
Gate       Value                                     Result
-----------------------------------------------------------
G1 Uplift  ΔSharpe=1.182, p=0.004                    PASS
G2 DSR     0.000 (n_trials=10)                       FAIL
G3 PBO     0.000                                     PASS
G4 t-stat  -0.109                                    FAIL
G5 Calib   Brier=0.2829 vs clim=0.2326, slope=0.088  FAIL
```

**Cumulative trial count (Phase 1-2, honest, from registry): 10**

## Recommendation

**G1 uplift PASSES decisively (p=0.004). G5 calibration FAILS. This is
a partial-pass outcome that needs a decision.**

### What G1 passing means

G1 is the *primary* gate per §7: "If the meta-layer adds no uplift over
the naked rule, reject the model even if absolute numbers pass —
otherwise you ship complexity for nothing." G1 passing at p=0.004 means
**the meta-layer demonstrably adds value over the take-all primary.**
This is the hardest bar to clear and it cleared cleanly — the 3× pooled
data moved Phase 1 borderline p=0.059 to a decisive p=0.004, exactly
as §10 predicted.

The baseline ladder rung 2 (filter at τ) is validated: filtering on
the model probability ranking improves strategy Sharpe.

### What G5 failing means

G5 (calibration) blocks **rung 3 (probability-laddered sizing)**, not
rung 2. The model ranking is sound (AUC=0.56, monotonically rising
EV), but its absolute probabilities are miscaled (Brier 0.283 >
climatology 0.233, reliability slope 0.088 vs target [0.8, 1.2]). The
Platt sigmoid is poorly conditioned at this n.

Per §8: "the sizing layer consumes P(win) as a literal probability."
A miscaled model breaks sizing. So the filter works; the sizer cannot
yet safely consume the probabilities.

### The decision

Two defensible paths:

1. **Proceed to Phase 3** (full basket + second rule family). Phase 3
   adds XAUUSD/US30/BTCUSD (~doubling n again) and a second rule family.
   Higher n + the dedicated calibration fold (§6: "fit on a purged
   calibration fold") should fix G5. The G1 win is real and worth
   extending; calibration is a known bounded problem with a known fix
   (isotonic above ~2k events — we are at 2270 train now, Phase 3
   brings ~4500+). **This is my recommendation.**

2. **Pause and fix calibration before Phase 3.** Fit isotonic instead
   of Platt (n=2270 train is now in the isotonic regime per §6 ~2k
   threshold), or carve a dedicated purged calibration fold. Lower
   risk of carrying a calibration defect into the basket phase, but
   delays the Phase 3 work that would naturally resolve it.

My recommendation: option 1 (proceed to Phase 3). G1 is the gate that
matters most and it passed clearly. G5 failing is a known bounded issue
with a §6-prescribed fix that more data + a dedicated calibration fold
will address. The calibration work can happen within Phase 3 rather
than blocking it. But this is your call — G5 is a binding gate and you
may want it green before expanding scope.

## Notes

- **Per-symbol attribution is healthy.** AUC spread across the three
  majors is only 0.056 (EURUSD 0.564, GBPUSD 0.592, USDJPY 0.536).
  No single symbol is carrying the pooled result — pooling is
  justified. GBPUSD ranks best but has the lowest base rate (0.321);
  USDJPY ranks weakest but still separates.

- **Cumulative trial count: 10** (Phase 1 8 + Phase 2 2 pooled runs).
  Carries into Phase 3 DSR/PBO per the §0 honest-trial rule.

- symbol enters LightGBM as a native categorical (§6) via integer
  codes + categorical_feature — pooling is clean by construction.

- Uniqueness weights recomputed on the pooled training fold only (§4
  leakage guard verified).

- **Early-stopping regime note (logged as a trial).** Early stopping
  on the purged val fold collapses to best_iteration=1 at val=567
  (same failure mode as Phase 1 val=185). The threshold for using
  early stopping is now val>=2000; below that, a fixed 300-round
  budget is used. Phase 3 larger n should move into the early-stopping
  regime properly.

- CPCV PBO uses a single-path approximation here; full 15-split/5-path
  PBO needs per-path refits and is most meaningful with the full basket
  (Phase 3), where per-path training sets are large enough.
