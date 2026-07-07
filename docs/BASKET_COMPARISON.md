# Basket Comparison — fix the calibrator vs. drop the symbol

The head-to-head that isolates the two variables behind the G5 calibration
failure: **basket composition** (full 6 vs curated 3) × **calibration method**
(internal CV vs dedicated purged holdout fold). Identical labels, identical
features (Tier 1–3 + all Tier 4), identical model params across all four cells,
so the only things that move are basket and calibrator. Produced by
`scripts/run_basket_comparison.py` over the 10,232-event pooled set.

Curated-3 here is the pre-declared `[EURUSD, USDJPY, BTCUSD]` set baked into the
harness — **not** a set chosen by this run to make a gate pass.

## The 2×2

G5 target: **Brier < clim AND reliability slope ∈ [0.8, 1.2]**.

| basket | calib | n | AUC | Brier | clim | slope | G1 | G2 | G3 | G4 | G5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full6 | cv | 10232 | 0.5113 | 0.2415 | 0.2417 | 0.550 | ✅ | . | ✅ | . | . |
| **full6** | **holdout** | 10232 | 0.5152 | 0.2421 | 0.2417 | 0.013 | . | . | ✅ | . | . |
| curated3 | cv | 5567 | 0.5133 | 0.2414 | 0.2400 | 0.342 | ✅ | . | ✅ | . | . |
| curated3 | holdout | 5567 | 0.5262 | 0.2438 | 0.2400 | 0.617 | ✅ | . | ✅ | . | . |

(`✅` = gate passes; `.` = fails. G3 passes everywhere but is vacuous — see
`PHASE3_RESULTS.md`: both filtered and primary lose, filter loses less.)

## Per-symbol reliability slope (target [0.8, 1.2])

```
full6/cv:        BTCUSD=0.528  EURUSD=0.728  GBPUSD=-1.386  US30=0.616  USDJPY=1.132  XAUUSD=-0.397
full6/holdout:   BTCUSD=0.920  EURUSD=-0.263 GBPUSD=-0.772  US30=0.488  USDJPY=2.178  XAUUSD=-1.470
curated3/cv:     BTCUSD=0.027  EURUSD=-0.048 USDJPY=0.590
curated3/holdout:BTCUSD=0.440  EURUSD=0.937  USDJPY=0.285
```

No symbol sits stably in the [0.8, 1.2] band across cells. Slopes swing sign
between calibration methods (EURUSD +0.728 → −0.263; XAU −0.397 → −1.470),
which is the fingerprint of a calibrator fitting **noise** — there is no stable
reliability curve to recover because the underlying ranking (AUC ≈ 0.51) is
barely above chance.

## Verdict

**Neither lever fixes G5, because G5 is not the binding problem — the binding
problem is that the model does not separate on this basket.**

- **Did full6+holdout clear G5?** **No.** slope=0.013 (essentially flat), Brier
  ≥ clim. And it *also* fails G1 under holdout. So the question the harness was
  built to answer — "was symbol-dropping unnecessary because the calibrator was
  the real fix?" — is moot: the calibrator has nothing to calibrate.
- **Did a per-symbol outlier justify a documented drop?** **No principled drop
  achieves PRIMARY.** GBPUSD is independently an outlier (AUC 0.470, slope
  −1.24 to −1.39) and *would* be the honest candidate if a drop could rescue the
  gate — but it can't: curated3 (which already excludes GBPUSD) still fails G5
  under both calibrations, and full6 fails G1 regardless. Dropping symbols here
  would be gate-chasing that doesn't even reach the gate, which §0/§10 forbid.
- **Calibration method is second-order.** holdout vs cv barely moves AUC
  (0.511 ↔ 0.515 on full6) and swings the slope around zero. The §6 "fit on a
  purged calibration fold" prescription works as designed — but a well-built
  calibrator on a chance-level ranking still yields a flat reliability curve.

### What this means for the Phase 3 decision

The 2×2 was designed to distinguish "the calibrator was the fix" from "a
symbol genuinely had to go." It returns a **third answer**: neither — the
meta-model has **no ranking edge to calibrate or attribute** on the full basket.
That routes back to the §6 decision tree (swap the rule, not the model), which
was then run in `scripts/_diag.py` and reached the §11 falsification recorded in
`docs/PHASE3_RESULTS.md`: the only slice with real edge is majors × donchian
(the Phase 2 result), and the Phase 3 expansion — second rule family +
XAU/US30/BTC — dilutes p̂ to chance.

**Bottom line: symbol-dropping is neither necessary nor sufficient. The fix does
not belong in the basket *or* the calibrator — there is no edge on this basket
for either to work with.**
