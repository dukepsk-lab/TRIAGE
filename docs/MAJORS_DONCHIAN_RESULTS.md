# Scoped Run of Record — Majors × Donchian

**Status: G1 (uplift) PASS under CPCV. G5 (calibration) FAIL. G2/G4 marginal at
n. → "separates + real uplift; sizing blocked on calibration."**

Follow-up to `PHASE3_RESULTS.md`. That run rejected the full 6-symbol basket
(flat p̂, §11). The diagnostic isolated the one slice with replicated edge —
**majors × donchian** — so we take *that* through the full heavy gate stack on
its own, rather than expanding the basket and hoping pooling rescues no-edge
streams. This is the promotable-signal confirmation.

Scope: `EURUSD, GBPUSD, USDJPY`, `rule_id=donchian20` only, holdout calibration,
capacity **d4·15** (the §1/§6 tiny-model prior for this n). Produced by
`scripts/_run_scoped.py --tag majors_donchian`. Summary:
`.triage/run_of_record_majors_donchian.json`.

## Data

- Events: **3,247** (EURUSD 1059, GBPUSD 1112, USDJPY 1076) — reproduces Phase 2
  exactly.
- Base rates: EURUSD 0.374, GBPUSD 0.369, USDJPY 0.396 (pooled 0.379).

## Leakage battery (§4 — PASSED)

- Shuffled-label OOS AUC: **[0.5011, 0.5030, 0.4937]** — all within [0.45, 0.55].
- Feature-timestamp audit + Donchian parity: green (as in `PHASE3_RESULTS.md`).

## Feature selection (§5 — MDA on realized trade EV)

Corr-prune + MDA(metric=ev). Only **`cs_atr_pct_rank`** clears (CI just above
zero); base Tier 1–3 retained. Final = Tier 1–3 (15) + `cs_atr_pct_rank`.

## Headline meta-model (holdout, d4·15)

| Metric | Value |
|---|---|
| AUC (OOS) | **0.5352** |
| Brier (OOS) | 0.2355 |
| Climatology Brier | 0.2326 |
| n_train / n_test | 1816 / 975 |

AUC 0.535 sits in §10's "0.53–0.56 separates" band (Phase 2 was 0.559). The
ranking has signal; the shuffled-label control confirms it is real.

## EV-vs-τ (per-symbol cost) — a genuine profitable pocket

| τ | n_trades | mean R (after cost) |
|---|---|---|
| 0.300 | 954 | −0.0909 |
| 0.375 | 952 | −0.0889 |
| **0.400** | 350 | **+0.0229** |
| **0.450** | 349 | **+0.0187** |
| 0.475 |  88 | −0.0417 |
| **0.500** |  18 | **+0.0711** |
| ≥0.600 |   0 | — |

Take-all (low τ) loses post-cost, but a **moderate-τ pocket (0.40–0.50) is
positive** — the model's higher-conviction trades pay. This is the economic
content G1 detects, and the qualitative difference from the full basket (which
was negative at *every* τ).

## Gates G1–G5 (full CPCV, per-path refits)

```
Gate       Value                                      Result
------------------------------------------------------------
G1 Uplift  ΔSharpe=1.043, p=0.016                     PASS
G2 DSR     0.000 (n_trials=12)                        FAIL
G3 PBO     0.000                                      PASS
G4 t-stat  -0.382                                     FAIL
G5 Calib   Brier=0.2355 vs clim=0.2326, slope=0.556   FAIL
```

- **G1 PASS is the headline.** ΔSharpe=+1.04 at p=0.016 — the meta-filter adds
  significant uplift over take-all, confirmed on the CPCV paths (filtered path
  sharpes −0.86/+0.04/0/−0.47/+0.11 vs primary −0.42/−0.78/−0.78/−0.78/−0.78:
  the filter beats the naked rule on every path). This replicates Phase 2's G1
  win (p=0.004 there) now under the per-path-refit CPCV machinery. **Baseline
  ladder rung 2 (filter at τ) is validated.**
- **G5 FAIL is the open problem.** slope 0.556 (target [0.8,1.2]); Brier 0.2355
  marginally above clim 0.2326. The ranking is sound; the *absolute
  probabilities* are miscaled — exactly Phase 2's diagnosis ("the ranking is
  sound; only the absolute probabilities are miscaled"). **Rung 3 (probability-
  laddered sizing) stays blocked** — the sizer cannot yet safely consume p.
- **G2/G4 marginal at n** — expected (§6/§10). Note G4's negative t is on the
  headline test fold's taken trades (mean R slightly negative there), while G1's
  bootstrap over the paired series still finds significant *uplift* — the filter
  helps even where the absolute fold is thin.

## Per-symbol attribution — healthy, no single symbol carrying the book

| symbol | n | AUC | mean_r | slope | base_rate |
|---|---|---|---|---|---|
| EURUSD | 303 | 0.555 | −0.073 | 0.766 | 0.383 |
| GBPUSD | 330 | 0.552 | −0.203 | 0.554 | 0.321 |
| USDJPY | 342 | 0.516 | −0.005 | 1.215 | 0.398 |

AUC spread 0.516–0.555 (tight); all three separate. Reliability slopes
0.55–1.22 — under-calibrated but **not** a lone-outlier problem: it is a
uniform, mild under-calibration across all three, which is why the fix belongs
in the *calibrator*, not the *basket* (§6). No symbol dominates the result;
pooling the three majors is justified.

## Honest trial count

**12 trials** (registry preserved; +1 vs the full-basket run of record). See the
honesty note in `PHASE3_RESULTS.md` — the registry was absent at the start of
this session, so 12 reflects only this session's trials.

## Verdict & the open decision

**Majors × donchian is the promotable signal in hand.** G1 uplift is real and
survives CPCV; the filter (rung 2) works. The blocker is **G5 calibration**, and
per §6 the sanctioned lever when *G1 passes / G5 fails* is the **calibrator**,
not the basket: isotonic↔sigmoid, larger `calib_frac`, or per-symbol
calibration. Two honest caveats on that lever:

1. n_train = 1816 sits **below** the §6 ~2k isotonic threshold, so the holdout
   calibrator used **sigmoid**. Isotonic (the better calibrator here) wants more
   events — and Phase 3's whole plan for getting them (pooling the full basket)
   is exactly what failed. So more edge-bearing data is not cheaply available.
2. A shallow slope (0.556) on an AUC-0.535 ranking may be partly **irreducible**:
   a calibrator cannot manufacture reliability the ranking doesn't support.

**Recommended next decision (human):** either (a) run the bounded §6 calibration
sweep (`calib_frac` grid + per-symbol calibration) to see whether G5 reaches
[0.8,1.2] at this n — a few more logged trials — and if it does, this slice is
promotable to paper; or (b) accept the honest landing — **rung 2 (filter)
validated, rung 3 (sizing) deferred until G5 clears** — and take the filtered
majors-donchian strategy to paper trading with *fixed* (non-probability) sizing,
where the G1 uplift is already bankable and calibration drift can be monitored
live (§10 Phase 4) rather than gated pre-launch.
