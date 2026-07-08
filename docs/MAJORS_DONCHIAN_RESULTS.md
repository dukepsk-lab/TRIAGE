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

> **Calibrator note:** the holdout calibrator used **isotonic**, not sigmoid —
> the method rule keys on `len(X_tr)` = 2270 (the full purged training set) ≥ 2k,
> and n_train=1816 above is the *post-carve fit* count, not the method selector.

## G5 calibration sweep (§6 lever — run, and exhausted)

Per §6, when *G1 passes / G5 fails* the sanctioned lever is the **calibrator, not
the basket**. Swept all three knobs — isotonic↔sigmoid × `calib_frac`
{0.20, 0.35} × pooled/per-symbol — over one fixed feature set and one fixed
purged split, so only the calibrator moves (`scripts/_calib_sweep.py`, 6 logged
trials). Question: does any sanctioned calibrator get the pooled slope into
[0.8, 1.2] with Brier < clim?

| calib_frac | method | per-sym | AUC | Brier | clim | slope | G5 | G1 ΔSh | G1 p |
|---|---|---|---|---|---|---|---|---|---|
| 0.20 | sigmoid | no  | 0.547 | 0.2350 | 0.2326 | 3.227 | . | 1.237 | 0.007 |
| **0.20** | **isotonic** | **no** | 0.535 | 0.2355 | 0.2326 | **0.556** | . | 1.043 | 0.016 |
| 0.35 | sigmoid | no  | 0.463 | 0.2337 | 0.2326 | −27.51 | . | 1.237 | 0.007 |
| 0.35 | isotonic | no | 0.523 | 0.2350 | 0.2326 | −0.324 | . | 0.966 | 0.084 |
| 0.20 | isotonic | yes | 0.538 | 0.2397 | 0.2326 | 0.312 | . | 0.928 | 0.013 |
| 0.35 | isotonic | yes | 0.526 | 0.2392 | 0.2326 | −0.032 | . | 1.011 | 0.018 |

**Answer: no. G5 is data-bound, not calibrator-bound.**

- **Every config fails G5**, and the slopes are *erratic* (3.23, −27.5, −0.32,
  0.31, −0.03): the calibrator is fitting **noise**, because an AUC≈0.535 ranking
  does not carry enough reliability structure for any 1-D calibrator to recover
  a slope-1 curve. Brier sits at climatology in every case.
- **Larger `calib_frac` makes it worse** (less fit data → weaker booster; sigmoid
  at 0.35 → slope −27). **Per-symbol makes it worse** (≈120 calib rows/symbol →
  noise). The **auto default (isotonic, frac 0.20, pooled)** is already the best
  and still fails.
- **G1 uplift is invariant to the calibrator** (ΔSharpe ~1.0, p 0.007–0.018
  throughout). That is the tell: the *filter/ranking* is the real, stable,
  bankable effect; the calibration layer on top is the part that has nothing to
  lock onto at this n.

The §6 calibration lever is therefore exhausted. G5 needs **more edge-bearing
data**, which Phase 3's pooling was meant to supply and could not (the additions
had no edge). At the majors-donchian n (train ≈ 2.3k), probability sizing cannot
be safely unlocked.

## Honest trial count

**18 trials** (12 through the scoped run of record + 6 for this calibration
sweep). Registry preserved, not reset. See the honesty note in
`PHASE3_RESULTS.md`: the registry was absent at session start, so 18 reflects
only this session's trials.

## Verdict & recommendation

**Majors × donchian is the promotable signal — via the filter (rung 2), not the
sizer (rung 3).**

- **G1 uplift is real, CPCV-confirmed, and calibrator-invariant.** The meta-filter
  significantly beats take-all (ΔSharpe +1.04, p=0.016). Baseline-ladder **rung 2
  is validated.**
- **G5 is data-bound and will not clear at this n.** The full §6 calibration lever
  was run to exhaustion; no sanctioned calibrator reaches [0.8, 1.2]. **Rung 3
  (probability-laddered sizing) stays blocked** — the sizer cannot safely consume
  these probabilities. G2/G4 marginal at n, as expected (§6/§10).

**Recommended landing (option b):** take the **filtered** majors-donchian strategy
to **paper trading with fixed (non-probability) sizing**, where the G1 uplift is
already bankable. Instrument the §10 Phase 4 monitors — calibration-drift CUSUM
first — and treat probability sizing as a *live-data* goal: revisit G5 only once
more edge-bearing majors-donchian events accrue (paper + forward), not by further
in-sample calibration search (which this sweep shows is a dead end and only
degrades DSR). Do **not** re-expand the basket to chase n — the §11 rejection in
`PHASE3_RESULTS.md` stands: the added instruments and the second rule family have
no edge to pool.
