# Paper Harness — Majors × Donchian, Filtered, Fixed Sizing

**Status: EDGE DEMONSTRATED end-to-end.** The promotable slice's G1 uplift
(docs/MAJORS_DONCHIAN_RESULTS.md) is materialized here as a positive OOS paper
equity curve: the naked donchian rule on the majors **loses** money post-cost;
the meta-filter turns it **profitable**. This is the "precision dial, not an
alpha source" thesis (§0) shown as PnL.

Deliverable: `src/triage/paper.py` (frozen `PaperStrategy` + §8-overlay
simulator) and `scripts/run_paper_majors_donchian.py`. The **exact** `decide()`
that drives this simulation is the one a live loop would call — no train/serve
skew by construction (single source of truth: `PaperStrategy` reuses the fitted
booster + calibrator captured from `train_meta_model`, it does not re-implement
them).

## Setup

- Slice: EURUSD/GBPUSD/USDJPY × `donchian20` (the only slice with replicated
  edge; §11 rejects the full basket, and every other stream fails G1 alone).
- Model: the promotable config — holdout (isotonic) calibration, d4·15,
  Tier 1–3 + `cs_atr_pct_rank`. Frozen after fit on the purged training period.
- **Sizing: fixed** (rung 2). τ0 = 0.466 (`(1+c)/2.5 + 0.05`, c=0.04). Every
  taken trade risks a constant `base_risk` (1% here). Probability-laddered
  sizing (rung 3) is **not** used — G5 has not cleared, so the probabilities are
  not trusted for sizing (§8).
- Execution overlays (§8): per-symbol single open position; aggregate open risk
  ≤ 3× base_risk. These make the paper book realistic vs the labeler (which
  labels every trigger).
- Evaluation: OOS **simulation** on the held-out test fold (975 candidate
  events). A dry-run of the live decision path — **not** forward live results.

## Result (held-out fold)

| metric | filtered (fixed size) | take-all baseline |
|---|---|---|
| candidates seen | 975 | 975 |
| trades taken | **262** | 975 |
| **total R (after cost)** | **+13.05** | **−91.06** |
| mean R / trade | +0.0498 | −0.0934 |
| win rate | 0.435 | — |
| Sharpe / trade (ann.) | **+0.642** | −1.237 |
| profit factor | 1.086 | — |
| max drawdown @1% risk | 18.1% | — |
| final equity @1% risk | **1.117** (+11.7%) | — |

- **The naked rule is a loser** (−91R, Sharpe −1.24): donchian take-all on majors
  bleeds cost, as it should — the primary is deliberately loose/recall-heavy
  (§0). **The meta-filter flips it to +13R (Sharpe +0.64, PF 1.09, +11.7% equity
  at 1% risk).** That is the G1 uplift, as PnL.
- The filter takes **27%** of candidates (262/975) — high conviction only.

## Per-symbol (taken trades)

| symbol | n | total R | mean R | win rate |
|---|---|---|---|---|
| USDJPY | 70 | +9.23 | +0.132 | 0.471 |
| EURUSD | 87 | +7.67 | +0.088 | 0.448 |
| GBPUSD | 105 | −3.85 | −0.037 | 0.400 |

EURUSD + USDJPY carry the book; GBPUSD is a mild drag (consistent with its
weaker attribution — AUC/slope in `MAJORS_DONCHIAN_RESULTS.md`). It is *not*
dropped: it does not independently disqualify, and curating it out to lift the
number is the §0/§10 anti-pattern.

## Honest caveats

- **OOS simulation, not live.** These are held-out-fold results (already looked
  at during gating), run through the live decision path — a dry-run, not forward
  paper trading. Forward data is the real test (§10 Phase 4).
- **Model draw.** LightGBM multi-threaded training is nondeterministic; this
  frozen model correlates ~0.88 with the gate-evaluated draw. Sign and magnitude
  (−91R → +13R) are robust across draws; the third-decimal metrics are not.
- **Modest edge.** PF 1.086 and 18% max DD (at 1% risk) are real but thin. This
  is a filter that makes a losing rule slightly profitable, not a high-Sharpe
  system. It clears G1; it does not clear the full promotion stack (G2/G4/G5).
- **Fixed sizing only.** Rung 3 (probability sizing) stays parked until G5 clears
  on live data — the calibration sweep showed it is data-bound at this n.

## Artifacts

- `.triage/majors_donchian_strategy.pkl` — the frozen, reload-tested strategy
  (`decide()` verified after a pickle round-trip).
- `.triage/paper_trades.csv`, `.triage/paper_equity.csv`,
  `.triage/paper_majors_donchian_summary.json`.

## Verdict

**Edge found and demonstrated: the majors × donchian meta-filter is a real,
G1-validated, positive-expectancy strategy under fixed sizing** — the honest
promotable result of this whole investigation. Recommended path: forward paper
with fixed sizing + the §10 Phase 4 monitors (calibration CUSUM first); revisit
probability sizing only when live data closes G5. Do not re-expand the basket
(§11) and do not curate GBPUSD to pad the number.
