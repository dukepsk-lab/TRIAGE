# TRIAGE

Triple-barrier meta-labeling on FX, gold, an index and crypto, built to find out whether an ML filter on top of a simple rule adds anything once validation is done properly. The short answer so far: on the full basket, it doesn't.

## What it does

1. A rule-based primary layer proposes trades (Donchian breakout, and a higher-timeframe pullback rule).
2. Each candidate is labeled by first touch on M1 bars against fixed barriers (take profit 1.5, stop 1.0, vertical 24 bars). Label is 1 if take profit is hit first.
3. A gradient-boosted meta-model estimates the probability that a candidate wins. It is a filter on the rule, not a source of signals.
4. Results go through gates: uplift over the primary rule, calibration, and a shuffled-label leakage check.

Every tuned parameter is logged to a trial registry so the deflated Sharpe / PBO numbers use the real trial count. Barriers and label policy are fixed constants, not tuned.

## Results

Write-ups are in `docs/`.

| phase | scope | outcome |
|---|---|---|
| 1 | EURUSD, Donchian-20, 1,059 events | shuffled-label check passed; OOS AUC 0.559 on 318 test events ([PHASE1_RESULTS](docs/PHASE1_RESULTS.md)) |
| 2 | EURUSD, GBPUSD, USDJPY pooled, 3,247 events | pooled OOS AUC 0.559, per-symbol mean R after cost still negative ([PHASE2_RESULTS](docs/PHASE2_RESULTS.md)) |
| 3 | six symbols, two rule families, 10,232 events | rejected: the uplift gate fails and calibration is flat ([PHASE3_RESULTS](docs/PHASE3_RESULTS.md)) |

Phase 3 also compared a curated 3-symbol basket against the full 6 with two calibration methods ([BASKET_COMPARISON](docs/BASKET_COMPARISON.md)). AUC stays near 0.51 to 0.53 in every cell, and the per-symbol reliability slope changes sign between calibration methods, which points to the calibrator fitting noise.

I'm keeping the Phase 3 result as is. The Phase 2 gain on three majors did not carry over when the basket was widened.

## Layout

```
src/triage/
  labeling/     first-touch triple-barrier labels
  rules/        primary rules
  features/     feature tiers
  models/       meta-model
  validation/   purged train split, CPCV splits
  gates.py      pass/fail gates, PBO from CPCV
  leakage.py    shuffled-label check
  registry/     trial registry
  uniqueness.py sample uniqueness weights
scripts/        phase runners (run_phase1.py ... run_phase3.py)
tests/          unit tests and synthetic data
docs/           architecture, per-phase results, audit notes
```

## Running it

```
pip install -e . pytest
python -m pytest -q
```

Tests that need the local bar database are skipped when it is missing. Data ingest (`scripts/ingest_csv.py`, `scripts/_ingest_m1_all.py`) reads MT5 history and needs a Windows terminal; the bar store and trial registry are gitignored. `uv.lock` is included if you prefer uv.

Design notes and the reasoning behind the sample-size budget are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
