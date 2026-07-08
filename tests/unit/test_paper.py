"""Paper harness tests — fixed-sizing strategy + §8 overlays."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from triage.paper import PaperStrategy


def _make_events(n=800, seed=0):
    rng = np.random.default_rng(seed)
    good = rng.normal(size=n)
    noise = rng.normal(size=n)
    p_win = 1.0 / (1.0 + np.exp(-1.6 * good))
    label = (rng.uniform(size=n) < p_win).astype(int)
    r = np.where(label == 1, 1.5, -1.0)
    # Non-overlapping 8h spans so the single-position overlay does not fire.
    t0 = [datetime(2020, 1, 1) + timedelta(hours=8 * i) for i in range(n)]
    return pd.DataFrame({
        "t0": t0, "t1": [t + timedelta(hours=4) for t in t0],
        "symbol": "EURUSD", "rule_id": "donchian20", "side": 1,
        "label": label, "r_realized": r, "good": good, "noise": noise,
    })


def test_fit_and_filter_improves_win_rate():
    df = _make_events()
    tr, te = df.iloc[:560].copy(), df.iloc[560:].copy()
    strat = PaperStrategy(features=["good", "noise"], base_risk=0.01,
                          params={"max_depth": 3, "num_leaves": 7, "learning_rate": 0.05})
    strat.fit(tr)
    rep = strat.simulate(te)
    # Filter must take a strict subset...
    assert 0 < rep.stats["n_trades"] < len(te)
    # ...and the taken trades win more often than take-all.
    assert rep.stats["win_rate"] > te["label"].mean()
    assert len(rep.equity_curve) == rep.stats["n_trades"]


def test_decide_respects_tau0_and_overlays():
    df = _make_events()
    strat = PaperStrategy(features=["good", "noise"], base_risk=0.01,
                          params={"max_depth": 3, "num_leaves": 7, "learning_rate": 0.05})
    strat.fit(df.iloc[:560])
    # A high-conviction row (large `good`) so p clears tau0 and the overlays,
    # not the threshold, decide the outcome.
    hi = df.iloc[600].copy()
    hi["good"] = 3.0
    hi["noise"] = 0.0
    assert strat.decide(hi).take  # clears tau0 on its own
    # symbol_busy always blocks regardless of p.
    d_busy = strat.decide(hi, symbol_busy=True)
    assert not d_busy.take and d_busy.reason == "symbol_busy"
    # risk cap blocks when already at the ceiling.
    d_cap = strat.decide(hi, open_risk=3 * strat.base_risk)
    assert not d_cap.take and d_cap.reason == "risk_cap"


def test_single_position_overlay_skips_overlap():
    # Two same-symbol events whose spans overlap: the second must be skipped.
    t0 = datetime(2021, 1, 1)
    df_tr = _make_events()
    strat = PaperStrategy(features=["good", "noise"], base_risk=0.01,
                          params={"max_depth": 3, "num_leaves": 7, "learning_rate": 0.05}).fit(df_tr.iloc[:560])
    overlap = pd.DataFrame({
        "t0": [t0, t0 + timedelta(hours=1)],
        "t1": [t0 + timedelta(hours=48), t0 + timedelta(hours=49)],
        "symbol": "EURUSD", "rule_id": "donchian20", "side": 1,
        "label": [1, 1], "r_realized": [1.5, 1.5],
        "good": [3.0, 3.0], "noise": [0.0, 0.0],  # high p -> both would be taken
    })
    rep = strat.simulate(overlap)
    assert rep.stats["n_trades"] == 1  # second overlapped -> skipped
