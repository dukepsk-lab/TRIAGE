"""Tier 4 MDA selection tests — §5 realized-trade-EV metric.

Pins the fix to ``mda_selection(metric="ev")``: the "ev" branch must score
features by realized trade EV (from ``r_realized``), NOT by mean predicted
probability. The historical bug returned mean p-hat, which is unrelated to
trade profitability.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from triage.features.tier4 import correlation_prune, mda_selection


def _make_events(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Synthetic event set: one feature drives the outcome, one is noise.

    ``good`` is correlated with a profitable outcome; ``noise`` is not.
    r_realized is +1.5 on wins, -1.0 on losses (the frozen-barrier geometry).
    """
    rng = np.random.default_rng(seed)
    good = rng.normal(size=n)
    noise = rng.normal(size=n)
    # Win probability rises with `good`; noise has no effect.
    p_win = 1.0 / (1.0 + np.exp(-1.5 * good))
    label = (rng.uniform(size=n) < p_win).astype(int)
    r = np.where(label == 1, 1.5, -1.0)
    t0 = [datetime(2020, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    X = pd.DataFrame({"good": good, "noise": noise})
    events = pd.DataFrame(
        {
            "t0": t0,
            "t1": [t + timedelta(hours=8) for t in t0],
            "symbol": "EURUSD",
            "r_realized": r,
            "label": label,
        }
    )
    return X, pd.Series(label), events


def test_ev_metric_uses_realized_returns_not_probability():
    """The informative feature must out-score noise under the EV metric."""
    X, y, events = _make_events()
    res = mda_selection(X, y, events, metric="ev", n_splits=5)
    by_feat = res.set_index("feature")["mda_mean"]
    # `good` genuinely helps select profitable trades; `noise` does not.
    assert by_feat["good"] > by_feat["noise"]
    # `good` should carry a positive captured-EV contribution.
    assert by_feat["good"] > 0.0


def test_ev_metric_requires_r_realized():
    """metric='ev' without r_realized must fail loudly, not silently degrade."""
    X, y, events = _make_events()
    events_no_r = events.drop(columns=["r_realized"])
    with pytest.raises(ValueError, match="r_realized"):
        mda_selection(X, y, events_no_r, metric="ev", n_splits=5)


def test_invalid_metric_rejected():
    X, y, events = _make_events()
    with pytest.raises(ValueError, match="metric"):
        mda_selection(X, y, events, metric="sharpe", n_splits=5)


def test_correlation_prune_drops_collinear():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    df = pd.DataFrame({"a": a, "b": a * 1.0 + rng.normal(scale=1e-6, size=400), "c": rng.normal(size=400)})
    kept = correlation_prune(df, threshold=0.85)
    assert "a" in kept and "b" not in kept and "c" in kept
