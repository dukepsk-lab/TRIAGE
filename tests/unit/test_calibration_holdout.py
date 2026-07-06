"""Purged holdout calibration path (§6) — smoke + sanity.

Verifies the ``calibration="holdout"`` branch of :func:`train_meta_model`:
it runs end-to-end, produces finite metrics, and — on a genuinely
separable synthetic set — yields a reliability slope that is not the
collapsed-to-zero signature the pooled Phase 2 run showed. Also pins the
reusable :func:`reliability_slope` helper.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from triage.models.meta import reliability_slope, train_meta_model


def _separable_dataset(n: int, seed: int = 0):
    """Feature whose signal drives the label, plus aligned t0/t1 spans."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    # P(win) rises with x -> a rankable, calibratable signal.
    p = 1.0 / (1.0 + np.exp(-1.2 * x))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    feats = pd.DataFrame({"f0": x, "f1": rng.normal(0, 1, n)})
    t0 = pd.to_datetime(
        [datetime(2020, 1, 1) + timedelta(hours=4 * i) for i in range(n)]
    )
    events = pd.DataFrame({"t0": t0, "t1": t0 + pd.Timedelta(hours=96)})
    return feats, pd.Series(y), events


def test_reliability_slope_perfect_calibration():
    """A perfectly calibrated predictor has slope ~1."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)
    slope = reliability_slope(y, p, n_bins=10)
    assert 0.8 <= slope <= 1.2


def test_reliability_slope_degenerate_returns_zero():
    """Constant predictions -> undefined slope -> 0.0 (gate-failing)."""
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.full(6, 0.4)
    assert reliability_slope(y, p) == 0.0


def test_holdout_calibration_runs_and_ranks():
    """Holdout path returns finite, sane metrics on a separable set."""
    n = 1200
    feats, y, events = _separable_dataset(n)
    n_tr = 900
    res = train_meta_model(
        feats.iloc[:n_tr],
        y.iloc[:n_tr],
        test_features=feats.iloc[n_tr:],
        test_labels=y.iloc[n_tr:],
        events=events.iloc[:n_tr],
        calibration="holdout",
        n_estimators=200,
    )
    # Booster fit on the earlier portion, calibrator on the purged tail.
    assert res.n_val > 0                       # a real calibration fold existed
    assert res.n_train < n_tr                  # fit set is a strict subset
    assert res.auc > 0.55                      # signal is recovered
    assert np.isfinite(res.brier)
    # Calibrated probabilities are not collapsed to a single constant.
    assert np.std(res.oof_pred) > 0.02


def test_holdout_falls_back_when_fold_degenerate():
    """A tiny input can't form a valid fold -> graceful fallback, no crash."""
    n = 160
    feats, y, events = _separable_dataset(n, seed=3)
    res = train_meta_model(
        feats.iloc[:120],
        y.iloc[:120],
        test_features=feats.iloc[120:],
        test_labels=y.iloc[120:],
        events=events.iloc[:120],
        calibration="holdout",
        n_estimators=100,
    )
    assert np.isfinite(res.auc) and np.isfinite(res.brier)


def test_invalid_calibration_mode_raises():
    feats, y, _ = _separable_dataset(300)
    try:
        train_meta_model(
            feats.iloc[:200], y.iloc[:200],
            test_features=feats.iloc[200:], test_labels=y.iloc[200:],
            calibration="bogus",
        )
    except ValueError as e:
        assert "calibration must be" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for bad calibration mode")
