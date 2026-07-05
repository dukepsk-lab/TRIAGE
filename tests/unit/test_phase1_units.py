"""Unit tests for Phase 1 modules: uniqueness weights, sizing, gates.

These pin the §4/§7/§8 logic that the orchestration relies on.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from triage.gates import (
    MAJOR_COST_R,
    block_bootstrap_uplift,
    pbo_from_cpcv,
    strategy_returns_r,
)
from triage.sizing import (
    SizingLadder,
    breakeven_p,
    default_tau0,
    ev_r,
    select_tau_on_train,
)
from triage.uniqueness import assert_train_only, uniqueness_weights


# ---------------------------------------------------------------------------
# §4 uniqueness weights — per-fold guard + math.
# ---------------------------------------------------------------------------


def test_uniqueness_full_set_guard_triggers():
    """§4 leakage vector: computing over the full set must raise."""
    idx = pd.Index(["a", "b", "c"])
    with pytest.raises(ValueError, match="leakage vector"):
        assert_train_only(idx, idx)


def test_uniqueness_proper_subset_ok():
    """A genuine training subset passes the guard."""
    all_idx = pd.Index(["a", "b", "c", "d"])
    train_idx = pd.Index(["a", "b"])
    assert_train_only(train_idx, all_idx)  # no raise


def test_uniqueness_non_overlapping_events_weight_one():
    """Two events with disjoint spans each get weight ~1.0."""
    t0 = datetime(2026, 1, 5)
    ev = pd.DataFrame(
        {
            "t0": [t0, t0 + timedelta(hours=200)],
            "t1": [t0 + timedelta(hours=100), t0 + timedelta(hours=300)],
        },
        index=["e0", "e1"],
    )
    w = uniqueness_weights(ev)
    # Disjoint spans -> no concurrency -> each ~1.0.
    assert w["e0"] == pytest.approx(1.0, abs=0.05)
    assert w["e1"] == pytest.approx(1.0, abs=0.05)


def test_uniqueness_overlapping_events_below_one():
    """Two fully-overlapping events each get weight ~0.5."""
    t0 = datetime(2026, 1, 5)
    ev = pd.DataFrame(
        {
            "t0": [t0, t0],
            "t1": [t0 + timedelta(hours=100), t0 + timedelta(hours=100)],
        },
        index=["e0", "e1"],
    )
    w = uniqueness_weights(ev)
    # Perfectly concurrent -> each bar has c_t=2 -> mean(1/2) = 0.5.
    assert w["e0"] == pytest.approx(0.5, abs=0.05)
    assert w["e1"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# §8 sizing ladder.
# ---------------------------------------------------------------------------


def test_ev_formula_matches_section_8():
    """§8: EV(p) = 2.5p - 1 - c."""
    assert ev_r(0.5, c=0.04) == pytest.approx(2.5 * 0.5 - 1 - 0.04)
    assert breakeven_p(0.04) == pytest.approx((1 + 0.04) / 2.5)
    # tau0 = p* + 0.05.
    assert default_tau0(0.04) == pytest.approx(breakeven_p(0.04) + 0.05)


def test_ladder_zero_below_tau0():
    """No trade below tau0.

    tau0 is derived from cost via ``default_tau0(cost_r)`` — at the default
    cost 0.04 that is ~0.466, so both 0.40 and 0.46 sit below it.
    """
    lad = SizingLadder(tau1_margin=0.08)
    assert lad.size(0.40) == 0.0
    assert lad.size(0.46) == 0.0


def test_ladder_half_between_tau0_and_tau1():
    """Half base_risk in [tau0, tau1)."""
    lad = SizingLadder(tau1_margin=0.08, base_risk=0.01)
    # tau0 ~= 0.466, tau1 ~= 0.546 -> 0.50 lands in the half-size rung.
    assert lad.size(0.50) == pytest.approx(0.005)


def test_ladder_full_above_tau1():
    """Full base_risk at/above tau1."""
    lad = SizingLadder(tau1_margin=0.08, base_risk=0.01)
    # tau1 ~= 0.546 -> 0.55 and 0.70 are both at/above it.
    assert lad.size(0.55) == pytest.approx(0.01)
    assert lad.size(0.70) == pytest.approx(0.01)


def test_ladder_atr_overlay_halves_size():
    """§8 hard overlay: ATR percentile > 95 -> halve size."""
    lad = SizingLadder(tau1_margin=0.08, base_risk=0.01)
    assert lad.size(0.60, atr_percentile=0.96) == pytest.approx(0.005)
    # Below the 95 threshold -> no overlay.
    assert lad.size(0.60, atr_percentile=0.94) == pytest.approx(0.01)


def test_select_tau_on_train_picks_positive_ev_tau1():
    """tau1-margin selection should yield a positive-margin ladder over tau0."""
    rng = np.random.default_rng(0)
    # Synthesize probs correlated with a positive-outcome label so EV>0
    # above some threshold.
    probs = rng.uniform(0.3, 0.8, 500)
    labels = (probs + rng.normal(0, 0.15, 500) > 0.55).astype(int)
    lad = select_tau_on_train(probs, labels)
    # tau0 is formula-fixed from cost (break-even-ish); tau1 sits above it.
    assert default_tau0(0.04) > 0.40
    assert lad.tau1_margin > 0.0


# ---------------------------------------------------------------------------
# §7 gates math.
# ---------------------------------------------------------------------------


def test_strategy_returns_take_all_subtracts_cost():
    """Take-all mode: r_realized - cost on every event."""
    labels = np.array([1, 0, 1])
    r = np.array([1.5, -1.0, 1.5])
    ret = strategy_returns_r(labels, r, cost_r=0.04)
    assert ret == pytest.approx(np.array([1.5 - 0.04, -1.0 - 0.04, 1.5 - 0.04]))


def test_strategy_returns_filter_zeros_below_tau0():
    """Filter mode: trades below tau0 contribute 0."""
    labels = np.array([1, 0, 1])
    r = np.array([1.5, -1.0, 1.5])
    probs = np.array([0.6, 0.3, 0.7])  # middle trade below tau0
    ret = strategy_returns_r(labels, r, probabilities=probs, ladder_tau0=0.5, cost_r=0.04)
    assert ret[1] == 0.0
    assert ret[0] != 0.0
    assert ret[2] != 0.0


def test_block_bootstrap_uplift_detects_real_signal():
    """G1 bootstrap: a genuinely better filtered series -> low p-value."""
    rng = np.random.default_rng(0)
    primary = rng.normal(0.0, 0.01, 500)
    filtered = primary + rng.normal(0.005, 0.005, 500)  # shifted up
    ds, p = block_bootstrap_uplift(primary, filtered, n_boot=500, seed=0)
    assert ds > 0
    assert p < 0.10  # strongly significant uplift


def test_block_bootstrap_uplift_no_signal():
    """G1 bootstrap: identical series -> p-value near 0.5+."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 500)
    ds, p = block_bootstrap_uplift(x, x.copy(), n_boot=200, seed=0)
    assert ds == pytest.approx(0.0, abs=1e-6)
    assert p >= 0.4  # no signal -> rarely favor filtered


def test_pbo_zero_when_strategy_always_wins():
    """G3 PBO: strategy beating primary on every path -> PBO=0."""
    strat = [np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5])]
    prim = [np.array([0.5, 1.0, 1.5]), np.array([0.2, 0.7])]
    assert pbo_from_cpcv(strat, prim) == 0.0


def test_pbo_one_when_strategy_always_loses():
    """G3 PBO: strategy losing on every path -> PBO=1.0."""
    strat = [np.array([0.1, 0.2]), np.array([0.0, 0.1])]
    prim = [np.array([1.0, 2.0]), np.array([0.5, 1.0])]
    assert pbo_from_cpcv(strat, prim) == 1.0
