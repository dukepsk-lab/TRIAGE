"""Execution adapter tests — pure lot sizing (no MT5 needed)."""
from __future__ import annotations

from triage.execution.broker import compute_lot


def _econ(**kw):
    base = dict(tick_value=1.0, tick_size=0.0001, vol_min=0.01, vol_max=100.0, vol_step=0.01)
    base.update(kw)
    return base


def test_risk_budget_maps_to_lots():
    # SL 20 pips away (0.0020). tick_value $1 per 0.0001 per 1.0 lot ->
    # loss/lot = 0.0020/0.0001 * 1 = $20. Risk $200 -> 10 lots.
    lot = compute_lot(entry_px=1.1000, sl_px=1.0980, risk_amount=200.0, **_econ())
    assert abs(lot - 10.0) < 1e-9


def test_floors_to_step_and_never_over_risks():
    # Risk that implies 3.47 lots must floor to 3.47->3.47? step 0.01 -> 3.47.
    lot = compute_lot(entry_px=1.1000, sl_px=1.0980, risk_amount=69.4, **_econ())
    assert lot <= 69.4 / 20.0 + 1e-9  # never rounds UP past the budget
    assert round(lot / 0.01) * 0.01 == lot


def test_below_min_lot_returns_zero_not_over_risk():
    # Budget too small to afford even vol_min at this SL -> 0 (skip), not min.
    lot = compute_lot(entry_px=1.1000, sl_px=1.0980, risk_amount=0.10, **_econ())
    assert lot == 0.0


def test_degenerate_inputs_return_zero():
    assert compute_lot(entry_px=1.1, sl_px=1.1, risk_amount=100, **_econ()) == 0.0
    assert compute_lot(entry_px=1.1, sl_px=1.09, risk_amount=100, **_econ(tick_size=0.0)) == 0.0


def test_clamps_to_vol_max():
    lot = compute_lot(entry_px=1.1000, sl_px=1.0999, risk_amount=1e9, **_econ(vol_max=50.0))
    assert lot == 50.0
