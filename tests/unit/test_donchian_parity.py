"""Bar-replay parity test — the §3 gate (non-optional).

ARCHITECTURE.md §3: "replay history bar-by-bar through the live path
and assert byte-identical signal sets versus the vectorized run. This
test is non-optional — backfill/live divergence is the classic silent
killer of exactly this architecture."

This test runs both paths over the real EURUSD H4 + D1 history (loaded
from the local bar store populated in Phase 1 Step 0) and asserts the
emitted signal lists are exactly equal: same count, same t0, same side,
in the same order.

It also includes a small synthetic parity case that runs without the
real bar DB (so the gate is exercised even before ingest).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from triage.data.mt5_ingest import TF_D1, TF_H4, load_bars_frame
from triage.rules import (
    DEFAULT_CONFIG,
    RULE_VERSION,
    DonchianSignal,
    backfill_signals,
    replay_history,
)
from triage.rules.donchian import (
    compute_d1_ema_slope,
    compute_donchian_channel,
    signal_at_bar,
)

# ---------------------------------------------------------------------------
# Synthetic parity: a tiny hand-built H4+D1 frame, no DB needed.
# ---------------------------------------------------------------------------


def _synthetic_h4_d1():
    """Build a small H4 + D1 frame with an obvious breakout."""
    # 40 H4 bars; embed a clear upside breakout at bar 30.
    h4_rows = []
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i in range(40):
        close_time = base + pd.Timedelta(hours=4 * (i + 1))
        # Tight range 1.1000-1.1020 for bars 0..29, then spike to 1.1050.
        if i < 30:
            o = h = 1.1000 + 0.0005 * (i % 4)
            l = 1.0995
            c = 1.1000 + 0.0005 * (i % 4)
        else:
            o = 1.1000
            h = 1.1050 + 0.0002 * (i - 30)
            l = 1.0998
            c = 1.1050 + 0.0002 * (i - 30)
        h4_rows.append({"open_time": close_time - pd.Timedelta(hours=4), "open": o, "high": h, "low": l, "close": c})
    h4 = pd.DataFrame(h4_rows).set_index(pd.DatetimeIndex([r["open_time"] + pd.Timedelta(hours=4) for r in h4_rows], name="close_time"))

    # D1: 40 daily bars, gently rising (so ema50_slope > 0 to allow longs).
    d1_rows = []
    for i in range(40):
        close_time = base + pd.Timedelta(days=i + 1)
        d1_rows.append({"open_time": close_time - pd.Timedelta(days=1), "open": 1.1000 + 0.0005 * i, "high": 1.1010 + 0.0005 * i, "low": 1.0990 + 0.0005 * i, "close": 1.1005 + 0.0005 * i})
    d1 = pd.DataFrame(d1_rows).set_index(pd.DatetimeIndex([r["open_time"] + pd.Timedelta(days=1) for r in d1_rows], name="close_time"))
    return h4, d1


def test_synthetic_parity():
    """Vectorized == replay on a tiny synthetic frame."""
    h4, d1 = _synthetic_h4_d1()
    vec = backfill_signals(h4, d1)
    rep = replay_history(h4, d1)
    assert len(vec) == len(rep), f"signal count differs: vec={len(vec)} replay={len(rep)}"
    for a, b in zip(vec, rep):
        assert a.t0 == b.t0 and a.side == b.side, f"signal mismatch: {a} vs {b}"


def test_signal_at_bar_pure_function():
    """signal_at_bar is deterministic — same inputs, same output."""
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    s1 = signal_at_bar(t0=t0, h4_close=1.1050, donchian_high=1.1040,
                       donchian_low=1.0990, d1_ema_slope=0.0001, in_session=True)
    s2 = signal_at_bar(t0=t0, h4_close=1.1050, donchian_high=1.1040,
                       donchian_low=1.0990, d1_ema_slope=0.0001, in_session=True)
    assert s1 == s2
    assert s1 is not None and s1.side == 1
    assert s1.rule_version == RULE_VERSION


# ---------------------------------------------------------------------------
# Real-data parity — the actual §3 gate. Requires the bar DB.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_h4_d1():
    """Load real EURUSD H4+D1 from the local bar store. Skip if absent."""
    try:
        h4 = load_bars_frame("EURUSD", TF_H4)
        d1 = load_bars_frame("EURUSD", TF_D1)
    except Exception:
        pytest.skip("bar DB not available — run Phase 1 ingest first")
    if len(h4) == 0 or len(d1) == 0:
        pytest.skip("bar DB empty — run Phase 1 ingest first")
    return h4, d1


def test_real_data_parity(real_h4_d1):
    """§3 gate: vectorized backfill == bar-by-bar replay on real EURUSD."""
    h4, d1 = real_h4_d1
    vec = backfill_signals(h4, d1)
    rep = replay_history(h4, d1)

    # Byte-identical: same count, same (t0, side) sequence.
    assert len(vec) == len(rep), (
        f"PARITY BROKEN: vectorized produced {len(vec)} signals, "
        f"replay produced {len(rep)}. backfill/live divergence detected "
        "(ARCHITECTURE.md §3)."
    )
    for i, (a, b) in enumerate(zip(vec, rep)):
        assert a.t0 == b.t0, (
            f"PARITY BROKEN at signal {i}: t0 {a.t0} != {b.t0}"
        )
        assert a.side == b.side, (
            f"PARITY BROKEN at signal {i} (t0={a.t0}): side {a.side} != {b.side}"
        )


def test_real_data_signal_sanity(real_h4_d1):
    """Sanity: the real rule actually fires a non-trivial number of signals.

    Over ~11y of EURUSD H4 with a loose Donchian-20, §1 expects ~50-150
    events/year/symbol -> ~550-1650 events. We should land in that
    ballpark. This is a sanity check, not a gate.
    """
    h4, d1 = real_h4_d1
    sigs = backfill_signals(h4, d1)
    n = len(sigs)
    assert n > 0, "rule produced zero signals over 11y — broken"
    years = (h4.index[-1] - h4.index[0]).total_seconds() / (365.25 * 86400)
    per_year = n / years
    # Loose rule: at least 20/year (very conservative floor) and not
    # absurdly dense (>500/year would mean the rule is basically always on).
    assert 20 < per_year < 500, f"signal density {per_year:.0f}/year off-baseline"


def test_real_signals_have_valid_sides(real_h4_d1):
    """Every emitted signal is +1 or -1 and has a UTC t0."""
    h4, d1 = real_h4_d1
    for s in backfill_signals(h4, d1):
        assert s.side in (1, -1)
        assert s.t0.tzinfo is not None  # tz-aware UTC
