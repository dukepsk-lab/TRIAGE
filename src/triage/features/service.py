"""Feature service — Tier 1+2+3 point-in-time snapshots (§5).

ARCHITECTURE.md §5: "Features are computed once, at signal time, from
closed bars only, and persisted into ``events.features`` with a
``feature_ver`` hash. They are never recomputed later from a full-history
frame."

Phase 1 ships Tier 1+2+3 (~14-16 columns). The architecture doc doesn't
enumerate exact columns (it says "as specified in the brief"), so this
module defines a defensible set covering the categories §5/§1 call out:
volatility regime, momentum/RSI, level proximity to the Donchian
channel, HTF (D1) alignment, and ATR-based risk geometry. Every
snapshot is run through :func:`triage.features.no_lookahead` before
return — the timestamp audit is wired into the service itself, not just
left as a standalone test (Phase 1 Step 1 requirement).

No-look-ahead invariant (load-bearing):

* Every indicator uses H4/D1 bars whose ``close_time <= t0``.
* The still-forming bar never enters any computation.
* Each snapshot carries ``max_source_close_time`` (the max close_time
  of any bar that fed any component) so the audit can assert it is
  ``<= t0``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .audit import no_lookahead
from ..data import Bars

__all__ = [
    "FEATURE_VERSION",
    "FEATURE_COLUMNS",
    "compute_snapshot",
    "compute_features_frame",
    "feature_hash",
]


# ---------------------------------------------------------------------------
# Feature inventory — Tier 1 / 2 / 3 (§5).
#
# Tier 1 — raw risk/price geometry (always available, no estimation):
#   atr_pct            ATR / price (volatility-normalized risk)
#   atr_per_high52     ATR percentile vs trailing 52-week H4 ATR (vol regime)
#   range_pct          last bar (high-low)/close
#   level_don_high     close vs Donchian-20 high (how far above channel)
#   level_don_low      close vs Donchian-20 low
#
# Tier 2 — momentum / mean-reversion structure:
#   rsi_14             14-bar RSI on H4 closes
#   rsi_change_3       RSI diff over 3 bars
#   ema_slope_h4       H4 EMA-20 slope (trend alignment, intraday)
#   roc_6              6-bar rate of change
#   bb_width           Bollinger (20,2) bandwidth (squeeze/expansion)
#   bb_pos             position within Bollinger bands [0,1]
#
# Tier 3 — HTF (D1) alignment + free context (§5: "Tier 3 is free"):
#   d1_ema_slope       D1 EMA-50 slope (the rule's own trend filter)
#   d1_atr_pct         D1 ATR/price
#   d1_roc_5           D1 5-bar ROC
#   hour_utc           hour-of-day (seasonality; categorical in Phase 2)
#
# Total: 16 columns. Tier 4 (cross-sectional) enters at Phase 3 only.
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    # Tier 1
    "atr_pct", "atr_per_high52", "range_pct", "level_don_high", "level_don_low",
    # Tier 2
    "rsi_14", "rsi_change_3", "ema_slope_h4", "roc_6", "bb_width", "bb_pos",
    # Tier 3
    "d1_ema_slope", "d1_atr_pct", "d1_roc_5", "hour_utc",
]

#: Stable feature-version tag. The real content hash (§5: "feature_ver
#: hash of code + config") is :func:`feature_hash` over the column list
#: + this tag. Computed once at import; every snapshot stamps it.
FEATURE_VERSION: str = "phase1_tier123_v1"


def feature_hash() -> str:
    """Content hash of the feature definition (§5 ``feature_ver``)."""
    blob = json.dumps({"v": FEATURE_VERSION, "cols": FEATURE_COLUMNS}).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Indicator helpers (all rolling, all on closed bars <= t0).
# ---------------------------------------------------------------------------


def _atr(closes: pd.Series, highs: pd.Series, lows: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's), per-bar, computed from closed bars.

    Standard Wilder's: TR = max(H-L, |H-prevC|, |L-prevC|), then Wilder
    smoothing (recursive, equivalent to ewm(alpha=1/period, adjust=False)).
    """
    prev_close = closes.shift(1)
    tr = pd.concat(
        [(highs - lows), (highs - prev_close).abs(), (lows - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder's)."""
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger(closes: pd.Series, period: int = 20, k: float = 2.0):
    """Return (width, pos) for Bollinger bands. pos in [0,1] within band."""
    mid = closes.rolling(period, min_periods=period).mean()
    sd = closes.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid
    pos = (closes - lower) / (upper - lower).replace(0.0, np.nan)
    return width, pos


# ---------------------------------------------------------------------------
# Snapshot computation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureContext:
    """Bars available up to (and including) ``t0`` for one event.

    All series indexed by close_time, containing only bars with
    ``close_time <= t0``. The caller (event-build pipeline) is
    responsible for the slicing; this object just holds the result so
    the audit can verify the invariant.
    """

    t0: datetime
    h4: pd.DataFrame       # bars with close_time <= t0
    d1: pd.DataFrame       # bars with close_time <= t0 (D1 perspective)


def compute_snapshot(ctx: FeatureContext) -> dict:
    """Compute the Tier 1+2+3 feature snapshot for one event at ``ctx.t0``.

    Returns a flat dict of feature name -> value, plus the audit fields
    ``max_source_close_time`` (the latest bar close that fed any
    component) and ``feature_ver``. Every indicator is computed only
    from rows in ``ctx.h4`` / ``ctx.d1``, which the caller sliced to
    ``close_time <= t0`` — so the snapshot is look-ahead-safe by
    construction, and :func:`no_lookahead` is invoked here as the
    enforcement check.
    """
    h4 = ctx.h4
    d1 = ctx.d1
    t0 = ctx.t0

    if len(h4) < 20 or len(d1) < 5:
        raise ValueError(
            f"insufficient history at t0={t0}: need >=20 H4 and >=5 D1 bars, "
            f"got {len(h4)}/{len(d1)}"
        )

    closes_h4 = h4["close"]
    highs_h4 = h4["high"]
    lows_h4 = h4["low"]

    # --- Tier 1 ---
    atr14 = _atr(closes_h4, highs_h4, lows_h4, 14)
    atr0 = float(atr14.iloc[-1])
    px = float(closes_h4.iloc[-1])
    atr_pct = atr0 / px if px else float("nan")

    # ATR percentile vs trailing ~52 weeks of H4 (~6/day*260d = 1560 bars).
    window52w = min(len(atr14) - 1, 1560)
    atr_per_high52 = (
        float((atr14.iloc[-1] >= atr14.iloc[-window52w:-1]).sum()) / window52w
        if window52w > 0
        else float("nan")
    )

    range_pct = (float(highs_h4.iloc[-1]) - float(lows_h4.iloc[-1])) / px

    # Donchian-20 channel (strictly prior 20 bars — exclude the bar at t0).
    if len(highs_h4) > 20:
        don_high = float(highs_h4.iloc[-21:-1].max())
        don_low = float(lows_h4.iloc[-21:-1].min())
    else:
        don_high = don_low = float("nan")
    level_don_high = (px - don_high) / atr0 if atr0 else float("nan")
    level_don_low = (px - don_low) / atr0 if atr0 else float("nan")

    # --- Tier 2 ---
    rsi = _rsi(closes_h4, 14)
    rsi_14 = float(rsi.iloc[-1])
    rsi_change_3 = float(rsi.iloc[-1] - rsi.iloc[-4]) if len(rsi) >= 4 else float("nan")

    ema20 = closes_h4.ewm(span=20, adjust=False).mean()
    ema_slope_h4 = float(ema20.iloc[-1] - ema20.iloc[-2]) if len(ema20) >= 2 else float("nan")

    roc_6 = (
        float(closes_h4.iloc[-1] / closes_h4.iloc[-7] - 1.0)
        if len(closes_h4) >= 7
        else float("nan")
    )

    bb_width, bb_pos = _bollinger(closes_h4, 20, 2.0)
    bb_width_v = float(bb_width.iloc[-1]) if len(bb_width) else float("nan")
    bb_pos_v = float(bb_pos.iloc[-1]) if len(bb_pos) else float("nan")

    # --- Tier 3 ---
    closes_d1 = d1["close"]
    highs_d1 = d1["high"]
    lows_d1 = d1["low"]
    ema50_d1 = closes_d1.ewm(span=50, adjust=False).mean()
    d1_ema_slope = float(ema50_d1.iloc[-1] - ema50_d1.iloc[-2]) if len(ema50_d1) >= 2 else float("nan")
    atr_d1 = _atr(closes_d1, highs_d1, lows_d1, 14)
    d1_atr_pct = float(atr_d1.iloc[-1] / closes_d1.iloc[-1]) if len(closes_d1) and closes_d1.iloc[-1] else float("nan")
    d1_roc_5 = (
        float(closes_d1.iloc[-1] / closes_d1.iloc[-6] - 1.0)
        if len(closes_d1) >= 6
        else float("nan")
    )

    # Hour-of-day in UTC (sessions are UTC in this build). tz-aware t0
    # converted to UTC; naive t0 assumed already UTC.
    if t0.tzinfo is not None:
        hour_utc = float(t0.astimezone(timezone.utc).hour)
    else:
        hour_utc = float(t0.hour)

    # --- Audit field: max source close_time feeding any component ---
    max_h4_close = pd.Timestamp(h4.index[-1]).to_pydatetime()
    max_d1_close = pd.Timestamp(d1.index[-1]).to_pydatetime()
    max_source = max(max_h4_close, max_d1_close)

    snap = {
        "atr_pct": atr_pct,
        "atr_per_high52": atr_per_high52,
        "range_pct": range_pct,
        "level_don_high": level_don_high,
        "level_don_low": level_don_low,
        "rsi_14": rsi_14,
        "rsi_change_3": rsi_change_3,
        "ema_slope_h4": ema_slope_h4,
        "roc_6": roc_6,
        "bb_width": bb_width_v,
        "bb_pos": bb_pos_v,
        "d1_ema_slope": d1_ema_slope,
        "d1_atr_pct": d1_atr_pct,
        "d1_roc_5": d1_roc_5,
        "hour_utc": hour_utc,
        "max_source_close_time": max_source,
        "feature_ver": feature_hash(),
    }
    # Enforcement: assert no look-ahead (raises LookaheadError if violated).
    no_lookahead(snap, t0)
    return snap


def compute_features_frame(snapshots: list[dict]) -> pd.DataFrame:
    """Stack a list of snapshots into a feature matrix (FEATURE_COLUMNS only)."""
    if not snapshots:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.DataFrame(snapshots)[FEATURE_COLUMNS]
