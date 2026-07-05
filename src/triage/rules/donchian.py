"""Donchian-20 breakout rule with D1 EMA-50 slope filter (§3, Phase 1).

ARCHITECTURE.md §3 commits this as the Phase 1 primary rule:

    Donchian-20 H4 breakout, direction-aligned with D1 EMA-50 slope,
    London/NY sessions only.

This module implements the *signal logic* shared by the two execution
paths (vectorized backfill + incremental live) in
:mod:`triage.rules.donchian_backfill` and
:mod:`triage.rules.donchian_live`. The bar-replay parity test
(ARCHITECTURE.md §3: "non-optional") asserts the two paths produce
byte-identical signal sets; it lives in ``tests/unit/test_donchian_parity.py``.

Hard look-ahead controls baked in here (§3/§4/§7 leakage checklist):

* **Entry is next_bar_open.** A signal at H4 bar close ``t0`` enters at
  the *next* H4 bar's open. There is no path by which the forming bar
  can enter barrier resolution. This is "kills same-bar look-ahead by
  construction" (§3).
* **Donchian / EMA use closed bars only.** The channel at ``t0`` is
  computed from bars closing at or before ``t0`` *excluding* the bar
  that just closed at ``t0`` (a breakout is defined against the prior
  20 bars, not including the bar doing the breaking). The D1 EMA-50
  uses D1 bars closed at or before the D1 bar containing ``t0``.
* **Session mask is wall-clock.** London/NY sessions are derived from
  the bar's close_time (UTC), never from anything not yet known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd

from ..data import Side

__all__ = [
    "CHANNEL_BARS",
    "D1_EMA_PERIOD",
    "LONDON_START_UTC",
    "LONDON_END_UTC",
    "NY_START_UTC",
    "NY_END_UTC",
    "DonchianConfig",
    "RuleVersion",
    "compute_donchian_channel",
    "compute_d1_ema_slope",
    "session_mask",
    "signal_at_bar",
    "RULE_VERSION",
]


# ---------------------------------------------------------------------------
# Frozen rule parameters — §3 Phase 1 commit.
#
# These are the rule's *shape*, not tunable HPO knobs. Per §0 ("every
# tuned knob is a trial") and §3 ("keep the trend filter as the only
# gate"), the model's job is conditional-edge recovery, not rule tuning.
# Any change to these values is a new rule_version and a new trial set.
# TODO(Phase 2+): move into configs/rules/donchian_break_v1.yaml.
# ---------------------------------------------------------------------------

#: Donchian channel length in H4 bars (§3: ``channel_bars: 20``).
CHANNEL_BARS: int = 20

#: D1 EMA period for the trend filter (§3: ``ema_period: 50``).
D1_EMA_PERIOD: int = 50


# Session windows in UTC. Forex sessions (§3: London/NY/overlap).
# IUX Markets bar timestamps are UTC (verified at ingest), so these
# are wall-clock UTC windows.
LONDON_START_UTC: time = time(7, 0)    # 07:00 UTC = 08:00 London (non-DST)
LONDON_END_UTC: time = time(16, 0)     # 16:00 UTC = 17:00 London
NY_START_UTC: time = time(12, 0)       # 12:00 UTC = 08:00 NY (non-DST)
NY_END_UTC: time = time(21, 0)         # 21:00 UTC = 17:00 NY


@dataclass(frozen=True)
class DonchianConfig:
    """Frozen configuration for the Donchian-20 rule.

    Carries the §3 parameters. ``rule_version`` is a content tag — the
    real content-hashing of code+config (§3: "versioned by content hash
    of code + config") lands with the config loader; this string is the
    stable tag every event row stamps into ``rule_version``.
    """

    channel_bars: int = CHANNEL_BARS
    d1_ema_period: int = D1_EMA_PERIOD
    london_start_utc: time = LONDON_START_UTC
    london_end_utc: time = LONDON_END_UTC
    ny_start_utc: time = NY_START_UTC
    ny_end_utc: time = NY_END_UTC
    rule_version: str = "donchian20_v1"


#: Default config instance (canonical Phase 1 commit).
DEFAULT_CONFIG: DonchianConfig = DonchianConfig()

#: Stable rule-version string stamped on every event. Phase 1 = donchian20_v1.
RULE_VERSION: str = DEFAULT_CONFIG.rule_version


# Type alias for clarity in the public API.
RuleVersion = str


# ---------------------------------------------------------------------------
# Indicators — all computed from closed bars only.
# ---------------------------------------------------------------------------


def compute_donchian_channel(
    h4: pd.DataFrame,
    cfg: DonchianConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return the rolling Donchian channel of prior highs/lows.

    Adds ``donchian_high`` / ``donchian_low`` columns to a copy of ``h4``.
    The channel at bar ``i`` is the max high / min low over the
    ``cfg.channel_bars`` bars **strictly before** ``i`` — i.e. the
    breakout reference excludes the bar doing the breaking. This is the
    look-ahead-safe definition: a signal at bar ``i``'s close compares
    ``i``'s close to a window that ends at ``i-1``.

    Args:
        h4: H4 bars DataFrame, indexed by close_time, with ``high``/
            ``low``/``close`` columns (see :func:`triage.data.bars_to_frame`).
        cfg: Rule config.

    Returns:
        Copy of ``h4`` with ``donchian_high`` / ``donchian_low`` columns.
        The first ``cfg.channel_bars`` rows have ``NaN`` channels
        (insufficient history) and will never signal.
    """
    n = cfg.channel_bars
    out = h4.copy()
    # shift(1) excludes the current bar -> strictly-prior window. Then a
    # rolling max/min over n bars. min_periods=n so no partial-window
    # signals leak out (a partial channel is not a real channel).
    out["donchian_high"] = out["high"].shift(1).rolling(n, min_periods=n).max()
    out["donchian_low"] = out["low"].shift(1).rolling(n, min_periods=n).min()
    return out


def compute_d1_ema_slope(
    d1: pd.DataFrame,
    cfg: DonchianConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return the D1 EMA and its slope, time-aligned to D1 bar close.

    Adds ``ema50`` and ``ema50_slope`` (signed difference vs prior bar's
    EMA). The EMA at D1 bar ``j`` is computed from D1 closes through
    ``j`` inclusive — that is safe because the *signal* is on an H4 bar
    whose ``t0`` falls *inside* D1 bar ``j``'s session, and we map the
    signal to D1 bar ``j-1``'s EMA (the last *closed* D1 bar at H4
    ``t0``). The mapping happens in :func:`signal_at_bar`; this function
    just produces the raw EMA series.

    Args:
        d1: Daily bars DataFrame, indexed by close_time, with ``close``.
        cfg: Rule config.

    Returns:
        Copy of ``d1`` with ``ema50`` and ``ema50_slope`` columns.
    """
    out = d1.copy()
    # Standard EMA: span=period, adjust=False (recursive — matches the
    # conventional definition and is stable for long series).
    out["ema50"] = out["close"].ewm(span=cfg.d1_ema_period, adjust=False).mean()
    out["ema50_slope"] = out["ema50"].diff()
    return out


# ---------------------------------------------------------------------------
# Session mask.
# ---------------------------------------------------------------------------


def session_mask(
    close_times: pd.DatetimeIndex,
    cfg: DonchianConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Boolean mask: True where bar close_time is in London ∪ NY sessions.

    Sessions are evaluated on the UTC wall-clock time of the bar's
    ``close_time``. §3 lists ``[london, ny, overlap]`` — the union is
    what gates the signal (overlap is automatically included).
    """
    t = pd.Series(close_times).dt.tz_convert("UTC").dt.time if pd.Series(close_times).dt.tz is not None else pd.Series(close_times).dt.time
    in_london = (t >= cfg.london_start_utc) & (t <= cfg.london_end_utc)
    in_ny = (t >= cfg.ny_start_utc) & (t <= cfg.ny_end_utc)
    return (in_london | in_ny).to_numpy()


# ---------------------------------------------------------------------------
# Signal logic — one bar at a time. Shared by both execution paths.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DonchianSignal:
    """A Donchian-20 breakout signal (the rule's output for one bar).

    ``t0`` is the signal H4 bar close. ``side`` is +1 (long, close
    broke *above* the prior-20 high with D1 EMA rising) or -1 (short,
    close broke *below* the prior-20 low with D1 EMA falling). Entry is
    always ``next_bar_open`` (§3); the entry price is filled by the
    labeler/event layer, not here.
    """

    t0: datetime
    side: Side
    entry_type: str = "next_bar_open"
    rule_version: str = RULE_VERSION


def signal_at_bar(
    *,
    t0: datetime,
    h4_close: float,
    donchian_high: float,
    donchian_low: float,
    d1_ema_slope: float,
    in_session: bool,
    cfg: DonchianConfig = DEFAULT_CONFIG,
) -> Optional[DonchianSignal]:
    """Evaluate the Donchian rule at a single H4 bar close.

    Pure function — the same inputs always yield the same output. Both
    the vectorized and incremental paths reduce to calling this per bar,
    which is what makes the parity test possible (§3).

    Long: ``h4_close > donchian_high`` AND ``d1_ema_slope > 0`` AND
    ``in_session``.
    Short: ``h4_close < donchian_low`` AND ``d1_ema_slope < 0`` AND
    ``in_session``.

    Returns ``None`` when no signal fires. Returns at most one signal
    per bar (a bar cannot break both sides simultaneously by
    construction: donchian_high > donchian_low for n>=2).

    NaN channel/ema values (warmup period) never signal: comparisons
    against NaN are False.
    """
    if not in_session:
        return None
    if donchian_high != donchian_high or donchian_low != donchian_low:
        # NaN guard (warmup). Cannot use `is None` — these are floats.
        return None
    if d1_ema_slope != d1_ema_slope:
        return None

    long_break = h4_close > donchian_high
    short_break = h4_close < donchian_low
    if long_break and d1_ema_slope > 0:
        return DonchianSignal(t0=t0, side=1)
    if short_break and d1_ema_slope < 0:
        return DonchianSignal(t0=t0, side=-1)
    return None
