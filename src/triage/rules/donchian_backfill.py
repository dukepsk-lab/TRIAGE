"""Donchian-20 rule: vectorized backfill + incremental live paths.

ARCHITECTURE.md §3 mandates **two code paths, one truth**:

    Vectorized backfill (vectorbt/numpy) for history; incremental
    evaluation for live. These must pass a bar-replay parity test:
    replay history bar-by-bar through the live path and assert
    byte-identical signal sets versus the vectorized run. This test is
    non-optional — backfill/live divergence is the classic silent
    killer of exactly this architecture.

Both paths here reduce to the same per-bar evaluation
(:func:`triage.rules.donchian.signal_at_bar`). The vectorized path
precomputes the indicator columns and applies the gate row-wise; the
incremental path maintains a rolling window of H4 bars and the D1 EMA
state and evaluates one bar at a time as it would live. Because they
share the *exact same* signal function and the *exact same* D1->H4
time alignment, parity is structural — and the parity test
(``tests/unit/test_donchian_parity.py``) asserts it on the real
EURUSD history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from ..data import Bars, bars_to_frame
from .donchian import (
    DEFAULT_CONFIG,
    DonchianConfig,
    DonchianSignal,
    RULE_VERSION,
    compute_d1_ema_slope,
    compute_donchian_channel,
    signal_at_bar,
)

__all__ = [
    "backfill_signals",
    "IncrementalDonchian",
    "d1_slope_at_h4_close",
    "align_d1_slope_to_h4",
]


# ---------------------------------------------------------------------------
# D1 -> H4 alignment (shared by both paths).
# ---------------------------------------------------------------------------


def align_d1_slope_to_h4(h4: pd.DataFrame, d1: pd.DataFrame,
                         cfg: DonchianConfig = DEFAULT_CONFIG) -> pd.Series:
    """Map each H4 bar close to the slope of the last *closed* D1 EMA.

    An H4 bar closing at ``t0`` falls inside some D1 bar ``j``. The only
    D1 bar whose EMA is fully formed at ``t0`` is ``j-1`` (D1 bar ``j``
    is still forming until its own close). So we use the EMA slope of
    D1 bar ``j-1``, i.e. the most recent D1 close strictly before the
    H4 bar's *session day*.

    Concretely: for H4 close_time ``t0``, find the latest D1 close_time
    that is strictly before the *date* of ``t0`` (so same-day D1 is
    excluded — it hasn't closed yet when intraday H4 bars fire).

    This is the load-bearing no-look-ahead choice: it would be a leak to
    use same-day D1 EMA for an H4 signal firing earlier that day.

    Returns a Series indexed like ``h4`` with the aligned slope values
    (NaN where no prior D1 bar exists yet).
    """
    d1_with_ema = compute_d1_ema_slope(d1, cfg)
    # Index D1 by its close date (the date the D1 bar *became known*).
    # For H4 t0, the usable D1 bar is the last one whose close_time < t0.
    d1_close = d1_with_ema["ema50_slope"]
    d1_close.index = pd.to_datetime(d1_close.index)
    # asof: for each h4 close_time, the last d1 value with index <= that
    # time. But we need strictly-before on the *day*, so we asof on a
    # value shifted to the d1 bar's *close* instant, which for a D1 bar
    # ending at day D is the morning of D+1 — i.e. already strictly after
    # any intraday H4 bar on day D. Using asof(t0) with close_time gives
    # the last D1 bar that closed at or before t0, which for an intraday
    # H4 bar on day D is D-1's bar. Exactly what we want.
    h4_idx = pd.to_datetime(h4.index)
    aligned = d1_close.reindex(d1_close.index).sort_index()
    return pd.Series(
        aligned.reindex(aligned.index.union(h4_idx)).ffill().reindex(h4_idx).to_numpy(),
        index=h4.index,
        name="d1_ema50_slope_aligned",
    )


def d1_slope_at_h4_close(t0: datetime, d1_with_ema: pd.DataFrame) -> float:
    """Slope of the last closed D1 bar strictly before ``t0`` (incremental).

    ``d1_with_ema`` must already carry ``ema50_slope`` (from
    :func:`compute_d1_ema_slope`). Returns NaN if no D1 bar is available
    yet (warmup). Used by the incremental path; the vectorized path uses
    :func:`align_d1_slope_to_h4` which is the bulk equivalent.
    """
    sub = d1_with_ema[d1_with_ema.index < pd.Timestamp(t0)]
    if len(sub) == 0:
        return float("nan")
    return float(sub["ema50_slope"].iloc[-1])


# ---------------------------------------------------------------------------
# Vectorized backfill path.
# ---------------------------------------------------------------------------


def backfill_signals(
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    cfg: DonchianConfig = DEFAULT_CONFIG,
) -> list[DonchianSignal]:
    """Vectorized backfill: produce every signal over the full H4 history.

    Computes the Donchian channel + D1 EMA slope in bulk, evaluates the
    gate row-wise via the shared :func:`signal_at_bar`, and returns the
    ordered list of signals. This is the "history" path per §3.
    """
    h4_chan = compute_donchian_channel(h4, cfg)
    slope = align_d1_slope_to_h4(h4, d1, cfg)
    h4_chan["d1_ema50_slope_aligned"] = slope

    # Session mask over close_time.
    from .donchian import session_mask
    in_session = session_mask(h4.index, cfg)

    signals: list[DonchianSignal] = []
    for i, (close_time, row) in enumerate(h4_chan.iterrows()):
        sig = signal_at_bar(
            t0=pd.Timestamp(close_time).to_pydatetime(),
            h4_close=float(row["close"]),
            donchian_high=(
                float(row["donchian_high"]) if pd.notna(row["donchian_high"]) else float("nan")
            ),
            donchian_low=(
                float(row["donchian_low"]) if pd.notna(row["donchian_low"]) else float("nan")
            ),
            d1_ema_slope=(
                float(row["d1_ema50_slope_aligned"])
                if pd.notna(row["d1_ema50_slope_aligned"])
                else float("nan")
            ),
            in_session=bool(in_session[i]),
            cfg=cfg,
        )
        if sig is not None:
            signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# Incremental live path.
# ---------------------------------------------------------------------------


@dataclass
class IncrementalDonchian:
    """Bar-by-bar incremental evaluator, as a live loop would call it.

    Maintains the rolling H4 window and the D1 EMA series, and on each
    new closed H4 bar evaluates :func:`signal_at_bar` with the *exact*
    same arguments the vectorized path computed for that bar. The
    parity test (§3) replays the H4 history through ``on_bar`` and
    compares to :func:`backfill_signals`'s output.

    The D1 EMA is provided up-front (closed D1 history) and extended
    incrementally only when a new D1 bar closes — matching how live
    works, where D1 bars close once per day and H4 bars six times daily.
    """

    cfg: DonchianConfig = DEFAULT_CONFIG
    _h4_window: deque = None  # type: ignore[assignment]
    _d1: pd.DataFrame = None  # type: ignore[assignment]
    _signals: list[DonchianSignal] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._h4_window is None:
            self._h4_window = deque(maxlen=self.cfg.channel_bars + 1)
        if self._signals is None:
            self._signals = []

    def attach_d1(self, d1: pd.DataFrame) -> None:
        """Provide the closed D1 history (with EMA computed on attach).

        In live use this is called once at startup with whatever D1
        history is available; :meth:`on_d1_bar` extends it as new D1
        bars close.
        """
        self._d1 = compute_d1_ema_slope(d1, self.cfg)

    def on_d1_bar(self, d1_bar: pd.Series) -> None:
        """Append a newly-closed D1 bar and recompute the EMA tail.

        Recomputation is over the full series to keep EMA numerically
        identical to the vectorized path (EMA is recursive; appending
        and using the closed-form ewm keeps both paths bit-stable).
        """
        if self._d1 is None:
            self._d1 = pd.DataFrame([d1_bar])
        else:
            self._d1 = pd.concat([self._d1, pd.DataFrame([d1_bar])])
        self._d1 = compute_d1_ema_slope(self._d1, self.cfg)

    def on_bar(self, h4_close_time: datetime, o: float, h: float, l: float, c: float) -> Optional[DonchianSignal]:
        """Process one closed H4 bar; emit a signal if the rule fires.

        Returns the signal (also appended internally) or ``None``.
        """
        self._h4_window.append((h4_close_time, h, l, c))
        n = self.cfg.channel_bars
        if len(self._h4_window) <= n:
            # Warmup: not enough strictly-prior bars for a channel yet.
            # (window holds current bar + n prior bars.)
            return None

        # Channel over the *strictly prior* n bars (exclude the just-added bar).
        prior = list(self._h4_window)[:-1][-n:]
        donchian_high = max(b[1] for b in prior)
        donchian_low = min(b[2] for b in prior)

        # D1 slope: last closed D1 bar strictly before this H4 close.
        d1_slope = d1_slope_at_h4_close(h4_close_time, self._d1) if self._d1 is not None else float("nan")

        from .donchian import session_mask
        in_session = bool(session_mask(pd.DatetimeIndex([h4_close_time]), self.cfg)[0])

        sig = signal_at_bar(
            t0=h4_close_time,
            h4_close=c,
            donchian_high=donchian_high,
            donchian_low=donchian_low,
            d1_ema_slope=d1_slope,
            in_session=in_session,
            cfg=self.cfg,
        )
        if sig is not None:
            self._signals.append(sig)
        return sig

    @property
    def signals(self) -> list[DonchianSignal]:
        return list(self._signals)


def replay_history(
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    cfg: DonchianConfig = DEFAULT_CONFIG,
) -> list[DonchianSignal]:
    """Replay H4 history bar-by-bar through the incremental path.

    Convenience for the parity test: feeds every H4 bar in order to a
    fresh :class:`IncrementalDonchian` and returns the emitted signals.
    Must equal :func:`backfill_signals` exactly (§3 gate).
    """
    inc = IncrementalDonchian(cfg=cfg)
    inc.attach_d1(d1)
    for close_time, row in h4.iterrows():
        ct = pd.Timestamp(close_time).to_pydatetime()
        inc.on_bar(ct, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
    return inc.signals
