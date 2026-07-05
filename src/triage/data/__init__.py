"""Bar data types and synthetic loaders.

Implements the bar representation referenced throughout
ARCHITECTURE.md §2 (data layer) and §4 (labeling).

Phase 0 ships **synthetic data only** (ARCHITECTURE.md §10, Phase 0).
Real market data / TimescaleDB is out of scope until Phase 1; see
``docker-compose.yml`` for the planned service.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import pandas as pd

__all__ = ["Bar", "Side", "Bars", "bars_to_frame", "frame_to_bars", "load_bars"]

# A trade side. +1 = long, -1 = short. Matches ARCHITECTURE.md §4
# (``side`` column on the ``events`` table) and the direction convention
# used in ``resolve()``.
Side = Literal[1, -1]


@dataclass(frozen=True)
class Bar:
    """A single OHLC bar with a closed timestamp.

    Per ARCHITECTURE.md §4, the entry convention is: signal evaluated at
    H4 bar close ``t0``; entry at next H4 bar open. Every feature / ATR
    value must be computed from bars whose ``close_time <= t0`` — the
    still-forming bar never enters any computation. This invariant is
    enforced by ``triage.features.no_lookahead`` and asserted in tests.

    ``close_time`` is the bar's *close* timestamp (the moment it becomes
    "formed"); ``open_time`` is retained for clarity and for parity with
    the planned TimescaleDB schema.
    """

    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float


# A time-ordered sequence of bars. The labeler and validators operate on
# this representation so that synthetic (Phase 0) and real (Phase 1+)
# data share one code path.
Bars = list[Bar]


def bars_to_frame(bars: Bars) -> pd.DataFrame:
    """Convert a bar list to a DataFrame indexed by ``close_time``.

    The DataFrame form is convenient for vectorized scans during M1
    barrier resolution (ARCHITECTURE.md §6: "vectorize the first-touch
    scan rather than looping bars in Python"). Phase 0 keeps the resolver
    explicit and Pythonic for legibility against §4's pseudocode; the
    vectorized fast path is a Phase 1 optimization.
    """
    if not bars:
        return pd.DataFrame(
            columns=["open_time", "open", "high", "low", "close"]
        ).set_index("close_time")
    df = pd.DataFrame(
        [
            {
                "open_time": b.open_time,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
            }
            for b in bars
        ],
        index=pd.DatetimeIndex([b.close_time for b in bars], name="close_time"),
    )
    return df


def frame_to_bars(df: pd.DataFrame) -> Bars:
    """Inverse of :func:`bars_to_frame`."""
    out: Bars = []
    for close_time, row in df.iterrows():
        out.append(
            Bar(
                open_time=row["open_time"],
                close_time=close_time.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
        )
    return out


def load_synthetic(name: str) -> Optional[Bars]:
    """Resolve a named synthetic bar path.

    Phase 0 only. Phase 1 will replace this with a TimescaleDB-backed
    loader (ARCHITECTURE.md §9). Returns ``None`` if the name is unknown
    so callers can fall back without raising on missing fixtures.
    """
    # Deliberately not wired to anything yet — generators live under
    # ``tests/synthetic`` so they ship next to the assertions that pin
    # their ground-truth outcomes (ARCHITECTURE.md §4 "Correctness
    # battery"). The production data loader is a Phase 1 concern.
    return None
