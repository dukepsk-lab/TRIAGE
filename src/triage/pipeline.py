"""Event generation pipeline: rule → features → labeling → events store.

Ties together the Phase 1 components produced so far:

    Donchian rule signals (§3)
        → Event with frozen barriers (§4)
            → point-in-time feature snapshot (§5, no_lookahead-checked)
                → M1 first-touch labeling (§4)
                    → persisted to the events store (§4 schema)

This is the backfill path. The events store is a local SQLite table
mirroring the §4 ``events`` schema (TimescaleDB is a later-phase swap;
the schema is identical so migration is a writer change).

Phase 0's labeler is reused unchanged — ``label_event`` resolves the
event against the M1 bars in ``(entry_time, t_vert]`` and returns the
labeled form. The pipeline's only addition is *slicing* the M1 stream
correctly per event and stamping the feature snapshot.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .data import Bars
from .data.mt5_ingest import DEFAULT_BAR_DB, TF_M1, load_bars
from .features import FEATURE_COLUMNS, FeatureContext, compute_snapshot, feature_hash
from .labeling import Event, LabeledEvent, VERTICAL_BARS, label_event, make_event
from .rules import DonchianSignal, RULE_VERSION
from .rules.pullback import PullbackSignal

from typing import Union
SignalType = Union[DonchianSignal, PullbackSignal]

__all__ = [
    "DEFAULT_EVENTS_DB",
    "build_events_for_signals",
    "persist_events",
    "load_events",
    "EventsStore",
]


DEFAULT_EVENTS_DB: Path = Path(".triage") / "events.sqlite"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol         TEXT NOT NULL,
    rule_id        TEXT NOT NULL,
    rule_version   TEXT NOT NULL,
    t0             TEXT NOT NULL,
    side           INTEGER NOT NULL,
    entry_px       REAL NOT NULL,
    atr0           REAL NOT NULL,
    tp_px          REAL NOT NULL,
    sl_px          REAL NOT NULL,
    t_vert         TEXT NOT NULL,
    t1             TEXT,
    outcome        TEXT,
    label          INTEGER,
    r_realized     REAL,
    features       TEXT NOT NULL,
    feature_ver    TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_events_t0   ON events(t0);
CREATE INDEX IF NOT EXISTS idx_events_rule ON events(rule_id, rule_version);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Per-event M1 slice + barrier resolution.
# ---------------------------------------------------------------------------


def _atr_at(h4_pre: pd.DataFrame, period: int = 14) -> float:
    """ATR(period) at the close of the last bar in ``h4_pre`` (Wilder's)."""
    import numpy as np
    closes = h4_pre["close"]; highs = h4_pre["high"]; lows = h4_pre["low"]
    prev_close = closes.shift(1)
    tr = pd.concat(
        [(highs - lows), (highs - prev_close).abs(), (lows - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return float(atr.iloc[-1])


def label_one_event(
    signal: SignalType,
    *,
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    m1_all: pd.DataFrame,
    symbol: str = "EURUSD",
    rule_id: str = "donchian20",
) -> Optional[LabeledEvent]:
    """Build + label a single event from a rule signal.

    Returns the labeled event (with features snapshot + outcome) or
    ``None`` if the event cannot be constructed (e.g. no next H4 bar
    for an entry price, or insufficient M1 to resolve).

    The entry price is the *next* H4 bar's open after ``signal.t0``
    (§3/§4: "entry at next H4 bar open"). ``t_vert`` = entry bar close
    + ``VERTICAL_BARS`` H4 bars (handled by :func:`make_event`).
    """
    t0 = signal.t0
    # Bars closed at or before t0 (for ATR + features).
    h4_pre = h4.loc[h4.index <= pd.Timestamp(t0)]
    if len(h4_pre) < 30:
        return None

    # Next H4 bar after t0 = the entry bar (its open is the entry price).
    future = h4.loc[h4.index > pd.Timestamp(t0)]
    if len(future) == 0:
        return None  # signal at end of history; no entry bar
    entry_bar_close_time = future.index[0]
    entry_px = float(future.iloc[0]["open"])

    # ATR at t0 (closed bars <= t0 only).
    atr0 = _atr_at(h4_pre, 14)

    event = make_event(
        symbol=symbol,
        rule_id=rule_id,
        rule_version=signal.rule_version,
        t0=t0,
        side=signal.side,
        entry_px=entry_px,
        atr0=atr0,
    )

    # M1 bars in (entry_bar_open_time, t_vert] — strictly after entry.
    entry_open_time = entry_bar_close_time - timedelta(hours=4)
    m1_slice = m1_all.loc[
        (m1_all.index > pd.Timestamp(entry_open_time))
        & (m1_all.index <= pd.Timestamp(event.t_vert))
    ]
    # Frame -> Bars for the labeler.
    from .data import Bar
    m1_bars: Bars = [
        Bar(
            open_time=pd.Timestamp(r["open_time"]).to_pydatetime() if not isinstance(r["open_time"], datetime) else r["open_time"],
            close_time=(idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
        )
        for idx, r in m1_slice.iterrows()
    ]

    labeled = label_event(event, m1_bars)

    # Attach feature snapshot (point-in-time, no_lookahead-checked inside).
    d1_pre = d1.loc[d1.index <= pd.Timestamp(t0)]
    snap = compute_snapshot(FeatureContext(t0=t0, h4=h4_pre, d1=d1_pre))

    # Splice the snapshot onto the labeled event via a side-channel dict;
    # LabeledEvent is frozen so we return a tuple-ish via a wrapper.
    return _LabeledWithFeatures(labeled=labeled, features=snap)


@dataclass(frozen=True)
class _LabeledWithFeatures:
    """Internal carrier: a labeled event + its feature snapshot."""

    labeled: LabeledEvent
    features: dict


# ---------------------------------------------------------------------------
# Bulk pipeline.
# ---------------------------------------------------------------------------


def build_events_for_signals(
    signals: list[SignalType],
    *,
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    m1_all: pd.DataFrame,
    symbol: str = "EURUSD",
    rule_id: str = "donchian20",
    progress_every: int = 200,
) -> list[_LabeledWithFeatures]:
    """Build + label events for every signal in ``signals``.

    This is the §4/§6 "M1 barrier-resolution backfill" — the actually
    compute-heavy job. M1 slicing dominates cost; we pre-sort M1 by
    close_time once and slice per event.
    """
    m1_all = m1_all.sort_index()
    out: list[_LabeledWithFeatures] = []
    for i, sig in enumerate(signals):
        if progress_every and i and i % progress_every == 0:
            print(f"[pipeline] labeled {i}/{len(signals)} events", flush=True)
        lwf = label_one_event(sig, h4=h4, d1=d1, m1_all=m1_all, symbol=symbol, rule_id=rule_id)
        if lwf is not None:
            out.append(lwf)
    return out


# ---------------------------------------------------------------------------
# Persistence.
# ---------------------------------------------------------------------------


def persist_events(
    labeled: list[_LabeledWithFeatures],
    db_path: Path = DEFAULT_EVENTS_DB,
) -> int:
    """Persist labeled events to the SQLite events store (§4 schema).

    Returns the number of rows written.
    """
    rows = []
    for lwf in labeled:
        e = lwf.labeled.event
        o = lwf.labeled.outcome
        rows.append(
            (
                e.symbol, e.rule_id, e.rule_version,
                e.t0.isoformat(), int(e.side),
                e.entry_px, e.atr0, e.tp_px, e.sl_px, e.t_vert.isoformat(),
                o.t1.isoformat(), o.kind, int(lwf.labeled.label),
                float(lwf.labeled.r_realized),
                json.dumps(lwf.features, default=str),
                lwf.features.get("feature_ver", feature_hash()),
            )
        )
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO events
              (symbol, rule_id, rule_version, t0, side, entry_px, atr0,
               tp_px, sl_px, t_vert, t1, outcome, label, r_realized,
               features, feature_ver)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_events(db_path: Path = DEFAULT_EVENTS_DB) -> pd.DataFrame:
    """Load all events as a DataFrame indexed by event_id."""
    with _connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM events", conn, parse_dates=["t0", "t1", "t_vert"])
    if len(df):
        df = df.set_index("event_id")
    return df


@dataclass
class EventsStore:
    """Convenience handle over the events DB."""

    db_path: Path = DEFAULT_EVENTS_DB

    def count(self) -> int:
        with _connect(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def label_counts(self) -> dict[int, int]:
        with _connect(self.db_path) as conn:
            rows = conn.execute("SELECT label, COUNT(*) FROM events GROUP BY label").fetchall()
        return {int(r[0]): int(r[1]) for r in rows}

    def outcome_counts(self) -> dict[str, int]:
        with _connect(self.db_path) as conn:
            rows = conn.execute("SELECT outcome, COUNT(*) FROM events GROUP BY outcome").fetchall()
        return {r[0]: int(r[1]) for r in rows}
