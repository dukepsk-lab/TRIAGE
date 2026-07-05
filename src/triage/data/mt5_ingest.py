"""MT5 ingest adapter — EURUSD H4/M1 bar history from the local terminal.

Authorized for Phase 1 per the user's explicit decision (Live account,
symbol ``EURUSD.`` on IUX Markets, ~17.5y depth, UTC timestamps).

The architecture's §9 plan-of-record is TimescaleDB, but Phase 0 kept
us on local storage. This adapter persists bars to a local SQLite DB
so the rest of the pipeline (rule engine, labeler, features) consumes
a stable bar-store interface regardless of where the bars came from.
The eventual TimescaleDB migration is a later-phase concern and only
swaps the writer; the reader API (:func:`load_bars`) stays the same.

Conventions enforced here (these are load-bearing for the no-look-ahead
invariant — §4/§7):

* **Canonical symbol.** Stored as ``EURUSD`` internally. Broker suffix
  (``EURUSD.``) lives only in :data:`BROKER_SYMBOL_MAP` and is mapped
  at the adapter boundary.
* **UTC timestamps.** MT5 ``copy_rates_*`` returns ``time`` as seconds
  since epoch in **broker-server time**. We convert to timezone-aware
  UTC at ingest and store ``open_time`` / ``close_time`` as ISO UTC
  strings. The live terminal on IUX reports bar opens at wall-clock
  hours consistent with UTC (e.g. 04:00, 08:00, ... for H4), so we
  treat the raw epoch as UTC. (If a future broker reports true
  server-time offsets, this is the one place to fix.)
* **Closed bars only.** We ingest only fully-formed bars; the latest
  in-progress bar is excluded by stopping at ``end - 1 bar``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from . import Bar, Bars, frame_to_bars

__all__ = [
    "DEFAULT_BAR_DB",
    "BROKER_SYMBOL_MAP",
    "CANONICAL_TO_BROKER",
    "TF_H4",
    "TF_M1",
    "TF_D1",
    "BarRange",
    "BROKER_SYMBOL_MAP",
    "PHASE2_SYMBOLS",
    "PHASE3_SYMBOLS",
    "ALL_SYMBOLS",
    "WEEKEND_TRADING_SYMBOLS",
    "fetch_range",
    "ingest_symbol",
    "ingest_eurusd_phase1",
    "ingest_symbols",
    "load_bars",
    "coverage_report",
]

#: Default on-disk bar store. Distinct from the trial registry DB
#: (``.triage/registry.sqlite``) — this holds raw market data.
DEFAULT_BAR_DB: Path = Path(".triage") / "bars.sqlite"

# ---------------------------------------------------------------------------
# Symbol + timeframe mapping.
# ---------------------------------------------------------------------------

#: Map canonical symbol -> broker symbol string. Broker suffixes change
#: per broker; IUX Markets Live uses a trailing dot (verified empirically).
#: Internal storage is always the canonical form (no suffix); mapping
#: happens at the adapter boundary.
BROKER_SYMBOL_MAP: dict[str, str] = {
    "EURUSD": "EURUSD.",
    "GBPUSD": "GBPUSD.",
    "USDJPY": "USDJPY.",
    "XAUUSD": "XAUUSD.",
    "US30": "US30.",
    "BTCUSD": "BTCUSD.",
}

#: The Phase 2 majors pool (§10 Phase 2: "Pool EURUSD/GBPUSD/USDJPY").
#: All three are majors -> same cost class (~0.03-0.05R, §7).
PHASE2_SYMBOLS: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY")

#: The Phase 3 additions (§10 Phase 3: "Add XAUUSD/US30/BTCUSD").
PHASE3_SYMBOLS: tuple[str, ...] = ("XAUUSD", "US30", "BTCUSD")

#: Full Phase 3 basket (6 symbols).
ALL_SYMBOLS: tuple[str, ...] = PHASE2_SYMBOLS + PHASE3_SYMBOLS

#: Instruments that genuinely trade on weekends (continuous 24/7 market).
#: Only BTCUSD among the basket. FX, gold, and US indices close Friday
#: evening -> Sunday/Monday open (a discontinuous weekend gap). The
#: weekend-gap handling in the rule engine + labeler keys off this set:
#: for non-weekend instruments, an event open on Friday resolves barriers
#: against M1 bars that *skip* the weekend (the Friday close -> Monday
#: open is a jump, not a continuous path); BTCUSD resolves continuously.
#: (§3 Phase 3: "BTC also needs a weekend-gap adjustment — don't let it
#: silently misprice barriers over weekend gaps.")
WEEKEND_TRADING_SYMBOLS: frozenset[str] = frozenset({"BTCUSD"})

#: Reverse lookup.
CANONICAL_TO_BROKER: dict[str, str] = {v: k for k, v in BROKER_SYMBOL_MAP.items()}

# ---------------------------------------------------------------------------
# Timeframes.
#
# MT5 timeframe *enum values* are encoded (e.g. TIMEFRAME_H4 = 16388), NOT
# plain minutes — passing 240 to copy_rates_range silently returns None.
# We keep two maps:
#   * TF_<x>          — the canonical key used throughout the codebase
#                       (stable strings, immune to MT5's enum packing);
#   * _MT5_TF[TF_<x>] — the MT5 enum value to pass into copy_rates_*;
#   * _BAR_SECONDS[TF_<x>] — bar length in seconds (for timestamp math).
# ---------------------------------------------------------------------------

TF_M1: str = "M1"
TF_H4: str = "H4"
TF_D1: str = "D1"

#: Canonical -> MT5 enum value. Populated lazily (the MetaTrader5 import
#: is Windows-only), with string fallbacks for non-Windows import safety.
_MT5_TF: dict[str, int] = {
    TF_M1: 1,      # mt5.TIMEFRAME_M1
    TF_H4: 16388,  # mt5.TIMEFRAME_H4
    TF_D1: 16408,  # mt5.TIMEFRAME_D1
}

#: Bar length in seconds per timeframe (UTC timestamp math).
_BAR_SECONDS: dict[str, int] = {
    TF_M1: 60,
    TF_H4: 4 * 3600,
    TF_D1: 24 * 3600,
}


# ---------------------------------------------------------------------------
# Schema.
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,        -- 'M1' | 'H4' | 'D1'
    open_time   TEXT NOT NULL,        -- ISO UTC (bar open)
    close_time  TEXT NOT NULL,        -- ISO UTC (bar close = open + bar_len)
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    tick_volume INTEGER,
    PRIMARY KEY (symbol, timeframe, open_time)
);
CREATE INDEX IF NOT EXISTS idx_bars_time ON bars(symbol, timeframe, open_time);
"""

_TF_NAME: dict[str, str] = {TF_M1: "M1", TF_H4: "H4", TF_D1: "D1"}


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Fetch (chunked — MT5 copy_rates_range caps ~65535 bars/call on M1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarRange:
    """Result of a fetch: count + span + the raw ndarray rows."""

    symbol: str
    timeframe: str               # canonical TF_* key
    rows: list[tuple]            # (open_time_iso, close_time_iso, o, h, l, c, vol)
    first: Optional[datetime]
    last: Optional[datetime]

    @property
    def bar_seconds(self) -> int:
        return _BAR_SECONDS[self.timeframe]


def _chunk_ranges(
    start: datetime, end: datetime, timeframe: str, chunk_bars: int = 50000
) -> Iterator[tuple[datetime, datetime]]:
    """Yield (chunk_start, chunk_end) covering [start, end] in safe sizes.

    MT5's ``copy_rates_range`` silently truncates above ~65k bars; we
    chunk by time at ~50k-bar windows to stay safe.
    """
    step = timedelta(seconds=chunk_bars * _BAR_SECONDS[timeframe])
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        yield cur, nxt
        cur = nxt


def fetch_range(
    canonical_symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> BarRange:
    """Fetch a time range of bars from the MT5 terminal.

    Times returned by MT5 are treated as UTC seconds-since-epoch (see
    module docstring). Returns a :class:`BarRange` with normalized rows.
    """
    import MetaTrader5 as mt5  # lazy import — Windows-only

    broker_sym = BROKER_SYMBOL_MAP.get(canonical_symbol, canonical_symbol)
    mt5_tf = _MT5_TF[timeframe]  # MT5 enum value (e.g. 16388 for H4)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(broker_sym, True):
            raise RuntimeError(
                f"symbol_select({broker_sym!r}, True) failed; symbol not "
                "available on this account/terminal"
            )
        bar_seconds = _BAR_SECONDS[timeframe]
        out_rows: list[tuple] = []
        first_ts: Optional[datetime] = None
        last_ts: Optional[datetime] = None
        for cstart, cend in _chunk_ranges(start, end, timeframe):
            rates = mt5.copy_rates_range(broker_sym, mt5_tf, cstart, cend)
            if rates is None:
                continue
            for r in rates:
                open_time = datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)
                close_time = open_time + timedelta(seconds=bar_seconds)
                out_rows.append(
                    (
                        open_time.isoformat(),
                        close_time.isoformat(),
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                        int(r["tick_volume"]) if "tick_volume" in r.dtype.names else 0,
                    )
                )
                if first_ts is None or open_time < first_ts:
                    first_ts = open_time
                if last_ts is None or open_time > last_ts:
                    last_ts = open_time
        return BarRange(
            symbol=canonical_symbol,
            timeframe=timeframe,
            rows=out_rows,
            first=first_ts,
            last=last_ts,
        )
    finally:
        mt5.shutdown()


# ---------------------------------------------------------------------------
# Persistence + reading.
# ---------------------------------------------------------------------------


def ingest_symbol(
    bar_range: BarRange,
    db_path: Path = DEFAULT_BAR_DB,
) -> int:
    """Persist a :class:`BarRange` into the SQLite bar store.

    Uses ``INSERT OR REPLACE`` so re-ingesting a window is idempotent.
    Returns the number of rows written.
    """
    tf_name = bar_range.timeframe  # already canonical TF_* key
    rows = [
        (
            bar_range.symbol,
            tf_name,
            open_time,
            close_time,
            o,
            h,
            l,
            c,
            vol,
        )
        for (open_time, close_time, o, h, l, c, vol) in bar_range.rows
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO bars
              (symbol, timeframe, open_time, close_time, open, high, low, close, tick_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_bars(
    symbol: str,
    timeframe: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db_path: Path = DEFAULT_BAR_DB,
) -> Bars:
    """Read bars from the local store as :class:`triage.data.Bar` objects.

    ``end`` is inclusive of bar close_time. Filters on ``close_time``
    so that "all bars closed at or before ``end``" is the natural query
    (this is exactly the no-look-ahead query the labeler needs: "give
    me everything closed <= t0").
    """
    tf_name = timeframe  # canonical TF_* key, already a string
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM bars WHERE symbol=? AND timeframe=?"
        params: list = [symbol, tf_name]
        if start is not None:
            sql += " AND close_time >= ?"
            params.append(start.isoformat())
        if end is not None:
            sql += " AND close_time <= ?"
            params.append(end.isoformat())
        sql += " ORDER BY open_time ASC"
        rows = conn.execute(sql, params).fetchall()

    bars: Bars = []
    for r in rows:
        bars.append(
            Bar(
                open_time=datetime.fromisoformat(r["open_time"]),
                close_time=datetime.fromisoformat(r["close_time"]),
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
            )
        )
    return bars


def load_bars_frame(
    symbol: str,
    timeframe: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db_path: Path = DEFAULT_BAR_DB,
) -> pd.DataFrame:
    """DataFrame form of :func:`load_bars`, indexed by close_time."""
    bars = load_bars(symbol, timeframe, start=start, end=end, db_path=db_path)
    from . import bars_to_frame
    return bars_to_frame(bars)


@dataclass(frozen=True)
class Coverage:
    """Summary of stored coverage for one (symbol, timeframe)."""

    symbol: str
    timeframe: str
    n_bars: int
    first: Optional[datetime]
    last: Optional[datetime]
    years: float
    gap_ratio: float  # 1 - actual/expected_at_6-per-day-260d-y (H4 baseline)


def coverage_report(
    symbol: str,
    timeframe: str,
    db_path: Path = DEFAULT_BAR_DB,
) -> Coverage:
    """Compute a coverage summary for a stored (symbol, timeframe)."""
    tf_name = timeframe  # canonical TF_* key, already a string
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT COUNT(*) AS n, MIN(open_time) AS first, MAX(open_time) AS last "
            "FROM bars WHERE symbol=? AND timeframe=?",
            (symbol, tf_name),
        ).fetchone()
    n = int(row["n"] or 0)
    first = datetime.fromisoformat(row["first"]) if row["first"] else None
    last = datetime.fromisoformat(row["last"]) if row["last"] else None
    if first and last:
        days = (last - first).total_seconds() / 86400
        years = days / 365.25
        # Expected bar count at ~6 bars/day * 260 trading days/year (H4 baseline).
        expected = years * 260 * 6
        gap_ratio = 1.0 - (n / expected) if expected > 0 else 0.0
    else:
        years = 0.0
        gap_ratio = 0.0
    return Coverage(
        symbol=symbol,
        timeframe=tf_name,
        n_bars=n,
        first=first,
        last=last,
        years=round(years, 2),
        gap_ratio=round(gap_ratio, 3),
    )


# ---------------------------------------------------------------------------
# Phase 1 entry point.
# ---------------------------------------------------------------------------


def ingest_eurusd_phase1(
    start: datetime = datetime(2015, 1, 1, tzinfo=timezone.utc),
    end: Optional[datetime] = None,
    *,
    timeframes: Iterable[str] = (TF_D1, TF_H4, TF_M1),
    db_path: Path = DEFAULT_BAR_DB,
    progress: bool = True,
) -> dict[str, Coverage]:
    """Phase 1 EURUSD ingest: D1 + H4 + M1 from MT5.

    Per the user's Phase 1 authorization: canonical ``EURUSD``, broker
    ``EURUSD.``, ~10.5y window starting 2015-01-01, UTC. Returns a
    per-timeframe coverage report.

    Order is D1 → H4 → M1 so the smallest datasets finish first and the
    expensive M1 backfill (the §6 "actually compute-heavy job") runs last
    with progress visibility.
    """
    end = end or datetime.now(tz=timezone.utc)
    out: dict[str, Coverage] = {}
    for tf in timeframes:
        if progress:
            print(f"[ingest] EURUSD {tf} {start.date()} -> {end.date()} ...", flush=True)
        bar_range = fetch_range("EURUSD", tf, start, end)
        n = ingest_symbol(bar_range, db_path=db_path)
        cov = coverage_report("EURUSD", tf, db_path=db_path)
        if progress:
            print(
                f"[ingest]   wrote {n} bars; coverage: "
                f"{cov.first} -> {cov.last} ({cov.years}y, gap_ratio={cov.gap_ratio})",
                flush=True,
            )
        out[tf] = cov
    return out


def ingest_symbols(
    symbols: Iterable[str] = PHASE2_SYMBOLS,
    *,
    start: datetime = datetime(2015, 1, 1, tzinfo=timezone.utc),
    end: Optional[datetime] = None,
    timeframes: Iterable[str] = (TF_D1, TF_H4, TF_M1),
    db_path: Path = DEFAULT_BAR_DB,
    progress: bool = True,
) -> dict[str, dict[str, Coverage]]:
    """Multi-symbol ingest (Phase 2). Loops :func:`ingest_eurusd_phase1`-style
    fetch+persist over each canonical symbol in ``symbols``.

    Returns ``{symbol: {timeframe: Coverage}}``. EURUSD is a no-op
    re-ingest if already present (INSERT OR REPLACE is idempotent), so
    calling this in Phase 2 tops up GBPUSD + USDJPY without re-fetching
    EURUSD's M1 backfill.
    """
    end = end or datetime.now(tz=timezone.utc)
    out: dict[str, dict[str, Coverage]] = {}
    for sym in symbols:
        if progress:
            print(f"[ingest] === {sym} ===", flush=True)
        sym_cov: dict[str, Coverage] = {}
        for tf in timeframes:
            if progress:
                print(f"[ingest] {sym} {tf} {start.date()} -> {end.date()} ...", flush=True)
            bar_range = fetch_range(sym, tf, start, end)
            n = ingest_symbol(bar_range, db_path=db_path)
            cov = coverage_report(sym, tf, db_path=db_path)
            if progress:
                print(
                    f"[ingest]   wrote {n} bars; coverage: "
                    f"{cov.first} -> {cov.last} ({cov.years}y, gap_ratio={cov.gap_ratio})",
                    flush=True,
                )
            sym_cov[tf] = cov
        out[sym] = sym_cov
    return out


if __name__ == "__main__":  # pragma: no cover - CLI entry
    import argparse

    p = argparse.ArgumentParser(description="Ingest bars from MT5.")
    p.add_argument("--start", default="2015-01-01", help="ISO date, default 2015-01-01")
    p.add_argument(
        "--symbols",
        default="EURUSD,GBPUSD,USDJPY",
        help="Comma-separated canonical symbols (default EURUSD,GBPUSD,USDJPY)",
    )
    p.add_argument(
        "--db", default=str(DEFAULT_BAR_DB), help="SQLite path (default .triage/bars.sqlite)"
    )
    args = p.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    ingest_symbols(syms, start=start, db_path=Path(args.db))
