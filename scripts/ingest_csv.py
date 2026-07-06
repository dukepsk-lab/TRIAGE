"""Load raw OHLCV(+spread) files into the local bar store (.triage/bars.sqlite).

A non-MT5 ingest path so bars produced anywhere (broker export, a Windows
MT5 dump, a data vendor) can feed the exact same pipeline the rest of the
codebase consumes via ``triage.data.mt5_ingest.load_bars``. It writes the
same schema and reuses :func:`triage.data.mt5_ingest.ingest_symbol`, so
downstream code cannot tell the difference between this and a live MT5 pull.

Input layout
------------
A directory of files, one per (symbol, timeframe), named ``<SYMBOL>_<TF>.csv``
or ``.parquet`` — e.g. ``EURUSD_H4.csv``, ``GBPUSD_M1.parquet``. ``TF`` is one
of ``M1`` / ``H4`` / ``D1``.

Columns (case-insensitive; extra columns ignored)
-------------------------------------------------
* time column: one of ``open_time`` / ``time`` / ``timestamp`` / ``date`` —
  ISO-8601 string or epoch seconds. Interpreted as UTC (§ ingest convention).
* ``open`` ``high`` ``low`` ``close`` (required)
* ``tick_volume`` or ``volume`` (optional -> 0 if absent)
* ``spread`` (optional) — not stored in the bars table, but a per-symbol
  median is reported so it can sanity-check the §7 ``COST_BY_SYMBOL`` table.

Usage
-----
    uv run python scripts/ingest_csv.py --dir /path/to/raw
    uv run python scripts/ingest_csv.py --dir raw --symbols EURUSD,GBPUSD
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from triage.data.mt5_ingest import (
    DEFAULT_BAR_DB,
    TF_D1,
    TF_H4,
    TF_M1,
    BarRange,
    coverage_report,
    ingest_symbol,
)

#: Bar length in seconds per canonical timeframe (UTC timestamp math).
_BAR_SECONDS: dict[str, int] = {TF_M1: 60, TF_H4: 4 * 3600, TF_D1: 24 * 3600}

_TIME_ALIASES = ("open_time", "time", "timestamp", "date", "datetime")
_VOL_ALIASES = ("tick_volume", "volume", "vol")


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _pick(cols: list[str], aliases: tuple[str, ...]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def load_file(path: Path, symbol: str, timeframe: str, db_path: Path) -> dict:
    """Parse one OHLCV file and persist it as bars. Returns a small report."""
    df = _read_any(path)
    cols = list(df.columns)
    tcol = _pick(cols, _TIME_ALIASES)
    if tcol is None:
        raise ValueError(f"{path.name}: no time column (looked for {_TIME_ALIASES})")
    for req in ("open", "high", "low", "close"):
        if _pick(cols, (req,)) is None:
            raise ValueError(f"{path.name}: missing required column {req!r}")

    o = _pick(cols, ("open",)); h = _pick(cols, ("high",))
    l = _pick(cols, ("low",)); c = _pick(cols, ("close",))
    vcol = _pick(cols, _VOL_ALIASES)
    scol = _pick(cols, ("spread",))

    # Time: accept epoch seconds or ISO strings, coerce to tz-aware UTC.
    raw_t = df[tcol]
    if pd.api.types.is_numeric_dtype(raw_t):
        open_time = pd.to_datetime(raw_t, unit="s", utc=True)
    else:
        open_time = pd.to_datetime(raw_t, utc=True)

    bar_seconds = _BAR_SECONDS[timeframe]
    close_time = open_time + pd.Timedelta(seconds=bar_seconds)
    vol = df[vcol] if vcol else pd.Series(0, index=df.index)

    rows = [
        (
            ot.isoformat(), ct.isoformat(),
            float(ov), float(hv), float(lv), float(cv), int(vv),
        )
        for ot, ct, ov, hv, lv, cv, vv in zip(
            open_time, close_time, df[o], df[h], df[l], df[c], vol
        )
    ]
    bar_range = BarRange(
        symbol=symbol, timeframe=timeframe, rows=rows,
        first=open_time.min().to_pydatetime() if len(open_time) else None,
        last=open_time.max().to_pydatetime() if len(open_time) else None,
    )
    n = ingest_symbol(bar_range, db_path=db_path)
    cov = coverage_report(symbol, timeframe, db_path=db_path)
    report = {
        "symbol": symbol, "timeframe": timeframe, "rows": n,
        "first": cov.first, "last": cov.last, "years": cov.years,
        "gap_ratio": cov.gap_ratio,
        "median_spread": float(df[scol].median()) if scol else None,
    }
    return report


def ingest_dir(
    directory: Path,
    *,
    symbols: Optional[list[str]] = None,
    db_path: Path = DEFAULT_BAR_DB,
) -> list[dict]:
    """Ingest every ``<SYMBOL>_<TF>.{csv,parquet}`` in ``directory``."""
    reports: list[dict] = []
    valid_tf = {TF_M1, TF_H4, TF_D1}
    files = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in (".csv", ".parquet", ".pq") and "_" in p.stem
    )
    for path in files:
        symbol, _, tf = path.stem.rpartition("_")
        tf = tf.upper()
        if tf not in valid_tf:
            print(f"[ingest_csv] skip {path.name}: unknown timeframe {tf!r}")
            continue
        if symbols and symbol not in symbols:
            continue
        rep = load_file(path, symbol, tf, db_path)
        reports.append(rep)
        sp = f", median_spread={rep['median_spread']}" if rep["median_spread"] is not None else ""
        print(
            f"[ingest_csv] {symbol} {tf}: {rep['rows']} bars, "
            f"{rep['first']} -> {rep['last']} ({rep['years']}y, "
            f"gap_ratio={rep['gap_ratio']}{sp})"
        )
    return reports


def main() -> None:  # pragma: no cover - CLI entry
    ap = argparse.ArgumentParser(description="Ingest OHLCV files into the bar store.")
    ap.add_argument("--dir", required=True, help="Directory of <SYMBOL>_<TF> files")
    ap.add_argument("--symbols", default=None, help="Comma-separated filter (optional)")
    ap.add_argument("--db", default=str(DEFAULT_BAR_DB), help="Bar store path")
    args = ap.parse_args()
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    ingest_dir(Path(args.dir), symbols=syms, db_path=Path(args.db))


if __name__ == "__main__":  # pragma: no cover
    main()
