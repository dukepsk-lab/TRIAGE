"""CSV/Parquet ingest -> bar store round-trip."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.ingest_csv import ingest_dir
from triage.data.mt5_ingest import TF_H4, load_bars


def _write_h4_csv(dirpath: Path, symbol: str, n: int = 30) -> None:
    t0 = pd.Timestamp("2015-01-01T00:00:00Z")
    times = [t0 + pd.Timedelta(hours=4 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "time": [t.isoformat() for t in times],
            "open": [1.10 + 0.001 * i for i in range(n)],
            "high": [1.11 + 0.001 * i for i in range(n)],
            "low": [1.09 + 0.001 * i for i in range(n)],
            "close": [1.105 + 0.001 * i for i in range(n)],
            "tick_volume": [100 + i for i in range(n)],
            "spread": [0.6 for _ in range(n)],
        }
    )
    df.to_csv(dirpath / f"{symbol}_H4.csv", index=False)


def test_ingest_csv_roundtrip(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_h4_csv(raw, "EURUSD", n=30)
    db = tmp_path / "bars.sqlite"

    reports = ingest_dir(raw, db_path=db)
    assert len(reports) == 1
    rep = reports[0]
    assert rep["rows"] == 30
    assert rep["median_spread"] == 0.6

    bars = load_bars("EURUSD", TF_H4, db_path=db)
    assert len(bars) == 30
    # close_time = open_time + 4h, OHLC preserved, ascending by open_time.
    assert (bars[0].close_time - bars[0].open_time).total_seconds() == 4 * 3600
    assert bars[0].open == 1.10
    assert bars[-1].close == pytest.approx(1.105 + 0.001 * 29)


def test_ingest_epoch_seconds(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    t0 = 1420070400  # 2015-01-01T00:00:00Z in epoch seconds
    df = pd.DataFrame(
        {
            "timestamp": [t0 + 14400 * i for i in range(10)],
            "open": [1.0] * 10, "high": [1.1] * 10,
            "low": [0.9] * 10, "close": [1.05] * 10,
        }
    )
    df.to_csv(raw / "GBPUSD_H4.csv", index=False)
    db = tmp_path / "bars.sqlite"
    ingest_dir(raw, db_path=db)
    bars = load_bars("GBPUSD", TF_H4, db_path=db)
    assert len(bars) == 10
    assert bars[0].open_time.year == 2015


import pytest  # noqa: E402  (kept at bottom so helpers read top-down)
