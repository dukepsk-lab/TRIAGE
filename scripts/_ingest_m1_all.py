"""One-off M1 backfill runner for the full 6-symbol basket (overnight job).

Ingests M1 only (D1/H4 already loaded). Per-symbol timing + coverage so the
background log is legible. Idempotent (INSERT OR REPLACE).
"""
from __future__ import annotations

import time

from triage.data.mt5_ingest import (
    ALL_SYMBOLS,
    TF_M1,
    coverage_report,
    fetch_range,
    ingest_symbol,
)
from datetime import datetime, timezone

START = datetime(2015, 1, 1, tzinfo=timezone.utc)


def main() -> None:
    end = datetime.now(tz=timezone.utc)
    for sym in ALL_SYMBOLS:
        t0 = time.time()
        print(f"[m1] {sym}: fetching M1 {START.date()} -> {end.date()} ...", flush=True)
        br = fetch_range(sym, TF_M1, START, end)
        n = ingest_symbol(br)
        cov = coverage_report(sym, TF_M1)
        print(
            f"[m1] {sym}: wrote {n} bars in {round(time.time()-t0,1)}s; "
            f"coverage {cov.first} -> {cov.last} ({cov.years}y, n={cov.n_bars})",
            flush=True,
        )
    print("[m1] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
