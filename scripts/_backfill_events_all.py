"""Backfill Donchian + htf_pullback events (M1 first-touch labeled) for the
full 6-symbol basket. Idempotent per (symbol, rule) via backfill_symbol_events.
"""
from __future__ import annotations

import time

from run_phase3 import backfill_symbol_events
from triage.pipeline import load_events, DEFAULT_EVENTS_DB

FULL6 = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "BTCUSD"]


def main() -> None:
    for sym in FULL6:
        t0 = time.time()
        print(f"[backfill] === {sym} ===", flush=True)
        n = backfill_symbol_events(sym)
        print(f"[backfill] {sym}: +{n} events in {round(time.time()-t0,1)}s", flush=True)

    df = load_events(DEFAULT_EVENTS_DB)
    print("\n[backfill] TOTALS", flush=True)
    print("total events:", len(df), flush=True)
    g = df.groupby(["symbol", "rule_id"]).size()
    print(g.to_string(), flush=True)
    br = df.groupby("symbol")["label"].mean().round(3)
    print("\nper-symbol base rate:\n", br.to_string(), flush=True)
    print("\noutcome counts:", df["outcome"].value_counts().to_dict(), flush=True)
    print("[backfill] DONE", flush=True)


if __name__ == "__main__":
    main()
