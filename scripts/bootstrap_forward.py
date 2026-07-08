"""One-time bootstrap for a forward test (run on the VPS after `uv sync`).

Requires a running MT5 terminal logged into the (demo) account. Produces
everything `scripts/live_shadow.py` needs:

  1. ingest majors D1/H4/M1 bar history from the terminal,
  2. backfill Donchian + pullback events (M1 first-touch labeled),
  3. train + save the frozen majors x donchian PaperStrategy (.pkl).

The M1 backfill is the heavy step (tens of minutes for 3 majors); it is
idempotent, so a re-run resumes cheaply.

    uv run python scripts/bootstrap_forward.py
"""
from __future__ import annotations

import time

from triage.data.mt5_ingest import ingest_symbols

from run_phase3 import backfill_symbol_events
import run_paper_majors_donchian

MAJORS = ["EURUSD", "GBPUSD", "USDJPY"]


def main() -> None:
    t0 = time.time()
    print("=" * 66)
    print("BOOTSTRAP FORWARD TEST — ingest -> backfill -> train frozen model")
    print("=" * 66)

    print("\n[1/3] Ingesting majors D1/H4/M1 from the MT5 terminal...")
    ingest_symbols(MAJORS)  # default timeframes D1, H4, M1

    print("\n[2/3] Backfilling events (Donchian + pullback, M1 labeled)...")
    for s in MAJORS:
        backfill_symbol_events(s)

    print("\n[3/3] Training + saving the frozen strategy (.triage/majors_donchian_strategy.pkl)...")
    run_paper_majors_donchian.main()

    print(f"\nBootstrap complete in {(time.time()-t0)/60:.1f} min. "
          f"Now run:  uv run python scripts/live_shadow.py --loop")


if __name__ == "__main__":
    main()
