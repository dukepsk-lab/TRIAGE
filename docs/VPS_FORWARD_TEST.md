# Forward Test on a Windows VPS — Shadow Paper (Stage A)

Run the frozen **majors × donchian** strategy forward on a demo account, with
**no orders placed** (shadow mode). This is the safe first stage: it validates
that the live decision path matches the backtest (the §11 parity concern) and
starts accumulating the live calibration data that G5 needs — at zero execution
risk.

## Prerequisites on the VPS

- **MT5 terminal installed and logged into your demo account, and running.**
  The Python API attaches to a live terminal; if it's closed, ingest fails.
- **Git + [uv](https://docs.astral.sh/uv/) installed.**
- **Check out to an ASCII path** (e.g. `C:\triage`). Do *not* put it under a
  folder with non-Latin characters — that triggers a venv startup bug
  (`site` decodes the editable `.pth` under cp1252). An ASCII path avoids it
  entirely, and then plain `uv run ...` works with no workarounds.

## One-time setup

```bat
git clone https://github.com/dukepsk-lab/TRIAGE.git C:\triage
cd C:\triage
git checkout claude/validate-architecture-docs-xx277i   :: work is on this branch, not main
uv sync --extra dev
uv run pytest tests/ -q                                 :: sanity: should be green
```

## Bootstrap (ingest → backfill → train the frozen model)

The trained model and bar data are **not** in git (gitignored). Build them on
the VPS from the terminal — one command, one time (the M1 backfill is the heavy
part, tens of minutes; it's idempotent so a re-run resumes):

```bat
uv run python scripts/bootstrap_forward.py
```

This ingests majors D1/H4/M1, backfills the events, and writes the frozen
strategy to `.triage\majors_donchian_strategy.pkl`.

## Run the shadow test

Test a single pass first (safe, no orders, verifies wiring):

```bat
uv run python scripts/live_shadow.py --once
```

Then run it forward — it sleeps until each H4 close (00/04/08/12/16/20 UTC),
tops up bars, logs decisions, and resolves matured trades:

```bat
uv run python scripts/live_shadow.py --loop
```

Keep it running in a persistent session. For a real deployment, wrap it as a
service (e.g. **NSSM**) or a **Task Scheduler** job that runs
`scripts/live_shadow.py --once` at 5 minutes past each H4 boundary — `--once` is
idempotent, so a scheduler is the robust option.

## What it does each pass

1. Tops up recent bars from the terminal (idempotent upsert).
2. Detects new Donchian signals on **closed** H4 bars for EUR/GBP/USDJPY.
3. Computes the point-in-time feature snapshot (incl. the cross-sectional Tier 4
   rank across the majors) and calls the **same** `decide()` a live executor
   would — fixed sizing, τ0 filter, §8 overlays (per-symbol single position,
   aggregate risk cap).
4. Logs every decision (taken or not) to `.triage\shadow.sqlite`.
5. Resolves taken decisions against M1 first-touch once their 96h window has
   fully elapsed — using the exact backfill labeler, so the shadow book is in
   parity with the backtest.

Nothing is sent to the broker.

## Inspecting the shadow book

```bat
uv run python -c "import sqlite3,pandas as pd; \
c=sqlite3.connect('.triage/shadow.sqlite'); \
print(pd.read_sql('SELECT symbol,t0,side,round(p_win,3) p,taken,reason,outcome,round(pnl_r,3) pnl,resolved FROM decisions ORDER BY t0 DESC LIMIT 40', c).to_string(index=False))"
```

Watch for: (a) **parity** — do live decisions and outcomes look like the
backtest's rate/mix? (b) **calibration drift** — realized win-rate of taken
trades vs mean predicted `p_win` (the §10 Phase-4 CUSUM signal). A persistent gap
is the earliest structural-break alarm and the thing that would eventually gate
whether probability sizing (rung 3) can be turned on.

## What to expect

The backtest dry-run (`docs/PAPER_HARNESS.md`) showed the filter turning the
naked rule's −91R into +13R on the held-out fold (Sharpe +0.64, +11.7% equity at
1% risk) — modest but real. Forward numbers will be noisier at first (few signals
per week per symbol). Judge it over months, not days, and on **parity + G1-style
uplift vs take-all**, not on any single week.

## Stage B (later): real demo orders

Once shadow is clean, Stage B is a thin adapter that sends the *taken* decisions
to the terminal via `mt5.order_send` with TP/SL attached (barrier prices are
already on each decision row), letting the broker resolve fills so you capture
real spread/slippage. Same `decide()`; only the execution sink changes. Keep
**fixed sizing** until G5 clears on the accumulated live data.
