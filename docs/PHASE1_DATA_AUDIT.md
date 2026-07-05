# Phase 1 — Step 0 Data Audit

**Status: RESOLVED — EURUSD data ingested from MT5 (Live account), full pipeline run.**

Phase 1 Step 0 of the build prompt requires, before anything else: *"audit
existing TimescaleDB EURUSD H4/M1 coverage — date range, gaps, quality
issues. If coverage is under ~5 years or gaps are severe, stop and
report — do not fetch fresh data from MT5 yourself, that decision isn't
yours to make."*

This document records the audit and the resolution.

## Initial audit (before any fetch)

| Check | Result |
|---|---|
| TimescaleDB container / Postgres listener | **None.** Docker installed but daemon down; nothing on 5432/5433. |
| EURUSD data files in `TRIAGE/` | **None** (Phase 0 was synthetic-only). |
| MT5 terminal | Installed (`IUX Markets MT5`) — off-limits without user authorization. |

**Initial verdict:** 0 years, 0 bars. Hard stop per the prompt.

## User authorization (the data-source decision the prompt reserves)

The prompt reserves the data-source decision to the user. The user
chose **option (c): authorize MT5 ingest**, and confirmed:
- **Symbol:** `EURUSD` (broker string `EURUSD.` on IUX Markets Live).
- **Date range:** 2015-01-01 → today (~10.5y target).
- **Timezone:** detected automatically → **UTC** (IUX bar timestamps
  are UTC; verified empirically — H4 bars open at 04:00/08:00/... UTC).

## Coverage actually retrieved

First attempt on a demo account returned only ~4.6y (2021-11 onward) —
below the ~5y threshold. The user switched to a **Live account**
(login `<redacted>`, IUXMarkets-Live), which exposed full depth.

| Timeframe | Bars | Span | Years | Notes |
|---|---|---|---|---|
| D1 | 3,547 | 2015-01-02 → 2026-07-03 | 11.5 | clean |
| H4 | 18,450 | 2015-01-02 → 2026-07-03 | 11.5 | gap_ratio ≈ 0 (weekends/holidays only) |
| M1 | 4,259,180 | 2015-01-02 → 2026-07-03 | 11.5 | ~1.4M min/year after weekend gaps |

**Final verdict: 11.5 years of EURUSD H4/M1/D1, well above the ~5y
threshold. Gaps are normal forex weekends/holidays only. No quality
issues detected.** Proceeding to Step 2.

## How the data is stored

- Local SQLite at `.triage/bars.sqlite` (Phase 0 deliberately stayed
  off Postgres; the schema mirrors the planned TimescaleDB layout so
  migration is a writer swap).
- Canonical symbol `EURUSD` internally; broker suffix `EURUSD.` mapped
  at the adapter boundary in `src/triage/data/mt5_ingest.py`.
- All timestamps stored as ISO UTC.

## Reproducing

```bash
# Requires the IUX Markets MT5 terminal running and the Live account
# logged in. The ingest is idempotent (INSERT OR REPLACE).
uv run python -c "
from triage.data.mt5_ingest import ingest_eurusd_phase1
ingest_eurusd_phase1()
"
```

The ingest is gated on the user authorization captured above; it does
not run automatically and will not run in CI.
