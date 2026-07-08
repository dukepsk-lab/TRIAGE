"""Live SHADOW paper loop — majors x donchian, fixed sizing (§8/§10 Phase 4).

Runs the frozen PaperStrategy forward on live MT5 data WITHOUT placing any
orders. Each pass: top up recent bars from the terminal, detect new Donchian
signals on closed H4 bars, compute the point-in-time feature snapshot (incl. the
cross-sectional Tier 4 rank across the majors), call the SAME decide() the live
executor would, and log the decision. Taken decisions are resolved against M1
first-touch once their 96h window has fully elapsed — reusing the exact backfill
labeler, so the shadow book is in parity with the backtest by construction.

Zero execution risk: nothing is sent to the broker. This is the safe first
stage of a forward test (Stage A). Stage B (real demo orders) is a thin
order_send adapter on top of the same decisions.

Usage (on the VPS, after bootstrap):
    uv run python scripts/live_shadow.py --once           # one pass (for a scheduler)
    uv run python scripts/live_shadow.py --loop           # sleep until each H4 close
    uv run python scripts/live_shadow.py --once --no-ingest  # offline test on the store
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from triage.data.mt5_ingest import (
    TF_D1, TF_H4, TF_M1, fetch_range, ingest_symbol, load_bars_frame,
)
from triage.features import FEATURE_COLUMNS
from triage.features.tier4 import attach_tier4_cross_sectional
from triage.gates import cost_for_symbol
from triage.pipeline import build_events_for_signals
from triage.rules import donchian_backfill

MAJORS = ["EURUSD", "GBPUSD", "USDJPY"]
H4 = timedelta(hours=4)
VERT = timedelta(hours=96)          # 24 H4 bars
REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / ".triage" / "majors_donchian_strategy.pkl"
DEFAULT_DB = REPO / ".triage" / "shadow.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    symbol TEXT NOT NULL, rule_id TEXT NOT NULL, t0 TEXT NOT NULL,
    ts_logged TEXT NOT NULL, side INTEGER, p_win REAL, tau0 REAL,
    taken INTEGER, reason TEXT, entry_px REAL, tp_px REAL, sl_px REAL,
    t_vert TEXT, outcome TEXT, r_realized REAL, pnl_r REAL, resolved INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, rule_id, t0)
);
"""


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db))
    c.executescript(_SCHEMA)
    return c


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=timezone.utc))


# ---------------------------------------------------------------------------
# Bar top-up.
# ---------------------------------------------------------------------------


def topup_bars(symbols, *, m1_days: int = 30, htf_days: int = 60) -> None:
    """Fetch + upsert recent bars (idempotent INSERT OR REPLACE)."""
    end = datetime.now(tz=timezone.utc)
    for sym in symbols:
        for tf, days in ((TF_D1, htf_days), (TF_H4, htf_days), (TF_M1, m1_days)):
            start = end - timedelta(days=days)
            ingest_symbol(fetch_range(sym, tf, start, end))


# ---------------------------------------------------------------------------
# Recent signal -> feature frame (parity with the backfill pipeline).
# ---------------------------------------------------------------------------


def build_recent_events(symbols, cutoff: pd.Timestamp, now: pd.Timestamp, *, m1_days: int = 30) -> pd.DataFrame:
    """Feature rows (incl. cross-sectional Tier 4) for Donchian signals whose
    signal-bar close t0 is in ``[cutoff, now]``. Reuses backfill_signals +
    build_events_for_signals so features are identical to training. Outcomes on
    immature signals are premature and must be ignored (resolution happens
    later, once M1 has elapsed)."""
    rows = []
    m1_start = (now - timedelta(days=m1_days)).to_pydatetime()
    for sym in symbols:
        h4 = load_bars_frame(sym, TF_H4)
        d1 = load_bars_frame(sym, TF_D1)
        if len(h4) < 60 or len(d1) < 55:
            continue
        sigs = [s for s in donchian_backfill.backfill_signals(h4, d1)
                if cutoff <= pd.Timestamp(s.t0) <= now]
        if not sigs:
            continue
        m1 = load_bars_frame(sym, TF_M1, start=m1_start)
        lwfs = build_events_for_signals(sigs, h4=h4, d1=d1, m1_all=m1, symbol=sym, progress_every=0)
        for lwf in lwfs:
            e, f = lwf.labeled.event, lwf.features
            row = {"symbol": sym, "rule_id": "donchian20", "t0": e.t0, "side": int(e.side),
                   "entry_px": e.entry_px, "tp_px": e.tp_px, "sl_px": e.sl_px, "t_vert": e.t_vert}
            row.update({k: f[k] for k in FEATURE_COLUMNS if k in f})
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = attach_tier4_cross_sectional(df)
    return df


# ---------------------------------------------------------------------------
# One pass.
# ---------------------------------------------------------------------------


def run_once(strat, db: Path, *, lookback_days: int, no_ingest: bool) -> None:
    now = _now()
    if not no_ingest:
        print(f"[{now}] topping up bars...", flush=True)
        topup_bars(MAJORS, m1_days=max(30, lookback_days + 6))

    # Only act on CLOSED H4 bars (t0 <= last completed boundary).
    last_closed = now.floor("4h") - H4 if now.floor("4h") == now else now.floor("4h")
    cutoff = now - timedelta(days=lookback_days)
    df = build_recent_events(MAJORS, cutoff, last_closed, m1_days=max(30, lookback_days + 6))

    con = _connect(db)
    cur = con.cursor()
    existing = {(r[0], r[1], r[2]) for r in cur.execute(
        "SELECT symbol, rule_id, t0 FROM decisions").fetchall()}

    # --- Decision pass: log any NEW signal ---
    n_new = n_taken = 0
    if len(df):
        preds = strat.predict_p(df)
        # open positions per symbol (for the §8 single-position overlay)
        open_rows = cur.execute(
            "SELECT symbol, t0, t_vert FROM decisions WHERE taken=1 AND resolved=0").fetchall()
        for i, r in df.reset_index(drop=True).iterrows():
            key = (r["symbol"], "donchian20", pd.Timestamp(r["t0"]).isoformat())
            if key in existing:
                continue
            t0 = pd.Timestamp(r["t0"])
            busy = any(o[0] == r["symbol"] and pd.Timestamp(o[1]) < t0 < pd.Timestamp(o[2])
                       for o in open_rows)
            open_risk = strat.base_risk * sum(
                1 for o in open_rows if pd.Timestamp(o[1]) < t0 < pd.Timestamp(o[2]))
            dec = strat.decide(r, symbol_busy=busy, open_risk=open_risk)
            cur.execute(
                "INSERT OR IGNORE INTO decisions (symbol, rule_id, t0, ts_logged, side, "
                "p_win, tau0, taken, reason, entry_px, tp_px, sl_px, t_vert) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["symbol"], "donchian20", t0.isoformat(), now.isoformat(), int(r["side"]),
                 float(preds[i]), strat.tau0, int(dec.take), dec.reason,
                 float(r["entry_px"]), float(r["tp_px"]), float(r["sl_px"]),
                 pd.Timestamp(r["t_vert"]).isoformat()))
            n_new += 1
            n_taken += int(dec.take)
    con.commit()

    # --- Resolution pass: finalize matured taken decisions via M1 first-touch ---
    n_resolved = _resolve_matured(con, now)
    con.commit()

    _print_summary(con, now, n_new, n_taken, n_resolved)
    con.close()


def _resolve_matured(con: sqlite3.Connection, now: pd.Timestamp) -> int:
    """Resolve taken, unresolved decisions whose 96h window has fully elapsed,
    by re-labeling the signal with the exact backfill labeler (parity)."""
    from triage.pipeline import label_one_event
    from triage.rules import DonchianSignal, RULE_VERSION

    cur = con.cursor()
    todo = cur.execute(
        "SELECT symbol, t0, t_vert, side FROM decisions "
        "WHERE taken=1 AND resolved=0").fetchall()
    n = 0
    frames: dict[str, tuple] = {}
    for sym, t0s, tvs, side in todo:
        if now < pd.Timestamp(tvs) + timedelta(hours=1):  # window not elapsed yet
            continue
        if sym not in frames:
            frames[sym] = (load_bars_frame(sym, TF_H4), load_bars_frame(sym, TF_D1),
                           load_bars_frame(sym, TF_M1, start=(pd.Timestamp(t0s) - H4).to_pydatetime()))
        h4, d1, m1 = frames[sym]
        sig = DonchianSignal(t0=pd.Timestamp(t0s).to_pydatetime(), side=int(side),
                             rule_version=RULE_VERSION)
        lwf = label_one_event(sig, h4=h4, d1=d1, m1_all=m1, symbol=sym, rule_id="donchian20")
        if lwf is None:
            continue
        r = float(lwf.labeled.r_realized)
        pnl = r - cost_for_symbol(sym)
        cur.execute(
            "UPDATE decisions SET outcome=?, r_realized=?, pnl_r=?, resolved=1 "
            "WHERE symbol=? AND rule_id='donchian20' AND t0=?",
            (lwf.labeled.outcome.kind, r, pnl, sym, pd.Timestamp(t0s).isoformat()))
        n += 1
    return n


def _print_summary(con: sqlite3.Connection, now, n_new, n_taken, n_resolved) -> None:
    cur = con.cursor()
    tot = cur.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    taken = cur.execute("SELECT COUNT(*) FROM decisions WHERE taken=1").fetchone()[0]
    open_ = cur.execute("SELECT COUNT(*) FROM decisions WHERE taken=1 AND resolved=0").fetchone()[0]
    closed = cur.execute("SELECT COUNT(*) FROM decisions WHERE taken=1 AND resolved=1").fetchone()[0]
    row = cur.execute("SELECT COALESCE(SUM(pnl_r),0), COALESCE(AVG(pnl_r),0), "
                      "COALESCE(SUM(CASE WHEN pnl_r>0 THEN 1 ELSE 0 END),0) "
                      "FROM decisions WHERE resolved=1").fetchone()
    total_r, mean_r, wins = row
    wr = (wins / closed) if closed else 0.0
    print(f"[{now}] +{n_new} new signals ({n_taken} taken this pass) | "
          f"logged={tot} taken={taken} open={open_} closed={closed}", flush=True)
    print(f"          shadow book (closed): total_R={total_r:+.3f} mean_R={mean_r:+.4f} "
          f"win_rate={wr:.3f}", flush=True)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _seconds_to_next_h4() -> float:
    now = datetime.now(tz=timezone.utc)
    nxt = (now.replace(minute=0, second=0, microsecond=0)
           + timedelta(hours=(4 - now.hour % 4)))
    return max(30.0, (nxt - now).total_seconds() + 20)  # +20s buffer past the close


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--lookback-days", type=int, default=10,
                    help="how far back to scan for signals each pass")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    ap.add_argument("--loop", action="store_true", help="sleep until each H4 close and repeat")
    ap.add_argument("--no-ingest", action="store_true", help="skip MT5 top-up (offline test)")
    args = ap.parse_args()

    strat = pickle.loads(Path(args.model).read_bytes())
    print(f"loaded strategy: method={strat._method} tau0={strat.tau0} base_risk={strat.base_risk}")
    db = Path(args.db)

    if args.loop:
        print("shadow loop started (Ctrl-C to stop).", flush=True)
        while True:
            try:
                run_once(strat, db, lookback_days=args.lookback_days, no_ingest=args.no_ingest)
            except Exception as e:  # keep the loop alive; log and continue
                print(f"[pass error] {e!r}", flush=True)
            s = _seconds_to_next_h4()
            print(f"sleeping {s/3600:.2f}h until next H4 close...", flush=True)
            time.sleep(s)
    else:
        run_once(strat, db, lookback_days=args.lookback_days, no_ingest=args.no_ingest)


if __name__ == "__main__":
    main()
