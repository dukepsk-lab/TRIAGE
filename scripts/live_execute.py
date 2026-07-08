"""Live EXECUTE loop — Stage B: real demo orders for majors x donchian (§9).

Same frozen strategy and decide() as the shadow loop, but this one PLACES ORDERS
on the connected MT5 terminal. At each H4 close it detects Donchian signals on
the just-closed bar (features from bars <= t0, so no entry-bar wait), decides
with fixed sizing + the §8 overlays, sizes the position from a risk budget, and
sends a market order with TP/SL attached. It closes positions at the 96h vertical
barrier and reconciles broker-closed (TP/SL) positions into an orders log.

SAFETY:
  * ``--dry-run`` is the DEFAULT: it computes and logs intended orders but sends
    nothing. ``--live`` is required to actually transmit.
  * It trades whatever account the running terminal is logged into. **Point it at
    a DEMO terminal.** Never run ``--live`` against a live account.
  * Fixed sizing only (rung 2); probability sizing waits on G5 (§8).

    uv run python scripts/live_execute.py --once            # dry-run, one pass
    uv run python scripts/live_execute.py --loop --live     # forward, real demo orders
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
    TF_D1, TF_H4, TF_M1, fetch_range, ingest_symbol, load_bars_frame, BROKER_SYMBOL_MAP,
)
from triage.execution.broker import (
    TRIAGE_MAGIC, account_equity, close_position, compute_lot, open_positions,
    place_market_order,
)
from triage.features import FEATURE_COLUMNS
from triage.features.service import FeatureContext, compute_snapshot
from triage.features.tier4 import attach_tier4_cross_sectional
from triage.labeling import SL_MULT, TP_MULT
from triage.rules import donchian_backfill

MAJORS = ["EURUSD", "GBPUSD", "USDJPY"]
H4 = timedelta(hours=4)
VERT = timedelta(hours=96)
REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / ".triage" / "majors_donchian_strategy.pkl"
DEFAULT_DB = REPO / ".triage" / "orders.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    symbol TEXT NOT NULL, t0 TEXT NOT NULL, ts_sent TEXT, side INTEGER,
    p_win REAL, lot REAL, entry_px REAL, sl_px REAL, tp_px REAL, t_vert TEXT,
    ticket INTEGER, dry_run INTEGER, status TEXT, profit REAL,
    PRIMARY KEY (symbol, t0)
);
"""


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db))
    c.executescript(_SCHEMA)
    return c


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=timezone.utc))


def topup(symbols, m1_days=15, htf_days=40):
    end = datetime.now(tz=timezone.utc)
    for sym in symbols:
        for tf, days in ((TF_D1, htf_days), (TF_H4, htf_days), (TF_M1, m1_days)):
            ingest_symbol(fetch_range(sym, tf, end - timedelta(days=days), end))


def _atr14(h4_pre: pd.DataFrame) -> float:
    from triage.features.service import _atr
    return float(_atr(h4_pre["close"], h4_pre["high"], h4_pre["low"], 14).iloc[-1])


def signals_at(t0: pd.Timestamp, symbols) -> pd.DataFrame:
    """One row per major whose Donchian rule fires exactly at closed bar ``t0``,
    with the point-in-time feature snapshot + cross-sectional Tier 4 rank.
    No entry bar needed (features are all <= t0)."""
    rows = []
    for sym in symbols:
        h4 = load_bars_frame(sym, TF_H4)
        d1 = load_bars_frame(sym, TF_D1)
        h4_pre = h4[h4.index <= t0]
        d1_pre = d1[d1.index <= t0]
        if len(h4_pre) < 60 or len(d1_pre) < 55 or h4_pre.index[-1] != t0:
            continue
        sig = next((s for s in donchian_backfill.backfill_signals(h4, d1)
                    if pd.Timestamp(s.t0) == t0), None)
        if sig is None:
            continue
        snap = compute_snapshot(FeatureContext(t0=t0.to_pydatetime(), h4=h4_pre, d1=d1_pre))
        row = {"symbol": sym, "rule_id": "donchian20", "t0": t0, "side": int(sig.side),
               "atr0": _atr14(h4_pre)}
        row.update({k: snap[k] for k in FEATURE_COLUMNS if k in snap})
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = attach_tier4_cross_sectional(df)
    return df


def run_once(strat, db, *, dry_run, no_ingest, risk_frac):
    now = _now()
    if not no_ingest:
        topup(MAJORS)
    last_closed = now.floor("4h") - H4 if now.floor("4h") == now else now.floor("4h")

    con = _connect(db)
    cur = con.cursor()

    # --- Manage existing positions: vertical-barrier close + reconcile ---
    _manage_positions(con, now, dry_run)

    # --- New signals on the just-closed bar ---
    df = signals_at(last_closed, MAJORS)
    n_taken = 0
    if len(df):
        preds = strat.predict_p(df)
        try:
            equity = account_equity()
        except Exception:
            equity = 0.0
        risk_amount = risk_frac * equity if equity > 0 else 0.0
        our = {p.symbol: p for p in open_positions()}
        for i, r in df.reset_index(drop=True).iterrows():
            sym, t0 = r["symbol"], pd.Timestamp(r["t0"])
            if cur.execute("SELECT 1 FROM orders WHERE symbol=? AND t0=?",
                           (sym, t0.isoformat())).fetchone():
                continue
            busy = BROKER_SYMBOL_MAP.get(sym, sym) in our
            open_risk = strat.base_risk * len(our)
            dec = strat.decide(r, symbol_busy=busy, open_risk=open_risk)
            if not dec.take:
                continue
            side = int(r["side"])
            entry = _ref_price(sym, side)
            atr0 = float(r["atr0"])
            tp = entry + side * TP_MULT * atr0
            sl = entry - side * SL_MULT * atr0
            info = _symbol_econ(sym)
            lot = compute_lot(entry_px=entry, sl_px=sl, risk_amount=risk_amount, **info) if (risk_amount > 0 and info) else 0.0
            res = place_market_order(symbol=sym, side=side, lot=lot, sl=sl, tp=tp, dry_run=dry_run)
            cur.execute(
                "INSERT OR IGNORE INTO orders (symbol,t0,ts_sent,side,p_win,lot,entry_px,"
                "sl_px,tp_px,t_vert,ticket,dry_run,status,profit) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sym, t0.isoformat(), now.isoformat(), side, float(preds[i]), res.lot,
                 res.price, sl, tp, (t0 + VERT).isoformat(), res.ticket, int(dry_run),
                 "open" if res.ok else "rejected", None))
            n_taken += int(res.ok)
            print(f"  {'DRY' if dry_run else 'LIVE'} {sym} side={side} lot={res.lot} "
                  f"entry~{res.price:.5f} sl={sl:.5f} tp={tp:.5f} p={preds[i]:.3f} [{res.comment}]", flush=True)
    con.commit()
    _summary(con, now, n_taken, dry_run)
    con.close()


def _ref_price(symbol, side):
    import MetaTrader5 as mt5
    mt5.initialize()
    bsym = BROKER_SYMBOL_MAP.get(symbol, symbol)
    mt5.symbol_select(bsym, True)
    t = mt5.symbol_info_tick(bsym)
    return float(t.ask if side == 1 else t.bid) if t else 0.0


def _symbol_econ(symbol):
    import MetaTrader5 as mt5
    mt5.initialize()
    bsym = BROKER_SYMBOL_MAP.get(symbol, symbol)
    info = mt5.symbol_info(bsym)
    if info is None:
        return None
    return {"tick_value": float(info.trade_tick_value), "tick_size": float(info.trade_tick_size),
            "vol_min": float(info.volume_min), "vol_max": float(info.volume_max),
            "vol_step": float(info.volume_step)}


def _manage_positions(con, now, dry_run):
    cur = con.cursor()
    open_orders = cur.execute(
        "SELECT symbol,t0,t_vert,ticket FROM orders WHERE status='open' AND dry_run=0").fetchall()
    if not open_orders:
        return
    live = {p.ticket: p for p in open_positions()}
    for sym, t0, tv, ticket in open_orders:
        if ticket in live:
            if now >= pd.Timestamp(tv):  # vertical barrier -> close at market
                res = close_position(live[ticket], dry_run=dry_run)
                if res.ok and not dry_run:
                    cur.execute("UPDATE orders SET status='closed_vert' WHERE symbol=? AND t0=?",
                                (sym, t0))
        else:  # not open at broker -> TP/SL closed; reconcile realized profit
            profit = _closed_profit(ticket)
            cur.execute("UPDATE orders SET status='closed', profit=? WHERE symbol=? AND t0=?",
                        (profit, sym, t0))
    con.commit()


def _closed_profit(ticket):
    try:
        import MetaTrader5 as mt5
        mt5.initialize()
        deals = mt5.history_deals_get(position=ticket)
        return float(sum(d.profit for d in deals)) if deals else None
    except Exception:
        return None


def _summary(con, now, n_taken, dry_run):
    cur = con.cursor()
    tot = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    opn = cur.execute("SELECT COUNT(*) FROM orders WHERE status='open'").fetchone()[0]
    prof = cur.execute("SELECT COALESCE(SUM(profit),0) FROM orders WHERE profit IS NOT NULL").fetchone()[0]
    print(f"[{now}] {'DRY-RUN' if dry_run else 'LIVE'} | placed_this_pass={n_taken} "
          f"orders={tot} open={opn} realized_profit={prof:+.2f}", flush=True)


def _seconds_to_next_h4():
    now = datetime.now(tz=timezone.utc)
    nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=(4 - now.hour % 4))
    return max(30.0, (nxt - now).total_seconds() + 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--risk-frac", type=float, default=0.01, help="equity fraction risked/trade")
    ap.add_argument("--live", action="store_true", help="ACTUALLY send orders (default: dry-run)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-ingest", action="store_true")
    args = ap.parse_args()

    strat = pickle.loads(Path(args.model).read_bytes())
    dry_run = not args.live
    mode = "LIVE ORDERS" if args.live else "DRY-RUN (no orders sent)"
    print(f"loaded strategy: tau0={strat.tau0} base_risk={strat.base_risk} | MODE: {mode}")
    if args.live:
        print("!! LIVE MODE: orders will be sent to the connected terminal. "
              "Ensure it is a DEMO account.", flush=True)
    db = Path(args.db)

    if args.loop:
        while True:
            try:
                run_once(strat, db, dry_run=dry_run, no_ingest=args.no_ingest, risk_frac=args.risk_frac)
            except Exception as e:
                print(f"[pass error] {e!r}", flush=True)
            s = _seconds_to_next_h4()
            print(f"sleeping {s/3600:.2f}h until next H4 close...", flush=True)
            time.sleep(s)
    else:
        run_once(strat, db, dry_run=dry_run, no_ingest=args.no_ingest, risk_frac=args.risk_frac)


if __name__ == "__main__":
    main()
