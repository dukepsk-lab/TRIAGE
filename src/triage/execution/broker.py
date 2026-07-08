"""MT5 execution adapter — Phase 4 (§9). Places demo orders with barrier prices.

Two layers:
  * :func:`compute_lot` — pure, testable position sizing (risk budget -> lots).
  * thin MT5 wrappers (:func:`place_market_order`, :func:`close_position`,
    :func:`open_positions`, :func:`account_equity`) that call ``MetaTrader5``.

Every mutating call honours ``dry_run`` (the default at the call sites): it logs
the intended request and returns a synthetic result WITHOUT touching the broker.
Real sends require an explicit opt-in from the orchestrator (``--live``).

Fixed sizing only (rung 2): the caller passes a risk budget in equity fraction;
this module converts it to lots via the symbol's tick economics. Probability-
laddered sizing (rung 3) is not used until G5 clears (§8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: Orders tagged with this magic number are "ours" — used to filter positions.
TRIAGE_MAGIC: int = 20260708


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    ticket: Optional[int]
    lot: float
    price: float
    sl: float
    tp: float
    comment: str
    dry_run: bool


def compute_lot(
    *,
    entry_px: float,
    sl_px: float,
    risk_amount: float,
    tick_value: float,
    tick_size: float,
    vol_min: float,
    vol_max: float,
    vol_step: float,
) -> float:
    """Lots such that a stop-out loses about ``risk_amount`` (account ccy).

    ``loss_per_lot = (|entry - sl| / tick_size) * tick_value``; lot =
    risk_amount / loss_per_lot, floored to ``vol_step`` and clamped to
    ``[vol_min, vol_max]``. Returns ``0.0`` if inputs are degenerate (caller
    must treat 0 as "do not trade").
    """
    sl_dist = abs(entry_px - sl_px)
    if sl_dist <= 0 or tick_size <= 0 or tick_value <= 0 or vol_step <= 0:
        return 0.0
    loss_per_lot = (sl_dist / tick_size) * tick_value
    if loss_per_lot <= 0:
        return 0.0
    raw = risk_amount / loss_per_lot
    # +1e-9 absorbs float-representation error in raw/vol_step (e.g. 999.9999
    # for an exact 1000) without ever nudging a genuine value up a whole step.
    lot = math.floor(raw / vol_step + 1e-9) * vol_step
    if lot < vol_min:
        return 0.0  # cannot risk this little at min lot -> skip, don't over-risk
    lot = min(lot, vol_max)
    # Normalize float noise to the step's decimals.
    decimals = max(0, int(round(-math.log10(vol_step)))) if vol_step < 1 else 0
    return round(lot, decimals)


# ---------------------------------------------------------------------------
# MT5 wrappers (import lazily so the module loads without the terminal).
# ---------------------------------------------------------------------------


def account_equity() -> float:
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    ai = mt5.account_info()
    return float(ai.equity) if ai else 0.0


def open_positions(magic: int = TRIAGE_MAGIC) -> list:
    """Our open positions (filtered by magic)."""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    pos = mt5.positions_get()
    return [p for p in (pos or []) if getattr(p, "magic", None) == magic]


def _broker_symbol(symbol: str) -> str:
    from ..data.mt5_ingest import BROKER_SYMBOL_MAP

    return BROKER_SYMBOL_MAP.get(symbol, symbol)


def place_market_order(
    *, symbol: str, side: int, lot: float, sl: float, tp: float,
    magic: int = TRIAGE_MAGIC, comment: str = "triage_donchian",
    deviation: int = 20, dry_run: bool = True,
) -> OrderResult:
    """Send a market order with SL/TP attached (or log it, if ``dry_run``).

    ``side``: +1 buy, -1 sell. Price is the current ask (buy) / bid (sell).
    """
    import MetaTrader5 as mt5

    bsym = _broker_symbol(symbol)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    mt5.symbol_select(bsym, True)
    tick = mt5.symbol_info_tick(bsym)
    info = mt5.symbol_info(bsym)
    if tick is None or info is None:
        return OrderResult(False, None, lot, 0.0, sl, tp, "no_tick", dry_run)
    price = float(tick.ask if side == 1 else tick.bid)
    if lot <= 0:
        return OrderResult(False, None, lot, price, sl, tp, "lot<=0", dry_run)

    if dry_run:
        return OrderResult(True, None, lot, price, sl, tp, "DRY_RUN", True)

    # Choose a filling mode the symbol supports.
    filling = getattr(info, "filling_mode", 0)
    fill = mt5.ORDER_FILLING_IOC
    if filling and not (filling & 1):  # 1 == FOK-capable bit varies by broker
        fill = mt5.ORDER_FILLING_FOK if (filling & 2) else mt5.ORDER_FILLING_RETURN
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": bsym, "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL,
        "price": price, "sl": float(sl), "tp": float(tp), "deviation": deviation,
        "magic": magic, "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill,
    }
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    ticket = int(res.order) if ok and getattr(res, "order", 0) else None
    return OrderResult(ok, ticket, lot, price,
                       sl, tp, f"retcode={getattr(res,'retcode',None)}", False)


def close_position(position, *, deviation: int = 20, dry_run: bool = True) -> OrderResult:
    """Close an open position at market (vertical-barrier / manual exit)."""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    bsym = position.symbol
    tick = mt5.symbol_info_tick(bsym)
    is_long = position.type == mt5.POSITION_TYPE_BUY
    price = float(tick.bid if is_long else tick.ask)
    if dry_run:
        return OrderResult(True, position.ticket, position.volume, price, 0.0, 0.0, "DRY_RUN_CLOSE", True)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": bsym, "volume": float(position.volume),
        "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
        "position": position.ticket, "price": price, "deviation": deviation,
        "magic": getattr(position, "magic", TRIAGE_MAGIC), "comment": "triage_vert_close",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    return OrderResult(ok, position.ticket, position.volume, price, 0.0, 0.0,
                       f"retcode={getattr(res,'retcode',None)}", False)
