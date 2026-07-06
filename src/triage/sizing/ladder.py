"""Discrete position-sizing ladder (§8).

ARCHITECTURE.md §8 division of labor: "regime information belongs in the
features (the model turns it into p); the sizer consumes calibrated p
and risk budget only, plus hard risk overlays."

EV (R units, vertical conservatively valued at 0):

    EV(p) = 2.5*p - 1 - c       p* = (1+c)/2.5       tau0 = p* + 0.05

Discrete ladder:

    p < tau0          -> no trade       (size = 0)
    tau0 <= p < tau1  -> 0.5 x base_risk
    p >= tau1         -> 1.0 x base_risk

tau0/tau1 selected on training folds only (§8). Hard overlays (risk,
not alpha): ATR percentile > 95 halves size; aggregate open risk <=
3 x base_risk; per-symbol single open position in live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "DEFAULT_COST_R",
    "DEFAULT_BASE_RISK",
    "ev_r",
    "breakeven_p",
    "default_tau0",
    "SizingLadder",
    "select_tau_on_train",
    "size_from_probability",
]


#: Conservative cost for majors (§7: "~0.03-0.05R against a 1xATR H4 stop").
DEFAULT_COST_R: float = 0.04
#: Default risk per trade (§8: "0.5-1.0% equity per trade").
DEFAULT_BASE_RISK: float = 0.01


def ev_r(p: float, c: float = DEFAULT_COST_R) -> float:
    """Expected value in R units: ``2.5*p - 1 - c`` (§8)."""
    return 2.5 * p - 1.0 - c


def breakeven_p(c: float = DEFAULT_COST_R) -> float:
    """Break-even win probability ``p* = (1+c)/2.5`` (§8)."""
    return (1.0 + c) / 2.5


def default_tau0(c: float = DEFAULT_COST_R) -> float:
    """Entry threshold with margin: ``tau0 = p* + 0.05`` (§8)."""
    return breakeven_p(c) + 0.05


@dataclass(frozen=True)
class SizingLadder:
    """Discrete sizing ladder (§8). Frozen once tau1_margin is chosen."""

    tau1_margin: float  # tau1 = tau0 + tau1_margin
    base_risk: float = DEFAULT_BASE_RISK

    def size(self, p: float, *, cost_r: float = DEFAULT_COST_R, atr_percentile: Optional[float] = None) -> float:
        """Return position size as a fraction of equity risk.

        ``atr_percentile`` (0..1) triggers the ATR>95 overlay (halve size).
        """
        tau0 = default_tau0(cost_r)
        tau1 = tau0 + self.tau1_margin
        
        if p < tau0:
            return 0.0
        size = self.base_risk * (0.5 if p < tau1 else 1.0)
        if atr_percentile is not None and atr_percentile > 0.95:
            size *= 0.5  # §8 hard overlay: ATR pct > 95 -> halve
        return size


def get_cost_r(symbol: str, t0: object = None) -> float:
    """Return the conservative per-symbol cost in R units (§7 cost model).

    Single source of truth: delegates to :func:`triage.gates.cost_for_symbol`
    so the sizer and the gates never disagree on cost. (An earlier draft kept
    a second, divergent table here — XAU/US30/BTC values that contradicted
    ``gates.COST_BY_SYMBOL`` — which is the exact self-deception §7 warns
    about when it says "gating on pre-cost numbers is self-deception".)

    The ``t0`` argument is retained for call-site back-compat but ignored:
    §7's BTC weekend-gap premium is baked into the flat ``COST_BY_SYMBOL``
    value, not applied per-timestamp here.
    """
    from ..gates import cost_for_symbol

    return cost_for_symbol(symbol)


def select_tau_on_train(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    *,
    costs_r: np.ndarray | float = DEFAULT_COST_R,
    tau1_margins: Optional[list[float]] = None,
) -> SizingLadder:
    """Select tau1 margin on training-fold predictions only (§8).

    tau0 is the EV break-even plus margin (formula-dixed per cost). tau1 is
    chosen by scanning candidates and picking the one that maximizes
    mean per-trade EV on the training set above tau0.

    Args:
        train_probs: Calibrated P(win) predictions on the training fold.
        train_labels: True training labels (1/0).
        costs_r: Cost in R units (scalar or array matching train_probs).
        tau1_margins: Grid to scan for tau1 - tau0.

    Returns:
        A frozen :class:`SizingLadder`.
    """
    if isinstance(costs_r, float):
        costs_r = np.full_like(train_probs, costs_r)
        
    tau0s = np.array([default_tau0(c) for c in costs_r])
    
    if tau1_margins is None:
        tau1_margins = [0.05, 0.10, 0.15, 0.20]
        
    best_margin = tau1_margins[0]
    best_ev = -1e9
    
    for margin in tau1_margins:
        tau1s = tau0s + margin
        # EV at the two rungs weighted by how many trades land in each.
        mask_lo = (train_probs >= tau0s) & (train_probs < tau1s)
        mask_hi = train_probs >= tau1s
        
        # Calculate EV dynamically based on individual costs
        evs_lo = (2.5 * train_probs[mask_lo] - 1.0 - costs_r[mask_lo]) if mask_lo.any() else np.array([0.0])
        evs_hi = (2.5 * train_probs[mask_hi] - 1.0 - costs_r[mask_hi]) if mask_hi.any() else np.array([0.0])
        
        ev_lo = evs_lo.mean() if mask_lo.any() else 0.0
        ev_hi = evs_hi.mean() if mask_hi.any() else 0.0
        
        # Weight by fraction of trades (assume 0.5x and 1.0x base).
        w_lo = 0.5 * mask_lo.sum()
        w_hi = 1.0 * mask_hi.sum()
        tot = w_lo + w_hi
        combined = ((ev_lo * w_lo) + (ev_hi * w_hi)) / tot if tot > 0 else -1e9
        if combined > best_ev:
            best_ev = combined
            best_margin = margin
            
    return SizingLadder(tau1_margin=best_margin, base_risk=DEFAULT_BASE_RISK)


def size_from_probability(
    p: float,
    ladder: SizingLadder,
    *,
    cost_r: float = DEFAULT_COST_R,
    atr_percentile: Optional[float] = None,
) -> float:
    """Apply the sizing ladder to a single calibrated probability."""
    return ladder.size(p, cost_r=cost_r, atr_percentile=atr_percentile)
