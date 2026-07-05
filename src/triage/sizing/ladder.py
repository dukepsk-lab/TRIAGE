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
    """Discrete sizing ladder (§8). Frozen once tau0/tau1 are chosen."""

    tau0: float
    tau1: float
    base_risk: float = DEFAULT_BASE_RISK
    cost_r: float = DEFAULT_COST_R

    def size(self, p: float, *, atr_percentile: Optional[float] = None) -> float:
        """Return position size as a fraction of equity risk.

        ``atr_percentile`` (0..1) triggers the ATR>95 overlay (halve size).
        """
        if p < self.tau0:
            return 0.0
        size = self.base_risk * (0.5 if p < self.tau1 else 1.0)
        if atr_percentile is not None and atr_percentile > 0.95:
            size *= 0.5  # §8 hard overlay: ATR pct > 95 -> halve
        return size


def select_tau_on_train(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    *,
    cost_r: float = DEFAULT_COST_R,
    tau1_candidates: Optional[list[float]] = None,
) -> SizingLadder:
    """Select tau0/tau1 on training-fold predictions only (§8).

    tau0 is the EV break-even plus margin (formula-dixed). tau1 is
    chosen by scanning candidates and picking the one that maximizes
    mean per-trade EV on the training set above tau0.

    Args:
        train_probs: Calibrated P(win) predictions on the training fold.
        train_labels: True training labels (1/0).
        cost_r: Cost in R units.
        tau1_candidates: Grid to scan for tau1. Defaults to a coarse grid.

    Returns:
        A frozen :class:`SizingLadder`.
    """
    tau0 = default_tau0(cost_r)
    if tau1_candidates is None:
        tau1_candidates = [0.50, 0.55, 0.60, 0.65, 0.70]
    best_tau1 = max(tau1_candidates[0], tau0 + 0.01)
    best_ev = -1e9
    for t1 in tau1_candidates:
        if t1 <= tau0:
            continue
        # EV at the two rungs weighted by how many trades land in each.
        mask_lo = (train_probs >= tau0) & (train_probs < t1)
        mask_hi = train_probs >= t1
        ev_lo = ev_r(train_probs[mask_lo].mean()) if mask_lo.any() else 0.0
        ev_hi = ev_r(train_probs[mask_hi].mean()) if mask_hi.any() else 0.0
        # Weight by fraction of trades (assume 0.5x and 1.0x base).
        w_lo = 0.5 * mask_lo.sum()
        w_hi = 1.0 * mask_hi.sum()
        tot = w_lo + w_hi
        combined = ((ev_lo * w_lo) + (ev_hi * w_hi)) / tot if tot > 0 else -1e9
        if combined > best_ev:
            best_ev = combined
            best_tau1 = t1
    return SizingLadder(tau0=tau0, tau1=best_tau1, base_risk=DEFAULT_BASE_RISK, cost_r=cost_r)


def size_from_probability(
    p: float,
    ladder: SizingLadder,
    *,
    atr_percentile: Optional[float] = None,
) -> float:
    """Apply the sizing ladder to a single calibrated probability."""
    return ladder.size(p, atr_percentile=atr_percentile)
