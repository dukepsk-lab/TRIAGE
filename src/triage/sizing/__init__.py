"""Position sizing — discrete ladder (§8).

ARCHITECTURE.md §8 division of labor: the sizer consumes calibrated
P(win) and risk budget only, plus hard risk overlays. Regime information
belongs in features, not the sizer.

Phase 1 ships the discrete ladder + EV/threshold machinery. Continuous
Kelly sizing is a post-Phase-4 revisit (§8).
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_COST_R",
    "DEFAULT_BASE_RISK",
    "SizingLadder",
    "ev_r",
    "breakeven_p",
    "breakeven_threshold",
    "default_tau0",
    "select_tau_on_train",
    "size_from_probability",
]

from .ladder import (  # noqa: F401
    DEFAULT_BASE_RISK,
    DEFAULT_COST_R,
    SizingLadder,
    breakeven_p,
    default_tau0,
    ev_r,
    select_tau_on_train,
    size_from_probability,
)


def breakeven_threshold(c: float = DEFAULT_COST_R) -> float:
    """Back-compat alias for :func:`breakeven_p` (§8 ``p*``)."""
    return breakeven_p(c)

