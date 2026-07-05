"""Phase 0 trivial stub rule (retained for synthetic labeler tests).

Per the original Phase 0 build prompt: "a single trivial deterministic
stub used only to exercise the labeler in tests." Not a real alpha
source. The real Phase 1 rule lives in :mod:`triage.rules.donchian`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..data import Bars, Side

__all__ = ["Signal", "RULE_VERSION", "trivial_long_rule"]

#: Phase 0 stub version tag.
RULE_VERSION: str = "trivial_long_v0"


@dataclass(frozen=True)
class Signal:
    """A candidate signal from a rule (Phase 0 form)."""

    t0: datetime
    side: Side
    entry_type: str = "next_bar_open"


def trivial_long_rule(bars: Bars) -> Optional[Signal]:
    """Deterministic stub: fire long on bar index 0, else None."""
    if not bars:
        return None
    return Signal(t0=bars[0].close_time, side=1)
