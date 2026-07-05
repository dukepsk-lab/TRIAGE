"""Rule engine — Phase 1 ships the Donchian-20 breakout rule (§3).

Phase 0 had only the ``trivial_long_rule`` stub. Phase 1 adds the real
``donchian`` rule (signal logic) plus its two execution paths
(``donchian_backfill``: vectorized + incremental), which the §3 parity
test keeps identical.
"""

from __future__ import annotations

from .donchian import (  # noqa: F401
    DEFAULT_CONFIG,
    CHANNEL_BARS,
    D1_EMA_PERIOD,
    DonchianConfig,
    DonchianSignal,
    RULE_VERSION,
    compute_donchian_channel,
    compute_d1_ema_slope,
    session_mask,
    signal_at_bar,
)
from .donchian_backfill import (  # noqa: F401
    IncrementalDonchian,
    align_d1_slope_to_h4,
    backfill_signals,
    d1_slope_at_h4_close,
    replay_history,
)
from . import pullback as pullback_rule  # noqa: F401  Phase 3 second family
from .pullback import (  # noqa: F401
    PULLBACK_RULE_VERSION,
    PullbackConfig,
    PullbackSignal,
)

# Phase 0 stub retained for the synthetic labeler tests.
from .stub import RULE_VERSION as TRIVIAL_RULE_VERSION  # noqa: F401
from .stub import Signal, trivial_long_rule  # noqa: F401

__all__ = [
    "DonchianConfig",
    "DonchianSignal",
    "RULE_VERSION",
    "backfill_signals",
    "replay_history",
    "IncrementalDonchian",
    "Signal",
    "trivial_long_rule",
]
