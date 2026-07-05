"""Feature service + no-lookahead audit utility.

Implements ARCHITECTURE.md §5 ("Point-in-time snapshotting is the
anti-leakage architecture") and the "features from closed bars <= t0"
line of the §7 leakage checklist.

- :mod:`triage.features.audit` — the no-lookahead enforcement function
  (shipped in Phase 0, wired into every Phase 1 snapshot).
- :mod:`triage.features.service` — the real Tier 1+2+3 feature service
  (~16 columns) added in Phase 1.
"""

from __future__ import annotations

__all__ = [
    "FEATURE_VERSION",
    "FEATURE_COLUMNS",
    "LookaheadError",
    "no_lookahead",
    "compute_snapshot",
    "compute_features_frame",
    "FeatureContext",
    "feature_hash",
]

from .audit import LookaheadError, no_lookahead  # noqa: F401
from .service import (  # noqa: E402,F401
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    FeatureContext,
    compute_features_frame,
    compute_snapshot,
    feature_hash,
)
