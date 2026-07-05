"""Feature-timestamp audit tests (Phase 0 correctness battery).

ARCHITECTURE.md §4/§7 leakage checklist: "features from closed bars <=
t0". :func:`triage.features.no_lookahead` is the enforcement mechanism.
This test exercises it with both a clean snapshot and a deliberately
leaky one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from triage.features import FEATURE_VERSION, LookaheadError, no_lookahead


T0 = datetime(2026, 1, 5, 8, 0, 0)


def test_feature_version_is_real_phase1():
    """Sanity: feature service is the real Phase 1 Tier 1+2+3 service."""
    assert FEATURE_VERSION.startswith("phase1_"), FEATURE_VERSION


def test_clean_snapshot_passes_audit():
    """All source timestamps <= t0: audit passes, returns the max ts."""
    snapshot = {
        "rsi": 0.42,
        "atr_pct": 0.012,
        "max_source_close_time": T0 - timedelta(hours=4),
        "rsi_source_close_time": T0 - timedelta(hours=8),
    }
    max_ts = no_lookahead(snapshot, T0)
    assert max_ts == T0 - timedelta(hours=4)


def test_snapshot_with_exact_t0_passes():
    """Boundary: source timestamp == t0 is allowed (closed bar at t0)."""
    snapshot = {"feat": 1.0, "max_source_close_time": T0}
    assert no_lookahead(snapshot, T0) == T0


def test_leaky_snapshot_is_caught():
    """Deliberately leaky fixture: source ts > t0 MUST raise.

    This is the test the prompt calls out: a deliberately leaky fixture
    that the checker must catch. If this passes silently, the no-look-
    ahead invariant is unenforced — which defeats the entire harness.
    """
    leaky = {
        "feat": 1.0,
        # Source bar closes AFTER t0 — the still-forming bar leaked in.
        "max_source_close_time": T0 + timedelta(minutes=1),
    }
    with pytest.raises(LookaheadError, match="look-ahead detected"):
        no_lookahead(leaky, T0)


def test_per_component_timestamp_leak_is_caught():
    """A leak in any single component timestamp trips the audit.

    Even when ``max_source_close_time`` is fine, a per-component field
    listed in ``timestamp_keys`` that exceeds t0 must be caught — that's
    the whole point of auditing each component independently.
    """
    leaky = {
        "feat_a": 1.0,
        "feat_b": 2.0,
        "max_source_close_time": T0 - timedelta(hours=4),
        "feat_b_source_close_time": T0 + timedelta(hours=1),  # leak
    }
    with pytest.raises(LookaheadError):
        no_lookahead(leaky, T0)


def test_unauditable_snapshot_fails_closed():
    """No timestamp fields at all -> fail closed (can't verify invariant)."""
    unauditable = {"rsi": 0.4, "atr_pct": 0.01}  # no ts keys
    with pytest.raises(LookaheadError, match="not auditable"):
        no_lookahead(unauditable, T0)


def test_lookahead_error_is_assertion_error_subclass():
    """Failures surface naturally under ``assert`` / pytest."""
    assert issubclass(LookaheadError, AssertionError)
