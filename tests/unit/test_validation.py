"""Validation tests: purged K-fold, embargo, CPCV.

ARCHITECTURE.md §7.

The purge/embargo test constructs a synthetic event set with known
overlapping spans and hand-enumerates the exact set of events that
must be removed — not a "fewer than before" smoke check.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from triage.validation import (
    DEFAULT_EMBARGO_BARS,
    DEFAULT_H4_SECONDS,
    backtest_paths,
    cpcv_splits,
    purged_train_idx,
)


# ---------------------------------------------------------------------------
# Test fixture: hand-constructed event spans on a single H4 timeline.
#
# t0 units below are H4-bar offsets from epoch; t1 is the first-touch
# time. We use real timestamps so the embargo Timedelta arithmetic runs
# through the production code path.
#
# Epoch: 2026-01-05 00:00. One H4 bar = 4h.
# Embargo = 48 H4 bars = 48 * 4h = 192h = 8 days.
#
# Test span under examination: bars [40, 60] -> test_start = T+40*4h,
# test_end = T+60*4h. Embargo end = T+(60+48)*4h = T+108*4h.
#
# Each event below is annotated with the hand-derived expectation.
# ---------------------------------------------------------------------------

EPOCH = datetime(2026, 1, 5, 0, 0, 0)
H4 = timedelta(hours=4)


def _bars(n: int) -> datetime:
    return EPOCH + n * H4


def _make_overlapping_events() -> pd.DataFrame:
    """Eight events with hand-analyzable overlap with the test span."""
    rows = [
        # --- Clearly BEFORE the test span (t1 < test_start) -> KEEP ---
        # e0: span [0, 10], ends well before test_start(40). KEEP.
        {"idx": "e0", "t0": _bars(0), "t1": _bars(10)},
        # e1: span [35, 39], t1(39) < test_start(40). KEEP (boundary-strict).
        {"idx": "e1", "t0": _bars(35), "t1": _bars(39)},

        # --- Overlap the test span [40, 60] -> PURGE ---
        # e2: starts before, ends inside. t0=38, t1=45 -> overlaps. PURGE.
        {"idx": "e2", "t0": _bars(38), "t1": _bars(45)},
        # e3: fully inside [42, 55]. PURGE.
        {"idx": "e3", "t0": _bars(42), "t1": _bars(55)},
        # e4: starts inside, ends after. t0=58, t1=80. PURGE.
        {"idx": "e4", "t0": _bars(58), "t1": _bars(80)},
        # e5: exactly straddles. t0=40 == test_start. PURGE (t0 not > end+emb,
        #     and t1(60) is NOT < test_start(40)).
        {"idx": "e5", "t0": _bars(40), "t1": _bars(60)},

        # --- After test span but inside embargo window -> PURGE ---
        # e6: t0 = 80 (test_end=60 < 80 <= test_end+emb=108).
        #     The keep condition is t0 > test_end + emb (strictly greater
        #     than 108). 80 is not > 108 -> PURGE.
        {"idx": "e6", "t0": _bars(80), "t1": _bars(95)},

        # --- After embargo window -> KEEP ---
        # e7: t0 = 200 > 108. KEEP.
        {"idx": "e7", "t0": _bars(200), "t1": _bars(215)},
    ]
    df = pd.DataFrame(rows).set_index("idx")[["t0", "t1"]]
    return df


# Hand-derived expectation for the default 48-bar embargo against the
# test span [bar 40, bar 60]:
#   KEEP:    e0, e1, e7
#   PURGE:   e2, e3, e4, e5, e6
EXPECTED_KEPT = {"e0", "e1", "e7"}
EXPECTED_PURGED = {"e2", "e3", "e4", "e5", "e6"}


def test_purge_and_embargo_removes_exactly_the_hand_enumerated_set():
    """§7: purge overlaps + embargo 48 H4 bars; exact removal set."""
    events = _make_overlapping_events()
    test_start = _bars(40)
    test_end = _bars(60)

    kept = set(purged_train_idx(events, test_start, test_end))
    removed = set(events.index) - kept

    assert kept == EXPECTED_KEPT, (
        f"kept set mismatch: got {sorted(kept)}, "
        f"expected {sorted(EXPECTED_KEPT)}"
    )
    assert removed == EXPECTED_PURGED, (
        f"purged set mismatch: got {sorted(removed)}, "
        f"expected {sorted(EXPECTED_PURGED)}"
    )


def test_embargo_length_is_48_h4_bars():
    """§7: embargo = 48 H4 bars = 2× the 24-bar vertical horizon."""
    assert DEFAULT_EMBARGO_BARS == 48
    assert DEFAULT_H4_SECONDS == 4 * 3600
    # 48 * 4h = 192h = 8 days.
    emb_seconds = DEFAULT_EMBARGO_BARS * DEFAULT_H4_SECONDS
    assert emb_seconds == 48 * 4 * 3600


def test_embargo_boundary_is_strictly_greater():
    """t0 exactly at test_end + emb is purged (keep needs strict >)."""
    events = _make_overlapping_events()
    # Add an event whose t0 sits exactly on the embargo boundary.
    test_start = _bars(40)
    test_end = _bars(60)
    emb_end = _bars(60 + DEFAULT_EMBARGO_BARS)  # bar 108
    edge = pd.DataFrame(
        [{"idx": "edge", "t0": emb_end, "t1": emb_end + H4}]
    ).set_index("index" if False else "idx")[["t0", "t1"]]
    events_with_edge = pd.concat([events, edge])

    kept = set(purged_train_idx(events_with_edge, test_start, test_end))
    # edge.t0 == emb_end exactly -> NOT strictly greater -> purged.
    assert "edge" not in kept, (
        "t0 == test_end+embargo must be purged (keep requires strict >). "
        "ARCHITECTURE.md §7 sketch: events.t0 > test_end + emb."
    )


def test_zero_embargo_only_purges_overlaps():
    """With embargo=0, only overlap-spanning events are removed.

    Sanity check that the embargo term is doing the post-test work and
    the overlap purge is independent of it. e6 (t0=80, after test_end=60)
    should now be KEPT because there's no embargo and its t0 > test_end.
    """
    events = _make_overlapping_events()
    test_start = _bars(40)
    test_end = _bars(60)

    kept = set(purged_train_idx(events, test_start, test_end, embargo_bars=0))
    # e6 (t0=80 > test_end=60) now kept; e2-e5 still purged for overlap.
    assert kept == {"e0", "e1", "e6", "e7"}


# ---------------------------------------------------------------------------
# CPCV: N=6, k=2 -> exactly 15 splits, 5 backtest paths.
# ---------------------------------------------------------------------------


def _make_cpcv_events(n: int = 60) -> pd.DataFrame:
    """n events with monotonically increasing, lightly overlapping spans."""
    rows = []
    for i in range(n):
        t0 = EPOCH + i * H4
        # t1 a few bars after t0 so spans overlap a little.
        t1 = t0 + (3 + (i % 4)) * H4
        rows.append({"idx": f"e{i:03d}", "t0": t0, "t1": t1})
    return pd.DataFrame(rows).set_index("idx")[["t0", "t1"]]


def test_cpcv_default_produces_15_splits_and_5_paths():
    """§7: N=6 groups, k=2 test groups -> 15 splits, 5 backtest paths."""
    from math import comb

    events = _make_cpcv_events(60)
    splits = cpcv_splits(events)  # defaults: n_groups=6, k_test=2

    # 15 splits = C(6,2).
    assert len(splits) == comb(6, 2) == 15

    # Every split's test_groups is a distinct 2-subset of {0..5}.
    test_group_sets = [set(s.test_groups) for s in splits]
    assert len({tuple(sorted(g)) for g in test_group_sets}) == 15
    for g in test_group_sets:
        assert g <= {0, 1, 2, 3, 4, 5} and len(g) == 2

    # 5 backtest paths, each a disjoint partition into 3 pairs.
    paths = backtest_paths(n_groups=6, k_test=2)
    assert len(paths) == 5, f"expected 5 backtest paths, got {len(paths)}"
    for path in paths:
        flat = [g for pair in path for g in pair]
        assert sorted(flat) == [0, 1, 2, 3, 4, 5], (
            f"path {path} is not a partition of all 6 groups"
        )


def test_cpcv_test_groups_are_disjoint_across_each_path():
    """Each backtest path partitions the 6 groups into 3 disjoint pairs."""
    paths = backtest_paths(n_groups=6, k_test=2)
    for path in paths:
        groups_seen: list[int] = []
        for pair in path:
            for g in pair:
                assert g not in groups_seen, (
                    f"group {g} reused within a single backtest path {path}"
                )
                groups_seen.append(g)


def test_cpcv_purge_leaves_only_events_outside_test_span():
    """Each split's train set excludes events overlapping its test span.

    Build a small but well-understood event set and check the purge
    takes effect: events whose [t0,t1] falls inside the held-out test
    groups must not appear in train_idx.
    """
    events = _make_cpcv_events(60)
    splits = cpcv_splits(events, n_groups=6, k_test=2, embargo_bars=0)

    # Pick split #0 = test groups (0, 1). Those cover the earliest 1/3
    # of the timeline. With embargo_bars=0, every event whose span
    # touches groups 0 or 1 is purged; only events fully after are kept.
    sp = splits[0]
    assert sp.test_groups == (0, 1)

    test_events = events.loc[sp.test_idx]
    test_start = test_events["t0"].min()
    test_end = test_events["t1"].max()
    # Every train event must satisfy t1 < test_start (early) OR
    # t0 > test_end (late). Since groups 0,1 are the earliest, there are
    # no "early" events -> all train events must have t0 > test_end.
    train_events = events.loc[sp.train_idx]
    assert (train_events["t0"] > test_end).all(), (
        "train events overlapping the test span leaked through the purge"
    )
    # And no test id appears in train.
    assert set(sp.test_idx).isdisjoint(set(sp.train_idx))


def test_cpcv_rejects_too_few_events():
    events = _make_cpcv_events(3)  # < n_groups=6
    with pytest.raises(ValueError):
        cpcv_splits(events)


def test_cpcv_rejects_bad_combinatorics():
    events = _make_cpcv_events(60)
    with pytest.raises(ValueError):
        cpcv_splits(events, n_groups=6, k_test=6)
    with pytest.raises(ValueError):
        cpcv_splits(events, n_groups=1, k_test=1)
