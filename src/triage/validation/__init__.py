"""Cross-validation machinery: purged K-fold, embargo, CPCV.

Implements ARCHITECTURE.md §7 ("Validation framework").

Two leakage controls are mandatory here (§7 "Leakage checklist"): (1)
every training fold is *purged* of events whose information span
``[t0, t1]`` overlaps the test span, plus an *embargo* of events that
start within ``embargo_bars`` after test end (covers label overlap plus
feature serial correlation); (2) CPCV (Combinatorial Purged CV) gives
the synthetic-backtest paths the PBO gate is computed over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

__all__ = [
    "DEFAULT_EMBARGO_BARS",
    "DEFAULT_H4_SECONDS",
    "purged_train_idx",
    "cpcv_splits",
    "CPCVSplit",
]


#: Default embargo = 48 H4 bars = 2× the 24-bar vertical horizon
#: (ARCHITECTURE.md §7). Covers label overlap + feature serial
#: correlation with margin.
DEFAULT_EMBARGO_BARS: int = 48

#: H4 bar length in seconds (4 hours).
DEFAULT_H4_SECONDS: int = 4 * 3600


def purged_train_idx(
    events: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
    h4_seconds: int = DEFAULT_H4_SECONDS,
) -> pd.Index:
    """Return the training-event index after purging + embargo.

    Implements ``purged_train_idx`` per ARCHITECTURE.md §7. An event's
    information lives over ``[t0, t1]``; for each test fold we drop
    every training event whose span overlaps the test span, then embargo
    training events starting within ``embargo_bars`` after the test end:

    .. code-block:: python

        emb = embargo_bars * H4
        ok  = (events.t1 < test_start) | (events.t0 > test_end + emb)
        return events.index[ok]

    Args:
        events: DataFrame indexed by event id, with ``t0`` and ``t1``
            columns (timestamps). ``t1`` may be ``NaT`` for unresolved
            events; such events span up to ``t_vert`` conceptually but
            for purging we treat a missing ``t1`` as ``+inf`` *only* in
            the sense that we cannot prove it ends before the test span
            — i.e. an unresolved event is conservatively *kept out* of
            training unless its ``t0`` clears the embargo after test
            end. (This matches §7's intent: when in doubt, drop.)
        test_start: Test fold start (inclusive boundary for the test
            span). An event with ``t1 == test_start`` is *not* kept —
            its information reaches the boundary.
        test_end: Test fold end (inclusive).
        embargo_bars: Number of H4 bars to embargo after ``test_end``.
        h4_seconds: Seconds per H4 bar, for converting the embargo.

    Returns:
        The subset of ``events.index`` that is safe to train on for
        this test fold.
    """
    emb = pd.Timedelta(seconds=embargo_bars * h4_seconds)

    # An event overlaps the test span if its [t0, t1] intersects
    # [test_start, test_end]. The §7 sketch writes the *keep* condition
    # directly: keep iff t1 < test_start OR t0 > test_end + emb.
    t0 = events["t0"]
    t1 = events["t1"]

    t1_before = t1 < test_start
    # t1 == NaT means "unresolved / unknown end". Conservatively treat
    # such events as NOT satisfying "t1 < test_start" — they cannot be
    # proven to clear the test span on the early side.
    t1_before = t1_before.fillna(False)

    t0_after_embargo = t0 > (test_end + emb)

    ok = t1_before | t0_after_embargo
    return events.index[ok.to_numpy()]


# ---------------------------------------------------------------------------
# CPCV — Combinatorial Purged Cross-Validation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPCVSplit:
    """One CPCV split: the test-group indices and the member event ids.

    ARCHITECTURE.md §7: N=6 groups, k=2 test groups -> 15 splits, 5
    synthetic backtest paths. Each split designates ``k`` of the ``N``
    time-ordered groups as the test set; the remaining groups form the
    training set *after* purging events whose spans reach into the test
    groups and applying the embargo.
    """

    split_id: int
    test_groups: tuple[int, ...]      # group indices (0-based) in test
    test_idx: pd.Index                # event ids in the test set
    train_idx: pd.Index               # purged+embargoed training ids


def cpcv_splits(
    events: pd.DataFrame,
    n_groups: int = 6,
    k_test: int = 2,
    *,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
    h4_seconds: int = DEFAULT_H4_SECONDS,
    time_col: str = "t0",
) -> list[CPCVSplit]:
    """Generate CPCV splits over a time-ordered event set.

    Implements CPCV per ARCHITECTURE.md §7 (N=6, k=2 -> 15 splits). The
    default ``n_groups=6, k_test=2`` reproduces the doc's spec exactly;
    both are parameters only so the unit tests can exercise smaller
    topologies (the production path always uses the defaults).

    Events are sorted by ``time_col`` (default ``t0``) and partitioned
    into ``n_groups`` contiguous groups. For each combination of
    ``k_test`` groups, the events in those groups form the test set;
    training events are purged + embargoed against the *union* span of
    the test groups via :func:`purged_train_idx` (treating the test
    groups' min start and max end as the test span).

    Args:
        events: DataFrame with ``t0``, ``t1`` columns and a meaningful
            index (event ids). ``t1`` may be ``NaT``.
        n_groups: Number of contiguous time groups (default 6).
        k_test: Number of groups held out as test per split (default 2).
        embargo_bars: Forward embargo after the test span (default 48).
        h4_seconds: Seconds per H4 bar (default 14400).
        time_col: Column to sort/group by (default ``t0``).

    Returns:
        One :class:`CPCVSplit` per combination, in deterministic order.
        For the defaults this is exactly 15 splits.

    Raises:
        ValueError: If there are fewer events than groups, or if the
            combination parameters are degenerate.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if not (1 <= k_test < n_groups):
        raise ValueError(
            f"k_test must satisfy 1 <= k_test < n_groups; got {k_test}/{n_groups}"
        )
    if len(events) < n_groups:
        raise ValueError(
            f"need at least n_groups={n_groups} events to form groups, "
            f"got {len(events)}"
        )

    # Sort by time and assign contiguous groups. Use rank positions so
    # groups are as equal in size as possible.
    order = events.sort_values(time_col).index
    n = len(order)
    # Boundaries: group g gets positions [g*n/N, (g+1)*n/N).
    group_of: dict[object, int] = {}
    groups: list[list[object]] = [[] for _ in range(n_groups)]
    for pos, idx in enumerate(order):
        g = min(n_groups - 1, (pos * n_groups) // n)
        group_of[idx] = g
        groups[g].append(idx)

    from itertools import combinations

    splits: list[CPCVSplit] = []
    for split_id, test_groups in enumerate(combinations(range(n_groups), k_test)):
        test_ids = [idx for g in test_groups for idx in groups[g]]
        test_idx = pd.Index(test_ids)

        # Union span of the test groups -> single [test_start, test_end]
        # for the purge. Using the actual t0/t1 of the held-out events
        # is what makes the purge correct under arbitrary groupings.
        test_events = events.loc[test_idx]
        test_start = test_events["t0"].min()
        # t1 may be NaT; for the test-span *end* use the max of the
        # non-NaT t1 values, falling back to t0 if all are NaT.
        t1_max = test_events["t1"].max()
        test_end = t1_max if pd.notna(t1_max) else test_events["t0"].max()

        train_idx = purged_train_idx(
            events,
            test_start,
            test_end,
            embargo_bars=embargo_bars,
            h4_seconds=h4_seconds,
        )

        splits.append(
            CPCVSplit(
                split_id=split_id,
                test_groups=tuple(test_groups),
                test_idx=test_idx,
                train_idx=train_idx,
            )
        )

    return splits


def backtest_paths(n_groups: int = 6, k_test: int = 2) -> list[list[tuple[int, ...]]]:
    """Return the CPCV backtest paths (the φ synthetic paths).

    Implements the path construction referenced by ARCHITECTURE.md §7
    ("N = 6 groups, k = 2 test groups -> 15 splits, 5 synthetic backtest
    paths"). For the canonical ``k_test = 2`` this is a **1-factorization
    of the complete graph** K_N: φ = N − 1 edge-disjoint perfect
    matchings that together cover every one of the C(N,2) pairs exactly
    once (φ · N/2 = (N−1)·N/2 = C(N,2)). Each matching is one backtest
    path — a set of N/2 disjoint test-pairs that partitions all groups.

    Concretely, for N=6, k=2:

    * 15 splits = all C(6,2) test-pairs;
    * 5 paths = the 5 matchings of a 1-factorization of K6;
    * each path = 3 disjoint pairs covering groups {0..5};
    * every split belongs to exactly one path.

    The construction is the textbook round-robin 1-factorization
    (López de Prado, *AFML*, CPCV), which is deterministic and provably
    covers every edge once.

    A path tells you *which splits to chain* into one full backtest; the
    PBO gate (§7) is computed across these paths. This helper does not
    touch event data.

    Args:
        n_groups: N (must be even). Default 6.
        k_test: k. Only ``k_test = 2`` is supported (the architecture's
            spec). General k requires a resolvable BIBD construction
            that is out of scope for Phase 0.

    Returns:
        ``φ = N − 1`` paths, each a list of ``N/2`` sorted 2-tuples.
    """
    if k_test != 2:
        raise NotImplementedError(
            f"backtest_paths supports k_test=2 only (got {k_test}). "
            "General-k CPCV paths need a resolvable-design construction "
            "and are out of scope for Phase 0; the architecture spec is "
            "exclusively N=6, k=2 (ARCHITECTURE.md §7)."
        )
    if n_groups % 2 != 0:
        raise ValueError(
            f"n_groups must be even for the k=2 1-factorization, got {n_groups}"
        )

    n = n_groups
    matchings: list[list[tuple[int, ...]]] = []
    # Fix the last group; rotate the remaining n-1 around it.
    last = n - 1
    for r in range(n - 1):
        pairs: list[tuple[int, ...]] = [(r, last)]  # last vs r
        for i in range(1, n // 2):
            a = (r + i) % (n - 1)
            b = (r - i) % (n - 1)
            pairs.append((min(a, b), max(a, b)))
        # Canonical key order within the path for stable comparison.
        pairs.sort()
        matchings.append(pairs)

    # Invariant: the matchings partition all C(n,2) pairs.
    from itertools import combinations

    all_pairs = set(combinations(range(n), 2))
    produced = {p for m in matchings for p in m}
    assert produced == all_pairs and len(produced) == len(all_pairs), (
        "1-factorization failed to cover every pair exactly once"
    )
    return matchings
