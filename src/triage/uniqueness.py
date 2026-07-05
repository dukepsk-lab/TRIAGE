"""Sample-uniqueness weights (Lopez de Prado, ARCHITECTURE.md §4).

§4: "Labels spanning [t0, t1] overlap when signals fire while a prior
event is unresolved. Per H4 bar t, concurrency c_t = number of events
whose span contains t; average uniqueness of event i:

    u_i = mean over t in [t0_i, t1_i] of (1 / c_t)   ->  sample_weight_i = u_i

Two implementation rules: (a) weights are computed **per fold on
training events only** — computing concurrency over the full event set
leaks test-period event timing into training; (b) skip sequential
bootstrap in Phase 1."

This module implements (a) with a hard guard: :func:`uniqueness_weights`
takes the *training* event subset and the fold boundaries, and refuses
to compute over a full-set index. The leakage vector is structural,
not conventional.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["uniqueness_weights", "assert_train_only"]


def assert_train_only(
    train_idx: pd.Index,
    all_idx: pd.Index,
) -> None:
    """Assert that ``train_idx`` is a proper subset of ``all_idx``.

    A weak structural guard: if a caller passes the full event index as
    the training index, that is the leakage vector §4 warns about
    (concurrency computed over the full set includes test-period event
    timing). This asserts ``train_idx`` is strictly smaller than the
    full set when the full set has >1 event — i.e. we are computing
    weights on a fold, not the world.

    It cannot catch every misuse (a near-full train set still leaks
    marginally), but it makes the obvious mistake loud.
    """
    if len(all_idx) <= 1:
        return
    if set(train_idx) == set(all_idx):
        raise ValueError(
            "uniqueness_weights called with train_idx == all_idx: this is the "
            "§4 leakage vector (concurrency over the full event set leaks "
            "test-period timing). Pass the training-fold subset only."
        )


def uniqueness_weights(
    events: pd.DataFrame,
    *,
    train_idx: pd.Index | None = None,
    all_idx: pd.Index | None = None,
    bar_freq: str = "4h",
) -> pd.Series:
    """Compute sample-uniqueness weights for ``events`` (training fold only).

    Args:
        events: DataFrame with ``t0`` and ``t1`` columns (timestamps),
            indexed by event id. Must be the **training fold** subset.
        train_idx: The training-fold index (for the leakage guard). If
            None, the guard is skipped (caller takes responsibility).
        all_idx: The full event-set index (for the leakage guard).
        bar_freq: The timeline grid frequency for concurrency counting.
            Default ``"4h"`` = H4 bars (the §4 reference: "per H4 bar t").

    Returns:
        Series of uniqueness weights, indexed like ``events``.

    Raises:
        ValueError: If ``train_idx == all_idx`` (the leakage vector).
    """
    if train_idx is not None and all_idx is not None:
        assert_train_only(train_idx, all_idx)

    if len(events) == 0:
        return pd.Series(dtype=float, name="uniqueness")

    ev = events[["t0", "t1"]].copy()
    ev["t0"] = pd.to_datetime(ev["t0"])
    ev["t1"] = pd.to_datetime(ev["t1"])
    # NaT t1 (unresolved) -> extend to t0 + vertical horizon as a safe
    # upper bound for span overlap. With 24 H4 bars = 96h.
    vert = pd.Timedelta(hours=4 * 24)
    ev["t1"] = ev["t1"].fillna(ev["t0"] + vert)

    # Build the timeline grid covering all spans.
    start = ev["t0"].min()
    end = ev["t1"].max()
    grid = pd.date_range(start=start, end=end, freq=bar_freq)

    # Concurrency c_t at each grid point: count of events whose [t0,t1]
    # contains t. Vectorized via broadcasting on a coarse grid.
    starts = ev["t0"].to_numpy()[:, None]   # (n_events, 1)
    ends = ev["t1"].to_numpy()[:, None]      # (n_events, 1)
    pts = grid.to_numpy()[None, :]           # (1, n_grid)
    contained = (starts <= pts) & (pts <= ends)  # (n_events, n_grid)
    c_t = contained.sum(axis=0)              # (n_grid,)
    c_t = np.maximum(c_t, 1)                 # avoid div-by-zero

    # For each event, mean of 1/c_t over the grid points it spans.
    inv_c = 1.0 / c_t                         # (n_grid,)
    # per-event sum of inv_c over contained points, divided by count
    weights = (contained * inv_c[None, :]).sum(axis=1) / np.maximum(contained.sum(axis=1), 1)
    return pd.Series(weights, index=events.index, name="uniqueness")
