"""Synthetic M1 path generators with analytically known outcomes.

Implements the "synthetic M1 paths with analytically known outcomes"
leg of the ARCHITECTURE.md §4 correctness battery ("Correctness battery
(Phase 0, blocking)").

Each generator returns a :class:`SyntheticCase` carrying:
  - the :class:`triage.labeling.Event` (barriers + side + t_vert);
  - the M1 :class:`triage.data.Bars` the resolver will walk;
  - the **analytically known** expected :class:`triage.labeling.Outcome`,
    plus the expected label and expected ``r_realized``.

Tests assert the resolver's output against these ground-truth values.
The expected values are derived by hand from the path geometry, *not*
by running the resolver — that is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

from triage.data import Bar, Bars, Side
from triage.labeling import (
    Event,
    LabeledEvent,
    Outcome,
    SL_MULT,
    TP_MULT,
    VERTICAL_BARS,
    compute_barriers,
    make_event,
)

__all__ = [
    "M1",
    "SyntheticCase",
    "linear_tp_long",
    "linear_sl_long",
    "flat_timeout",
    "same_bar_ambiguity_long",
    "linear_tp_short",
    "late_tp_long",
]

#: M1 bar length in seconds.
M1: int = 60


@dataclass(frozen=True)
class SyntheticCase:
    """A synthetic path + its analytically known resolution.

    The fields under ``expected_*`` are the ground truth tests assert
    against. They are computed by hand from the path's geometry, so a
    passing test means the resolver agrees with an independent
    derivation — not that it agrees with itself.
    """

    name: str
    event: Event
    m1_bars: Bars
    expected_outcome: Outcome
    expected_label: int
    expected_r_realized: float
    note: str = ""


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _m1_bars_after(
    entry_time: datetime,
    n_bars: int,
    prices: list[tuple[float, float, float, float]],
) -> Bars:
    """Build M1 bars starting one minute after ``entry_time``.

    ``prices`` is a list of ``(open, high, low, close)`` tuples. Each
    M1 bar spans one minute; the resolver walks bars in
    ``(entry_time, t_vert]`` (§4).
    """
    bars: Bars = []
    base = entry_time + timedelta(seconds=M1)
    for i, (o, h, l, c) in enumerate(prices):
        open_time = base + timedelta(seconds=M1 * i)
        close_time = open_time + timedelta(seconds=M1)
        bars.append(
            Bar(
                open_time=open_time,
                close_time=close_time,
                open=o,
                high=h,
                low=l,
                close=c,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# Generators. Entry price = 100.00, ATR0 = 1.00 everywhere unless noted,
# so barriers are in clean round numbers:
#   long:  tp = 100 + 1.5 = 101.50,  sl = 100 - 1.0 = 99.00
#   short: tp = 100 - 1.5 =  98.50,  sl = 100 + 1.0 = 101.00
# ---------------------------------------------------------------------------


def _base_event(
    side: Side,
    *,
    t0: Optional[datetime] = None,
    entry_px: float = 100.0,
    atr0: float = 1.0,
) -> tuple[Event, datetime]:
    t0 = t0 or datetime(2026, 1, 5, 8, 0, 0)
    event = make_event(
        symbol="SYNTH",
        rule_id="trivial_long",
        rule_version="synthetic_v0",
        t0=t0,
        side=side,
        entry_px=entry_px,
        atr0=atr0,
        # Keep H4 bar length = 4h so t_vert = t0 + 24*4h = 4 days later.
        h4_bar_seconds=4 * 3600,
    )
    return event, t0


def linear_tp_long() -> SyntheticCase:
    """Deterministic ramp that touches TP on bar 3, never near SL.

    Long @100, TP=101.50, SL=99.00. Prices step up 0.50/bar; TP hit on
    the 3rd M1 bar (high 101.50). Expected: outcome=tp @ bar-3 close,
    label=1, r=+1.5.
    """
    event, t0 = _base_event(side=1)
    entry_time = t0  # entry conceptually at next H4 open; for the M1
    # stream we only need bars *after* entry, which we anchor at t0.
    prices = [
        (100.00, 100.50, 99.95, 100.50),  # bar 1
        (100.50, 101.00, 100.40, 101.00),  # bar 2
        (101.00, 101.50, 100.90, 101.50),  # bar 3 -> TP touched
        (101.50, 101.80, 101.40, 101.70),  # bar 4 (unreached)
    ]
    bars = _m1_bars_after(entry_time, len(prices), prices)
    tp_close_time = bars[2].close_time
    return SyntheticCase(
        name="linear_tp_long",
        event=event,
        m1_bars=bars,
        expected_outcome=Outcome(kind="tp", t1=tp_close_time),
        expected_label=1,
        expected_r_realized=+TP_MULT / SL_MULT,  # +1.5
        note="Clean single-touch TP on bar 3; SL never approached.",
    )


def linear_sl_long() -> SyntheticCase:
    """Deterministic ramp that touches SL on bar 2, never near TP.

    Long @100, TP=101.50, SL=99.00. Prices step down; SL hit on bar 2
    (low 99.00). Expected: outcome=sl @ bar-2 close, label=0, r=-1.0.
    """
    event, t0 = _base_event(side=1)
    prices = [
        (100.00, 100.10, 99.50, 99.60),  # bar 1
        (99.60, 99.70, 99.00, 99.10),    # bar 2 -> SL touched
        (99.10, 99.40, 98.80, 98.90),    # bar 3 (unreached)
    ]
    bars = _m1_bars_after(t0, len(prices), prices)
    sl_close_time = bars[1].close_time
    return SyntheticCase(
        name="linear_sl_long",
        event=event,
        m1_bars=bars,
        expected_outcome=Outcome(kind="sl", t1=sl_close_time),
        expected_label=0,
        expected_r_realized=-1.0,
        note="Clean single-touch SL on bar 2; TP never approached.",
    )


def flat_timeout() -> SyntheticCase:
    """Flat path that touches neither barrier -> vertical timeout.

    Long @100, TP=101.50, SL=99.00. Every bar is the identical flat
    100.00 +- 0.10, so neither barrier is ever touched and the resolver
    walks the whole stream and returns ``vert`` at ``t_vert``.

    For the timeout to be reachable inside a tractable test we pass a
    *short* M1 stream whose final bar closes exactly at ``t_vert``;
    ``_r_realized`` then marks to that bar's close (100.00) -> r=0.
    Expected: outcome=vert @ t_vert, label=0, r=0.0.
    """
    event, t0 = _base_event(side=1)
    # Build bars in (t0, t_vert] with the LAST bar closing exactly at
    # t_vert, so the vertical outcome's t1 and the mark-to-market land
    # on a real bar. The first M1 bar opens strictly after entry (t0);
    # each bar is 1 minute. To close exactly at t_vert we need
    # n_bars = (t_vert - t0)/1min - 1.
    t_vert = event.t_vert
    span_min = int((t_vert - t0).total_seconds() // M1)  # minutes t0..t_vert
    n_bars = span_min - 1
    assert n_bars >= 1, "vertical horizon too short for an M1 walk"
    flat = (100.00, 100.10, 99.90, 100.00)
    prices = [flat] * n_bars
    bars = _m1_bars_after(t0, n_bars, prices)
    # Invariant: the last bar closes exactly at t_vert (anchored by hand).
    assert bars[-1].close_time == t_vert, (
        f"flat_timeout fixture misaligned: last close {bars[-1].close_time} "
        f"!= t_vert {t_vert}"
    )
    return SyntheticCase(
        name="flat_timeout",
        event=event,
        m1_bars=bars,
        expected_outcome=Outcome(kind="vert", t1=t_vert),
        expected_label=0,
        expected_r_realized=0.0,
        note="Flat path; neither barrier touched; vertical timeout at t_vert.",
    )


def same_bar_ambiguity_long() -> SyntheticCase:
    """One M1 bar whose range spans BOTH barriers -> SL (tie-break).

    Long @100, TP=101.50, SL=99.00. The first M1 bar has high=101.60
    (>= TP) AND low=98.90 (<= SL). Per §4's named tie-break this
    resolves to SL — "deliberate pessimism". This is the test the
    prompt calls out: a named, tested branch, not incidental behavior.

    Expected: outcome=sl @ bar-1 close, label=0, r=-1.0.
    """
    event, t0 = _base_event(side=1)
    prices = [
        # bar 1 spans both barriers in a single M1 candle
        (100.00, 101.60, 98.90, 100.20),
        (100.20, 100.40, 100.00, 100.30),  # bar 2 (unreached)
    ]
    bars = _m1_bars_after(t0, len(prices), prices)
    sl_close_time = bars[0].close_time
    return SyntheticCase(
        name="same_bar_ambiguity_long",
        event=event,
        m1_bars=bars,
        expected_outcome=Outcome(kind="sl", t1=sl_close_time),
        expected_label=0,
        expected_r_realized=-1.0,
        note=(
            "§4 named tie-break: same M1 bar touches TP and SL -> SL "
            "(conservative). This is a deliberate, tested branch."
        ),
    )


def linear_tp_short() -> SyntheticCase:
    """Short-side mirror: deterministic ramp down to TP.

    Short @100, TP=98.50, SL=101.00. Prices step down; TP touched on
    bar 2 (low 98.50). Exercises the ``side == -1`` branch of resolve()
    — for shorts, tp is checked via ``bar.low <= tp_px`` and sl via
    ``bar.high >= sl_px``.

    Expected: outcome=tp @ bar-2 close, label=1, r=+1.5.
    """
    event, t0 = _base_event(side=-1)
    prices = [
        (100.00, 100.10, 99.50, 99.50),  # bar 1
        (99.50, 99.60, 98.50, 98.60),    # bar 2 -> TP touched (low <= 98.50)
        (98.60, 98.80, 98.30, 98.40),    # bar 3 (unreached)
    ]
    bars = _m1_bars_after(t0, len(prices), prices)
    tp_close_time = bars[1].close_time
    return SyntheticCase(
        name="linear_tp_short",
        event=event,
        m1_bars=bars,
        expected_outcome=Outcome(kind="tp", t1=tp_close_time),
        expected_label=1,
        expected_r_realized=+TP_MULT / SL_MULT,  # +1.5
        note="Short side: TP via low <= tp_px; verifies side==-1 branch.",
    )


def late_tp_long() -> SyntheticCase:
    """TP touched only on a later bar after several flat bars.

    Long @100, TP=101.50, SL=99.00. Bars 1-4 are tame (never near
    either barrier); bar 5 spikes up to TP. Verifies the resolver keeps
    walking and does not short-circuit, and that ``t1`` is the bar-5
    close (not bar 1).

    Expected: outcome=tp @ bar-5 close, label=1, r=+1.5.
    """
    event, t0 = _base_event(side=1)
    prices = [
        (100.00, 100.20, 99.90, 100.10),  # bar 1
        (100.10, 100.25, 99.95, 100.15),  # bar 2
        (100.15, 100.30, 100.00, 100.20),  # bar 3
        (100.20, 100.35, 100.05, 100.25),  # bar 4
        (100.25, 101.60, 100.20, 101.55),  # bar 5 -> TP touched
    ]
    bars = _m1_bars_after(t0, len(prices), prices)
    tp_close_time = bars[4].close_time
    return SyntheticCase(
        name="late_tp_long",
        event=event,
        m1_bars=bars,
        expected_outcome=Outcome(kind="tp", t1=tp_close_time),
        expected_label=1,
        expected_r_realized=+TP_MULT / SL_MULT,
        note="TP only on bar 5 after tame bars; resolver must keep walking.",
    )


# ---------------------------------------------------------------------------
# Convenience registry.
# ---------------------------------------------------------------------------


def all_cases() -> list[SyntheticCase]:
    """Return every synthetic case — the canonical Phase 0 battery."""
    return [
        linear_tp_long(),
        linear_sl_long(),
        flat_timeout(),
        same_bar_ambiguity_long(),
        linear_tp_short(),
        late_tp_long(),
    ]
