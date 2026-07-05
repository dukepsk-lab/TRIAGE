"""Triple-barrier labeling pipeline.

Implements ARCHITECTURE.md §4 ("Labeling pipeline").

The labeler resolves an event against the M1 bar stream to determine
which barrier was touched first (TP, SL, or vertical timeout), assigns
a binary label (1 iff TP), and records the signed realized R-multiple.
This is the module every downstream correctness check ultimately rests
on, so it is kept deliberately close to the §4 pseudocode.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional

from ..data import Bar, Bars, Side

__all__ = [
    "BARRIERS",
    "TP_MULT",
    "SL_MULT",
    "VERTICAL_BARS",
    "ATR_PERIOD",
    "OutcomeKind",
    "Outcome",
    "Event",
    "LabeledEvent",
    "resolve",
    "label_event",
    "compute_barriers",
]

# ---------------------------------------------------------------------------
# Frozen barrier constants — ARCHITECTURE.md §3 / §4.
#
# These are PRE-REGISTERED CONSTANTS for Phase 0/1, not free parameters
# (§0 commitment #2: "Every tuned knob is a trial ... Barriers and label
# policy are pre-registered constants"). Do not expose them as tunable.
#
# TODO(Phase 1): move these into configs/rules/<rule>.yaml under the
# ``barriers:`` block (see §3 example) once the config loader exists.
# They remain non-tunable there too — they are logged as a trial the
# moment a run uses any value other than these.
# ---------------------------------------------------------------------------

#: Take-profit multiplier vs. ATR. ARCHITECTURE.md §3 (``tp_atr: 1.5``).
TP_MULT: float = 1.5

#: Stop-loss multiplier vs. ATR. ARCHITECTURE.md §3 (``sl_atr: 1.0``).
SL_MULT: float = 1.0

#: Vertical barrier horizon in H4 bars. ARCHITECTURE.md §3
#: (``vertical_bars: 24``).
VERTICAL_BARS: int = 24

#: ATR computation period. ARCHITECTURE.md §3 (``atr.period: 14``).
#: Phase 0 labeler accepts ``atr0`` directly on the event, so this is a
#: reference value only; the ATR service itself lands in Phase 1.
ATR_PERIOD: int = 14

#: Read-only view of the frozen barrier set, for documentation / logging.
BARRIERS = {
    "tp_atr": TP_MULT,
    "sl_atr": SL_MULT,
    "vertical_bars": VERTICAL_BARS,
    "atr_period": ATR_PERIOD,
}


# ---------------------------------------------------------------------------
# Event / outcome types.
# ---------------------------------------------------------------------------

#: The three terminal states a triple-barrier event can resolve to.
#: ARCHITECTURE.md §4.
OutcomeKind = Literal["tp", "sl", "vert"]


@dataclass(frozen=True)
class Outcome:
    """Result of resolving an event against the M1 path.

    ``kind`` is one of ``tp`` / ``sl`` / ``vert``. ``t1`` is the
    first-touch time (or the vertical barrier time on timeout). The
    same-M1-bar TP+SL ambiguity resolves to ``sl`` — the conservative
    tie-break mandated by §4 ("deliberate pessimism; at M1 granularity
    it is rare, but the rule must be deterministic and documented").
    """

    kind: OutcomeKind
    t1: datetime

    @property
    def is_tp(self) -> bool:
        return self.kind == "tp"


@dataclass(frozen=True)
class Event:
    """A candidate signal awaiting barrier resolution.

    Mirrors the columns of the ``events`` table (ARCHITECTURE.md §4)
    that the labeler actually consumes. ``entry_px`` is the next-bar-open
    entry (§4: "entry at next H4 bar open"); ``tp_px`` / ``sl_px`` are
    the frozen barrier prices; ``t_vert`` is the vertical-barrier close
    time = entry + ``VERTICAL_BARS`` H4 bars.
    """

    symbol: str
    rule_id: str
    rule_version: str
    t0: datetime          # signal bar close
    side: Side            # +1 long / -1 short
    entry_px: float       # next-bar-open entry
    atr0: float           # ATR at t0, from closed bars <= t0 only
    tp_px: float          # = entry_px + side * TP_MULT * atr0
    sl_px: float          # = entry_px - side * SL_MULT * atr0
    t_vert: datetime      # vertical barrier close time


@dataclass(frozen=True)
class LabeledEvent:
    """An event plus its resolved outcome, label, and realized R.

    Per §4 timeout policy: ``label = 1`` iff outcome = TP; SL and
    vertical timeout both -> 0. ``r_realized`` (signed R-multiple) is
    stored on **every** event regardless of label — it feeds EV analysis
    and later weighting experiments. Verticals are mark-to-market.
    """

    event: Event
    outcome: Outcome
    label: int            # 1 iff TP
    r_realized: float     # signed R-multiple at exit


# ---------------------------------------------------------------------------
# Barrier price construction.
# ---------------------------------------------------------------------------


def compute_barriers(
    entry_px: float, atr0: float, side: Side
) -> tuple[float, float]:
    """Return ``(tp_px, sl_px)`` for the frozen barrier set.

    Implements the geometry implied by ARCHITECTURE.md §3/§4: TP is
    ``TP_MULT * atr0`` in the trade direction, SL is ``SL_MULT * atr0``
    against it. Both barriers use the same ``atr0`` (computed from bars
    closing <= t0 only — see :mod:`triage.features.no_lookahead`).
    """
    if side == 1:
        tp_px = entry_px + TP_MULT * atr0
        sl_px = entry_px - SL_MULT * atr0
    elif side == -1:
        tp_px = entry_px - TP_MULT * atr0
        sl_px = entry_px + SL_MULT * atr0
    else:  # pragma: no cover - Side is Literal[1,-1]
        raise ValueError(f"side must be +1 or -1, got {side!r}")
    return tp_px, sl_px


# ---------------------------------------------------------------------------
# Core resolver — ARCHITECTURE.md §4.
# ---------------------------------------------------------------------------


def resolve(event: Event, m1_bars: Bars) -> Outcome:
    """First-touch barrier resolution against an M1 bar stream.

    Implements ``resolve()`` per ARCHITECTURE.md §4 exactly, including
    the same-M1-bar TP+SL tie-break to SL (the conservative, named
    branch — not incidental). Returns the :class:`Outcome` (kind +
    first-touch time).

    Args:
        event: The event with frozen ``tp_px`` / ``sl_px`` / ``t_vert``.
        m1_bars: M1 bars in ``(entry_time, t_vert]`` — i.e. strictly
            after the entry bar up to and including the vertical
            barrier. Callers are responsible for slicing; the resolver
            itself just walks what it is given and falls through to
            ``vert`` at the end.

    Notes:
        The §4 pseudocode checks TP first inside the same-bar branch but
        returns SL when both hit — that is the deterministic tie-break.
        We preserve the same ordering here so the behavior is auditable
        against the doc line-by-line.
    """
    for bar in m1_bars:
        if event.side == 1:
            tp_hit = bar.high >= event.tp_px
            sl_hit = bar.low <= event.sl_px
        else:  # side == -1
            tp_hit = bar.low <= event.tp_px
            sl_hit = bar.high >= event.sl_px

        # Named branch: same-M1-bar ambiguity -> SL (§4, "deliberate
        # pessimism"). Tested explicitly in tests/unit/test_labeler.py.
        if tp_hit and sl_hit:
            return Outcome(kind="sl", t1=bar.close_time)
        if sl_hit:
            return Outcome(kind="sl", t1=bar.close_time)
        if tp_hit:
            return Outcome(kind="tp", t1=bar.close_time)

    # No horizontal barrier touched inside the window -> vertical timeout.
    return Outcome(kind="vert", t1=event.t_vert)


# ---------------------------------------------------------------------------
# Realized R-multiple.
# ---------------------------------------------------------------------------


def _r_realized(event: Event, outcome: Outcome, m1_bars: Bars) -> float:
    """Signed R-multiple at exit, in units of ``SL_MULT * atr0`` risk.

    Per §4 ``r_realized`` is stored on every event regardless of label.
    TP exits realize ``+TP_MULT/SL_MULT`` R (= +1.5 with the frozen
    barriers); SL exits realize ``-1.0`` R. Verticals are mark-to-market
    at the close of the bar at ``t_vert`` (the bar whose close_time ==
    t_vert), conservatively; §8 values verticals at 0 for EV, but the
    event store keeps the actual mark so EV/ablation analysis can
    revisit it (§4's one pre-registered label ablation is exactly
    "vertical -> sign(r_realized) instead of 0").
    """
    risk = SL_MULT * event.atr0  # 1R in price units
    if risk <= 0:
        # atr0 == 0 collapses the barriers; nothing meaningful to scale.
        return 0.0

    if outcome.kind == "tp":
        return float(event.side) * (event.tp_px - event.entry_px) / risk
    if outcome.kind == "sl":
        return float(event.side) * (event.sl_px - event.entry_px) / risk

    # Vertical: mark-to-market at t_vert. Find the bar whose close_time
    # matches t_vert; if none (shouldn't happen for well-formed inputs),
    # fall back to 0R per §8's conservative vertical valuation.
    exit_px: Optional[float] = None
    for bar in m1_bars:
        if bar.close_time == event.t_vert:
            exit_px = bar.close
            break
    if exit_px is None:
        return 0.0
    return float(event.side) * (exit_px - event.entry_px) / risk


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------


def label_event(event: Event, m1_bars: Bars) -> LabeledEvent:
    """Resolve ``event`` and return its full labeled form.

    Convenience wrapper combining :func:`resolve` + label policy +
    ``r_realized`` (ARCHITECTURE.md §4 timeout policy).
    """
    outcome = resolve(event, m1_bars)
    # label = 1 iff outcome is TP; SL and vertical timeout both -> 0.
    label = 1 if outcome.kind == "tp" else 0
    r = _r_realized(event, outcome, m1_bars)
    return LabeledEvent(event=event, outcome=outcome, label=label, r_realized=r)


def make_event(
    *,
    symbol: str,
    rule_id: str,
    rule_version: str,
    t0: datetime,
    side: Side,
    entry_px: float,
    atr0: float,
    h4_bar_seconds: int = 4 * 3600,
) -> Event:
    """Construct an :class:`Event` with barriers + ``t_vert`` filled in.

    Small helper so test fixtures and the trivial stub rule share one
    construction path. ``t_vert`` is ``entry + VERTICAL_BARS * H4``.
    """
    tp_px, sl_px = compute_barriers(entry_px, atr0, side)
    t_vert = t0 + timedelta(seconds=h4_bar_seconds * VERTICAL_BARS)
    return Event(
        symbol=symbol,
        rule_id=rule_id,
        rule_version=rule_version,
        t0=t0,
        side=side,
        entry_px=entry_px,
        atr0=atr0,
        tp_px=tp_px,
        sl_px=sl_px,
        t_vert=t_vert,
    )
