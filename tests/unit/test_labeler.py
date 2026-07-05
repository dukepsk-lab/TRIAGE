"""Labeler tests: resolve() correctness against every synthetic path.

Asserts exact match to the analytically known outcome, label, and
r_realized for every case in ``tests.synthetic.all_cases()``.
This is the core of the ARCHITECTURE.md §4 correctness battery.
"""

from __future__ import annotations

import pytest

from triage.labeling import (
    BARRIERS,
    SL_MULT,
    TP_MULT,
    VERTICAL_BARS,
    label_event,
    resolve,
)

from tests.synthetic import all_cases, same_bar_ambiguity_long


# ---------------------------------------------------------------------------
# Barrier constants are frozen — Phase 0/1 pre-registered, not tunable.
# ---------------------------------------------------------------------------


def test_barrier_constants_match_architecture():
    """§3/§4: TP=1.5×ATR, SL=1.0×ATR, vertical=24 H4 bars."""
    assert TP_MULT == 1.5
    assert SL_MULT == 1.0
    assert VERTICAL_BARS == 24
    assert BARRIERS == {
        "tp_atr": 1.5,
        "sl_atr": 1.0,
        "vertical_bars": 24,
        "atr_period": 14,
    }


# ---------------------------------------------------------------------------
# Parametrized exact-match over the whole synthetic battery.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_resolve_matches_known_outcome(case):
    """resolve() returns the analytically known outcome kind + t1."""
    got = resolve(case.event, case.m1_bars)
    assert got.kind == case.expected_outcome.kind, (
        f"{case.name}: outcome kind {got.kind!r} != "
        f"expected {case.expected_outcome.kind!r}"
    )
    assert got.t1 == case.expected_outcome.t1, (
        f"{case.name}: t1 {got.t1} != expected {case.expected_outcome.t1}"
    )


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_label_and_r_realized_match(case):
    """label_event() assigns the known label and signed R-multiple."""
    labeled = label_event(case.event, case.m1_bars)
    assert labeled.label == case.expected_label, (
        f"{case.name}: label {labeled.label} != expected {case.expected_label}"
    )
    # r_realized is exact for tp/sl; for vertical it is mark-to-market
    # at the t_vert bar. Use a tight tolerance for float compare.
    assert labeled.r_realized == pytest.approx(
        case.expected_r_realized, abs=1e-9
    ), f"{case.name}: r_realized {labeled.r_realized} != expected"


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_label_policy_tp_iff_one(case):
    """§4 timeout policy: label=1 iff outcome=TP; SL and vert -> 0."""
    labeled = label_event(case.event, case.m1_bars)
    if labeled.outcome.kind == "tp":
        assert labeled.label == 1
    else:
        # sl or vert
        assert labeled.label == 0


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_r_realized_always_present(case):
    """§4: r_realized stored on every event regardless of label."""
    labeled = label_event(case.event, case.m1_bars)
    assert isinstance(labeled.r_realized, float)
    # TP -> +1.5, SL -> -1.0 by construction; vertical is whatever mark.
    if labeled.outcome.kind == "tp":
        assert labeled.r_realized == pytest.approx(+TP_MULT / SL_MULT)
    elif labeled.outcome.kind == "sl":
        assert labeled.r_realized == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Named, isolated test for the same-bar tie-break (per the prompt).
# ---------------------------------------------------------------------------


def test_same_bar_ambiguity_resolves_to_sl():
    """§4: same M1 bar touching TP+SL resolves to SL (conservative).

    The prompt requires this be a named, tested branch — not incidental
    behavior. This test isolates exactly that branch and pins it.
    """
    case = same_bar_ambiguity_long()
    out = resolve(case.event, case.m1_bars)
    assert out.kind == "sl", (
        "Same-M1-bar TP+SL ambiguity MUST resolve to SL per §4's "
        "conservative tie-break. Got %r." % out.kind
    )
    labeled = label_event(case.event, case.m1_bars)
    assert labeled.label == 0
    assert labeled.r_realized == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Determinism: same inputs -> same outputs, twice in a row.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", all_cases(), ids=lambda c: c.name)
def test_resolve_is_deterministic(case):
    a = label_event(case.event, case.m1_bars)
    b = label_event(case.event, case.m1_bars)
    assert a == b
