"""No-lookahead audit utility (§4/§5/§7 leakage checklist).

Lives in its own module so :mod:`triage.features.service` can import it
without a circular dependency on :mod:`triage.features.__init__`.

This is the enforcement mechanism for the no-look-ahead invariant
(ARCHITECTURE.md §4 / §7): "features from closed bars <= t0". It is
*assertable*, not a comment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

__all__ = ["LookaheadError", "no_lookahead"]


class LookaheadError(AssertionError):
    """Raised when a feature snapshot references bars after its t0.

    Subclasses :class:`AssertionError` so ``assert``-style calls and
    plain pytest assertions surface it naturally.
    """


def no_lookahead(
    snapshot: Mapping[str, object],
    t0: datetime,
    *,
    timestamp_keys: Sequence[str] = ("source_close_time",),
    exact_keys: Sequence[str] = ("max_source_close_time",),
) -> datetime:
    """Assert that no component source timestamp in ``snapshot`` exceeds ``t0``.

    See ARCHITECTURE.md §4 / §7. A field is a source timestamp if its
    name appears in ``exact_keys`` or ends with any suffix in
    ``timestamp_keys``. A leak in any single component trips the audit.

    Args:
        snapshot: Feature snapshot (flat mapping).
        t0: Signal bar close time. Every source timestamp must be ``<= t0``.
        timestamp_keys: Suffixes identifying per-component timestamps.
        exact_keys: Exact field names identifying aggregate timestamps.

    Returns:
        The maximum source timestamp found.

    Raises:
        LookaheadError: If any source timestamp > t0 (a leak), or if the
            snapshot has no auditable timestamp (fail closed).
    """
    found: list[datetime] = []

    def _parse(key: str, val: object) -> datetime:
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val))
        except (ValueError, TypeError):
            raise LookaheadError(
                f"timestamp field {key!r} is not datetime-parseable: {val!r}"
            )

    for key, val in snapshot.items():
        is_exact = key in exact_keys
        is_suffix = any(key.endswith(s) for s in timestamp_keys)
        if is_exact or is_suffix:
            found.append(_parse(key, val))

    if not found:
        raise LookaheadError(
            "snapshot is not auditable: no timestamp field found (looked for "
            f"exact keys {list(exact_keys)!r} and suffixes {list(timestamp_keys)!r}). "
            "Fail closed per the no-look-ahead invariant (ARCHITECTURE.md §4/§7)."
        )

    max_ts = max(found)
    if max_ts > t0:
        raise LookaheadError(
            f"look-ahead detected: max source timestamp {max_ts} > t0 {t0} "
            "(ARCHITECTURE.md §4: features must come from bars closing <= t0)"
        )
    return max_ts
