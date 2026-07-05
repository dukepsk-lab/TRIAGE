"""Execution adapter (stub) — Phase 4.

Implements the MT5 execution concern referenced in ARCHITECTURE.md §2
(``MT5 EXEC ADAPTER``) and §9 ("MT5 execution + Telegram: unchanged;
consumes (symbol, side, size) with barrier prices attached").

Phase 0 does not touch execution. Live execution is a Phase 4 concern
(§10), and even Phase 1 only does offline backfill labeling. The stub
exists so the import surface is stable.
"""

from __future__ import annotations

__all__ = ["send_order", "watch_barriers"]


def send_order(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Submit an order to MT5 with barrier prices attached.

    NOT IMPLEMENTED. Phase 4 (ARCHITECTURE.md §9, §10).
    """
    raise NotImplementedError(
        "Execution adapter is a Phase 4 deliverable (ARCHITECTURE.md §9, §10)."
    )


def watch_barriers(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Live TP/SL/vertical watcher (incremental barrier resolution).

    NOT IMPLEMENTED. The vectorized backfill form lives in
    :mod:`triage.labeling`; this is the live twin (§2/§9). Phase 4.
    """
    raise NotImplementedError(
        "Live barrier watcher is a Phase 4 deliverable (ARCHITECTURE.md §9, §10)."
    )
