"""Trial registry: every configuration ever attempted, with gate results.

Implements the trial-registry concern referenced throughout
ARCHITECTURE.md (§0 commitment #2: "Every tuned knob is a trial ... all
configurations attempted get logged to a central trial registry"; §2
data-flow diagram; §6 Optuna/HPO; §7 promotion gates computed "with
trial counts from the registry"; §9 "every attempt logs to the trial
registry — this is what keeps DSR/PBO honest").

Phase 0 uses a local SQLite store. Postgres/TimescaleDB is a later
phase; see ``docker-compose.yml`` for the planned service.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

__all__ = [
    "DEFAULT_DB_PATH",
    "TrialRecord",
    "TrialRegistry",
    "default_registry",
    "logged_trial",
    "config_hash",
]


#: Default on-disk location. Tests override via the ``TRIAGE_REGISTRY_DB``
#: env var or by constructing their own :class:`TrialRegistry`.
DEFAULT_DB_PATH: Path = Path(".triage") / "registry.sqlite"


# ---------------------------------------------------------------------------
# Schema.
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    run_id            TEXT    PRIMARY KEY,
    timestamp         TEXT    NOT NULL,
    config_hash       TEXT    NOT NULL,
    config_json       TEXT    NOT NULL,
    gate_results_json TEXT,
    status            TEXT    NOT NULL,    -- 'passed' | 'failed' | 'error'
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_config_hash ON trials(config_hash);
CREATE INDEX IF NOT EXISTS idx_trials_timestamp   ON trials(timestamp);
"""


@dataclass
class TrialRecord:
    """One row in the trial registry.

    Per the Phase 0 schema in the build prompt:
    ``run_id, timestamp, config_hash, config_json, gate_results_json``.
    ``status`` / ``error`` are added so a fail/error trial is recorded
    with its reason rather than dropped (§0: "writes a row regardless
    of pass/fail").
    """

    run_id: str
    timestamp: datetime
    config_hash: str
    config_json: str
    gate_results_json: Optional[str] = None
    status: str = "passed"            # 'passed' | 'failed' | 'error'
    error: Optional[str] = None

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "config_hash": self.config_hash,
            "config_json": self.config_json,
            "gate_results_json": self.gate_results_json,
            "status": self.status,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


class TrialRegistry:
    """A local SQLite-backed store of trial records.

    Every experiment run writes one row, regardless of outcome
    (ARCHITECTURE.md §0). This is the object the DSR/PBO gates query to
    recover an honest trial count (§7).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self._ensure_schema()

    # -- low-level connection -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False: tests and the decorator may run across
        # threads (e.g. pytest-xdist in a later phase). SQLite handles
        # serialization via the GIL + file lock for our write-once-per-trial
        # access pattern.
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # -- writes ---------------------------------------------------------------

    def log(self, record: TrialRecord) -> None:
        """Insert one trial record (the only write path)."""
        row = record.to_row()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trials
                    (run_id, timestamp, config_hash, config_json,
                     gate_results_json, status, error)
                VALUES
                    (:run_id, :timestamp, :config_hash, :config_json,
                     :gate_results_json, :status, :error)
                """,
                row,
            )
            conn.commit()

    # -- reads ----------------------------------------------------------------

    def __len__(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    def all_records(self) -> list[TrialRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trials ORDER BY timestamp").fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_by_status(self, status: str) -> int:
        with self._connect() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM trials WHERE status = ?", (status,)
                ).fetchone()[0]
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TrialRecord:
        return TrialRecord(
            run_id=row["run_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            config_hash=row["config_hash"],
            config_json=row["config_json"],
            gate_results_json=row["gate_results_json"],
            status=row["status"],
            error=row["error"],
        )


# ---------------------------------------------------------------------------
# Hashing + run id.
# ---------------------------------------------------------------------------


def config_hash(config: Any) -> str:
    """Stable SHA-256 hash of a JSON-serializable config object.

    The hash is what makes a "trial" countable: two runs with the same
    config_hash are the same trial (§0: "every tuned knob is a trial").
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _new_run_id(config_hash: str) -> str:
    """Deterministic-enough run id: hash + timestamp + pid suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{config_hash[:12]}-{ts}-{_PROCESS_TAG}"


# Process tag disambiguates two runs started in the same microsecond
# across separate processes (e.g. parallel pytest workers).
_PROCESS_TAG: str = f"p{__import__('os').getpid()}"


# ---------------------------------------------------------------------------
# Decorator + context manager.
# ---------------------------------------------------------------------------


def logged_trial(
    config: Any,
    *,
    registry: Optional[TrialRegistry] = None,
    run_id: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: log every invocation of a training/eval call as a trial.

    ARCHITECTURE.md §0: "Every experiment run (even a test run) writes
    to the trial registry. Build this as a decorator or context manager
    from day one, not something bolted on later."

    The wrapped function may:
    - return a dict of gate results -> recorded as ``gate_results_json``,
      status ``passed``;
    - raise :class:`AssertionError` (a gate failure) -> recorded with
      status ``failed`` and the assertion text, then re-raised;
    - raise any other exception -> recorded with status ``error`` and
      the traceback, then re-raised.

    A row is written **regardless** of pass/fail/error.

    Usage::

        @logged_trial({"rule": "donchian", "barriers": "1.5/1.0/24"})
        def run(cfg, events, features):
            ...
            return {"auc": 0.55, "brier": 0.21}
    """

    reg = registry if registry is not None else default_registry()
    chash = config_hash(config)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rid = run_id or _new_run_id(chash)
            started = datetime.now(timezone.utc)
            status = "passed"
            gate_results: Any = None
            err: Optional[str] = None
            try:
                result = fn(*args, **kwargs)
                gate_results = result
                return result
            except AssertionError as e:
                status = "failed"
                err = str(e) or repr(e)
                raise
            except Exception:  # noqa: BLE001 - log every failure
                status = "error"
                err = traceback.format_exc()
                raise
            finally:
                reg.log(
                    TrialRecord(
                        run_id=rid,
                        timestamp=started,
                        config_hash=chash,
                        config_json=json.dumps(config, default=str),
                        gate_results_json=(
                            json.dumps(gate_results, default=str)
                            if gate_results is not None
                            else None
                        ),
                        status=status,
                        error=err,
                    )
                )

        return wrapper

    return decorator


@contextmanager
def trial_context(
    config: Any,
    *,
    registry: Optional[TrialRegistry] = None,
    run_id: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    """Context-manager form of :func:`logged_trial`.

    Yields a dict the caller can populate with gate results; whatever is
    in it at exit is recorded. Same pass/fail/error semantics.
    """
    reg = registry if registry is not None else default_registry()
    chash = config_hash(config)
    rid = run_id or _new_run_id(chash)
    started = datetime.now(timezone.utc)
    bucket: dict[str, Any] = {}
    status = "passed"
    err: Optional[str] = None
    try:
        yield bucket
    except AssertionError as e:
        status = "failed"
        err = str(e) or repr(e)
        raise
    except Exception:  # noqa: BLE001
        status = "error"
        err = traceback.format_exc()
        raise
    finally:
        reg.log(
            TrialRecord(
                run_id=rid,
                timestamp=started,
                config_hash=chash,
                config_json=json.dumps(config, default=str),
                gate_results_json=json.dumps(bucket, default=str) if bucket else None,
                status=status,
                error=err,
            )
        )


# ---------------------------------------------------------------------------
# Module-level default registry (overridable for tests).
# ---------------------------------------------------------------------------

_default: Optional[TrialRegistry] = None


def default_registry() -> TrialRegistry:
    """Return the process-wide default registry, lazily created."""
    global _default
    if _default is None:
        _default = TrialRegistry(DEFAULT_DB_PATH)
    return _default


def set_default_registry(reg: Optional[TrialRegistry]) -> None:
    """Override (or clear) the default registry — used by tests."""
    global _default
    _default = reg
