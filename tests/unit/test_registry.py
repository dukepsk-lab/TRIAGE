"""Trial registry tests (Phase 0).

ARCHITECTURE.md §0: "Every experiment run (even a test run) writes to
the trial registry. Build this as a decorator or context manager from
day one."

The registry is exercised in three modes — passed, failed (assertion),
and errored — to confirm a row is written regardless of outcome.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from triage.registry import (
    DEFAULT_DB_PATH,
    TrialRecord,
    TrialRegistry,
    config_hash,
    logged_trial,
    trial_context,
)
from datetime import datetime, timezone


@pytest.fixture
def registry(tmp_path: Path) -> TrialRegistry:
    """A fresh on-disk registry in a tmp dir for each test."""
    return TrialRegistry(tmp_path / "reg.sqlite")


def test_schema_created(registry: TrialRegistry, tmp_path: Path):
    """The trials table exists with the required columns."""
    conn = sqlite3.connect(str(tmp_path / "reg.sqlite"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trials)").fetchall()}
    conn.close()
    required = {
        "run_id", "timestamp", "config_hash", "config_json",
        "gate_results_json", "status", "error",
    }
    assert required <= cols, f"missing columns: {required - cols}"


def test_passed_trial_is_logged(registry: TrialRegistry):
    """A passing run writes a row with status='passed' + gate results."""
    @logged_trial({"rule": "x", "barriers": "1.5/1.0/24"}, registry=registry)
    def run(cfg):
        return {"auc": 0.55, "brier": 0.21}

    result = run({"rule": "x"})
    assert result == {"auc": 0.55, "brier": 0.21}
    assert len(registry) == 1
    rec = registry.all_records()[0]
    assert rec.status == "passed"
    assert rec.gate_results_json is not None
    assert "0.55" in rec.gate_results_json
    assert rec.error is None
    assert rec.config_hash == config_hash({"rule": "x", "barriers": "1.5/1.0/24"})


def test_failed_trial_is_logged(registry: TrialRegistry):
    """An AssertionError (gate failure) is logged as 'failed' then re-raised."""
    @logged_trial({"rule": "y"}, registry=registry)
    def run(cfg):
        assert False, "G2 DSR below 0.95"

    with pytest.raises(AssertionError):
        run({"rule": "y"})

    assert len(registry) == 1
    rec = registry.all_records()[0]
    assert rec.status == "failed"
    assert "G2 DSR" in (rec.error or "")


def test_errored_trial_is_logged(registry: TrialRegistry):
    """A generic exception is logged as 'error' with traceback then re-raised."""
    @logged_trial({"rule": "z"}, registry=registry)
    def run(cfg):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run({"rule": "z"})

    assert len(registry) == 1
    rec = registry.all_records()[0]
    assert rec.status == "error"
    assert "RuntimeError" in (rec.error or "")


def test_context_manager_form_logs(registry: TrialRegistry):
    """The context-manager variant also writes a row regardless of outcome."""
    config = {"rule": "ctx"}
    with trial_context(config, registry=registry) as bucket:
        bucket["auc"] = 0.57
        bucket["passed_gates"] = True

    assert len(registry) == 1
    rec = registry.all_records()[0]
    assert rec.status == "passed"
    assert "0.57" in (rec.gate_results_json or "")


def test_two_trials_with_same_config_share_hash(registry: TrialRegistry):
    """Same config -> same hash (§0: this is what makes a trial countable)."""
    cfg = {"rule": "dup", "barriers": [1.5, 1.0, 24]}

    @logged_trial(cfg, registry=registry)
    def run(cfg):
        return {"auc": 0.5}

    run(cfg)
    run(cfg)

    recs = registry.all_records()
    assert len(recs) == 2
    assert recs[0].config_hash == recs[1].config_hash
    assert registry.count_by_status("passed") == 2


def test_config_hash_is_stable_and_order_independent():
    """Dict key order must not change the hash."""
    a = config_hash({"a": 1, "b": 2})
    b = config_hash({"b": 2, "a": 1})
    assert a == b


def test_default_db_path_under_triage_dir():
    """Default registry path lives under .triage/ (kept out of the way)."""
    assert DEFAULT_DB_PATH == Path(".triage") / "registry.sqlite"


def test_run_id_is_unique_per_call(registry: TrialRegistry):
    """Two runs get distinct run_ids even with identical configs."""
    @logged_trial({"rule": "uniq"}, registry=registry)
    def run(cfg):
        return {"auc": 0.5}

    run({"rule": "uniq"})
    run({"rule": "uniq"})

    recs = registry.all_records()
    assert recs[0].run_id != recs[1].run_id
