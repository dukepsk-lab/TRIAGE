"""TRIAGE — Triple-Barrier Meta-Label Architecture.

See ``docs/ARCHITECTURE.md`` for the source of truth.

This package is being built phase-by-phase. Phase 0 implements the
correctness harness only: synthetic data, the triple-barrier labeler,
purged K-fold / embargo / CPCV validation, a feature-timestamp audit
utility, and the trial registry. The rule engine, feature service,
meta-model, sizer, and execution adapter are intentionally stubbed.
"""

__version__ = "0.0.1"
