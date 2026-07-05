"""Meta-label model (stub) — Phase 1+.

Implements ARCHITECTURE.md §6 ("Meta-label model"). The committed model
is a LightGBM binary classifier with the reference configuration in §6,
fit on uniqueness-weighted events under purged CV, with a first-class
calibration stage.

Phase 0 does **not** train or ship a model. The only model-shaped code
in Phase 0 is the *shuffled-label leakage harness* in
``tests/unit/test_shuffled_label.py``, which trains a trivial plain
LightGBM on permuted labels and asserts OOS AUC ~= 0.50 — that is a
harness deliverable, not the meta-model.
"""

from __future__ import annotations

__all__ = ["train", "predict", "calibrate"]


def train(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Train the LightGBM meta-model on uniqueness-weighted events.

    NOT IMPLEMENTED in Phase 0. Phase 1 will fit the §6 reference config
    under purged CPCV with sample weights and early stopping on a purged
    validation fold.
    """
    raise NotImplementedError(
        "Meta-model training is a Phase 1+ deliverable (ARCHITECTURE.md §6, §10)."
    )


def predict(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Return calibrated P(win) for a batch of feature snapshots.

    NOT IMPLEMENTED in Phase 0. See ARCHITECTURE.md §6.
    """
    raise NotImplementedError(
        "Meta-model inference is a Phase 1+ deliverable (ARCHITECTURE.md §6)."
    )


def calibrate(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Platt/isotonic calibration on a purged calibration fold.

    NOT IMPLEMENTED in Phase 0. §6: "Calibration is a first-class
    stage, not a nicety — the sizing layer consumes P(win) as a literal
    probability." Platt below ~2k events, isotonic above.
    """
    raise NotImplementedError(
        "Calibration is a Phase 1+ deliverable (ARCHITECTURE.md §6)."
    )
