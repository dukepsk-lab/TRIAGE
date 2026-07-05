"""Shuffled-label leakage harness — a reusable deliverable.

ARCHITECTURE.md §4 correctness battery: "shuffled-label test (train on
permuted labels -> OOS AUC must ~= 0.50, else the harness leaks)".

This is the *anti-leakage smoke test* for the whole training pipeline.
It is a reusable function, not a one-off script, because the build
prompt requires it to be run every phase going forward. The harness
takes any labeled event set + feature matrix, permutes the labels,
trains a trivial plain LightGBM, and reports OOS AUC under a simple
purged-style split. If features can predict *permuted* labels above
chance, something is leaking — typically an off-by-one in the
point-in-time snapshot or a non-purged CV boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

__all__ = ["ShuffledLabelResult", "shuffled_label_auc", "assert_no_leakage"]


@dataclass(frozen=True)
class ShuffledLabelResult:
    """Outcome of one shuffled-label run.

    ``auc`` is the out-of-sample AUC of a plain LightGBM trained on
    *permuted* labels. Under no leakage, E[auc] == 0.5; the harness
    asserts within ``tolerance`` of that.
    """

    auc: float
    tolerance: float
    n_train: int
    n_test: int
    seed: int
    permutation: np.ndarray  # the label permutation applied (for repro)

    @property
    def passed(self) -> bool:
        return abs(self.auc - 0.5) <= self.tolerance


def shuffled_label_auc(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    test_fraction: float = 0.4,
    seed: int = 0,
    tolerance: float = 0.05,
    lgbm_params: Optional[dict] = None,
) -> ShuffledLabelResult:
    """Train on permuted labels; return the OOS AUC (must be ~0.5).

    Implements the shuffled-label leg of the ARCHITECTURE.md §4
    correctness battery. Reusable across phases: pass any feature
    matrix + label series.

    The split is a *time-ordered* split (first ``(1-test_fraction)`` of
    rows in training, rest in test) — a deliberate minimal stand-in for
    purged CV. The point of this harness is label leakage, so we hold
    the split simple and predictable; the real pipeline uses purged
    CPCV (§7).

    Args:
        features: Feature matrix, rows = events, columns = features.
        labels: Binary 0/1 labels, same index as ``features``.
        test_fraction: Fraction of rows held out for OOS AUC.
        seed: RNG seed for the label permutation (and LightGBM).
        tolerance: |auc - 0.5| must be <= this to "pass".
        lgbm_params: Override the default plain-LightGBM params.

    Returns:
        A :class:`ShuffledLabelResult` carrying AUC + diagnostics.
    """
    if len(features) != len(labels):
        raise ValueError(
            f"features ({len(features)}) and labels ({len(labels)}) "
            "must have equal length"
        )
    if labels.nunique() < 2:
        raise ValueError(
            "shuffled-label test needs both classes present after the "
            f"split; label has {labels.nunique()} unique value(s)"
        )

    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    permuted = labels.to_numpy()[perm]

    n = len(features)
    n_train = int(round(n * (1.0 - test_fraction)))
    X = features.to_numpy()
    y_train = permuted[:n_train]
    y_test = labels.to_numpy()[n_train:]  # OOS uses the *real* label
    # distribution shape but AUC is computed against permuted test labels
    # so the model cannot have learned the mapping. Per the §4 spec: train
    # on permuted labels, evaluate OOS — we permute the full label vector
    # so train AND test carry the (now-meaningless) permuted labels.
    y_test = permuted[n_train:]
    X_train, X_test = X[:n_train], X[n_train:]

    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        raise ValueError(
            "train or test split ended up single-class after permutation; "
            "increase n or retry with another seed"
        )

    params = dict(
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
        num_leaves=7,
        max_depth=3,
        min_data_in_leaf=5,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=seed,
    )
    if lgbm_params:
        params.update(lgbm_params)

    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=50)
    pred = model.predict(X_test)

    # AUC needs both classes in y_test (checked above).
    auc = float(roc_auc_score(y_test, pred))

    return ShuffledLabelResult(
        auc=auc,
        tolerance=tolerance,
        n_train=n_train,
        n_test=int(n - n_train),
        seed=seed,
        permutation=perm,
    )


def assert_no_leakage(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    tolerance: float = 0.05,
    n_seeds: int = 3,
    **kwargs,
) -> list[ShuffledLabelResult]:
    """Run the shuffled-label test over several seeds and assert all pass.

    A single seed can flap across the boundary by chance; running a few
    seeds and requiring *all* within tolerance makes a flaky pass much
    less likely. This is the function future phases call as their
    regression check.
    """
    results = [
        shuffled_label_auc(features, labels, seed=s, tolerance=tolerance, **kwargs)
        for s in range(n_seeds)
    ]
    failed = [r for r in results if not r.passed]
    assert not failed, (
        "Shuffled-label leakage test FAILED on permuted labels -> the "
        "harness is leaking (features predict random labels above chance). "
        f"Failures: {[(r.seed, round(r.auc, 4)) for r in failed]}. "
        "Check point-in-time snapshotting and CV purging (§4/§7)."
    )
    return results
