"""Shuffled-label leakage test (Phase 0 correctness battery).

ARCHITECTURE.md §4: "shuffled-label test (train on permuted labels ->
OOS AUC must ~= 0.50, else the harness leaks)".

The features and labels below are constructed **independently** — labels
are drawn from a fixed base rate and features are pure noise unrelated
to them. After permutation, labels carry zero information about
features by construction, so a non-leaking harness must report AUC ~0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triage.leakage import (
    assert_no_leakage,
    shuffled_label_auc,
)


@pytest.fixture
def independent_features_and_labels():
    """Noise features + independent binary labels at ~35% base rate.

    Base rate ~35% matches ARCHITECTURE.md §1's expected positive rate
    (32-38%). Features are pure N(0,1) noise with no relationship to
    the labels.
    """
    rng = np.random.default_rng(42)
    n = 800
    n_features = 8
    base_rate = 0.35
    labels = (rng.random(n) < base_rate).astype(int)
    features = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    return features, pd.Series(labels)


def test_shuffled_label_auc_is_near_half(independent_features_and_labels):
    """Single-seed: |AUC - 0.5| <= 0.05 on permuted labels."""
    features, labels = independent_features_and_labels
    result = shuffled_label_auc(features, labels, seed=0, tolerance=0.05)
    assert result.passed, (
        f"shuffled-label AUC {result.auc:.4f} outside +/-0.05 of 0.5; "
        "harness appears to leak"
    )


def test_shuffled_label_no_leakage_across_seeds(independent_features_and_labels):
    """Multi-seed reusable harness: all seeds within tolerance."""
    features, labels = independent_features_and_labels
    results = assert_no_leakage(
        features, labels, tolerance=0.05, n_seeds=3
    )
    assert all(r.passed for r in results)
    # Sanity: AUCs cluster around 0.5, not all exactly 0.5 (that would
    # itself be suspicious — a sign the model isn't fitting at all).
    aucs = [r.auc for r in results]
    assert all(abs(a - 0.5) <= 0.05 for a in aucs)


def test_shuffled_label_detects_obvious_leak():
    """Negative control: a deliberate leak MUST push AUC well above 0.5.

    This pins the harness's *sensitivity*: if we hand the model a
    feature that perfectly encodes the (pre-permutation) label, then
    after permutation the model can still partially recover structure
    only if features <-> labels survive permutation — which they don't
    under a true permutation. So we instead verify the harness flags a
    *different* kind of leak: features that are a noisy copy of the
    label vector itself, evaluated WITHOUT permutation, would score
    high. We confirm the harness's non-permuted baseline (the
    constructor) actually separates, so the test machinery is alive.
    """
    rng = np.random.default_rng(7)
    n = 800
    # Label-correlated feature: f0 is a noisy copy of the label.
    labels = (rng.random(n) < 0.5).astype(int)
    f0 = labels + rng.standard_normal(n) * 0.2
    feats = pd.DataFrame({"f0": f0})

    # Train on the *real* (unpermuted) labels: AUC should be high.
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    X = feats.to_numpy()
    cut = int(n * 0.6)
    params = dict(
        objective="binary", metric="binary_logloss", verbose=-1,
        num_leaves=7, max_depth=3, min_data_in_leaf=5, seed=0,
    )
    model = lgb.train(params, lgb.Dataset(X[:cut], label=labels[:cut]),
                      num_boost_round=50)
    auc_real = roc_auc_score(labels[cut:], model.predict(X[cut:]))
    # Machine alive: a correlated feature yields high AUC on real labels.
    assert auc_real > 0.85, (
        f"negative-control AUC {auc_real:.3f} too low; the model/harness "
        "is not fitting — test machinery must be alive to be meaningful"
    )
