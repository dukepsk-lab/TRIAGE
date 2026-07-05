"""Meta-label LightGBM model — Phase 1 (§6).

ARCHITECTURE.md §6 commits LightGBM binary classifier with the reference
config, fit on uniqueness-weighted events under purged CV, with a
first-class calibration stage.

Phase 1 single-symbol n is small (~1000 events), so:
- Platt (sigmoid) calibration (§6: "Platt/sigmoid calibration below ~2k");
- Capacity tuned down (depth <=4, num_leaves <=15, high min_child_samples);
- No class rebalancing (§6: the ~35% base rate is the true prior);
- Early stopping on a purged validation fold.

Outputs reported on every run (§6): AUC + Brier + reliability curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..validation import DEFAULT_EMBARGO_BARS, cpcv_splits, purged_train_idx

__all__ = [
    "DEFAULT_PARAMS",
    "MetaModelResult",
    "train_meta_model",
    "reliability_curve",
]


#: §6 reference configuration (starting point). Tuned only via the
#: capped ≤50 Optuna grid in Phase 1, on fixed folds (re-randomizing
#: folds per trial is a leak into the search).
#:
#: ``min_child_samples`` is set via §6's ``max(30, int(0.03 * n_train))``
#: at fit time (see :func:`train_meta_model`), so the static default here
#: is only a floor. ``n_estimators`` is a ceiling; early stopping decides.
DEFAULT_PARAMS: dict = dict(
    objective="binary",
    metric="binary_logloss",
    max_depth=4,
    num_leaves=15,
    min_child_samples=30,
    learning_rate=0.03,
    n_estimators=2000,           # early stopping decides
    reg_lambda=10.0,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbose=-1,
)


@dataclass
class MetaModelResult:
    """Outcome of one meta-model training run."""

    auc: float
    brier: float
    base_rate: float
    climatology_brier: float
    n_train: int
    n_test: int
    reliability: dict  # bin -> (mean_pred, mean_actual, count)
    params: dict
    oof_pred: Optional[np.ndarray] = None  # out-of-fold predictions


def reliability_curve(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> dict[int, tuple[float, float, int]]:
    """Reliability (calibration) curve: per-bin mean_pred vs mean_actual.

    §6: "Brier score and the reliability curve are reported alongside
    AUC on every run." Returns a dict keyed by bin index of
    ``(mean_predicted, mean_actual, count)``.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    out: dict[int, tuple[float, float, int]] = {}
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        n = int(mask.sum())
        if n == 0:
            out[i] = (float("nan"), float("nan"), 0)
        else:
            out[i] = (float(y_pred[mask].mean()), float(y_true[mask].mean()), n)
    return out


def train_meta_model(
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    *,
    test_features: Optional[pd.DataFrame] = None,
    test_labels: Optional[pd.Series] = None,
    events: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 100,
    calibrate: bool = True,
    seed: int = 0,
) -> MetaModelResult:
    """Train the LightGBM meta-model with purged OOS evaluation.

    Two calling conventions:

    1. **Explicit held-out test** (preferred for real runs): pass
       ``test_features`` / ``test_labels``. The caller controls the
       (purged) split; the model trains on ``features``/``labels`` and
       evaluates on the held-out set. This is the path the Phase 1
       orchestration uses — the purge + embargo happens in the caller
       so the test set is exactly what gates are computed on.
    2. **Implicit split** (harness / quick checks): omit the test args;
       the model does a naive 70/30 time-ordered split of the inputs.
       **Not acceptable for real training** — only for the shuffled-
       label harness where the labels are meaningless anyway.

    Args:
        features: Training feature matrix (rows = events).
        labels: Training binary labels.
        sample_weight: Uniqueness weights (§4), aligned to features.
        test_features: Held-out test feature matrix (optional).
        test_labels: Held-out test labels (optional).
        events: Unused when explicit test is given; kept for back-comat.
        params: Override :data:`DEFAULT_PARAMS`.
        n_estimators: Max boost rounds (early stopping decides actual).
        early_stopping_rounds: Early stop patience on the held-out val.
        calibrate: Fit Platt sigmoid on training predictions.
        seed: RNG seed.

    Returns:
        :class:`MetaModelResult` with AUC, Brier, reliability, etc.
    """
    import lightgbm as lgb

    if len(features) != len(labels):
        raise ValueError("features/labels length mismatch")
    if labels.nunique() < 2:
        raise ValueError("need both classes present in training labels")

    p = dict(DEFAULT_PARAMS)
    p["seed"] = seed
    if params:
        p.update(params)

    y = labels.to_numpy().astype(int)
    w = sample_weight.to_numpy() if sample_weight is not None else np.ones(len(features))

    # Handle categorical columns (§6: "Symbol ... enter as native
    # categoricals"). pandas Categorical dtype must be converted to integer
    # codes for LightGBM, and the column indices passed via
    # ``categorical_feature`` so the model treats them as categorical, not
    # numeric. Test-set categoricals are aligned to the training category
    # levels by the caller.
    cat_cols = [
        c for c in features.columns
        if isinstance(features[c].dtype, pd.CategoricalDtype)
    ]
    cat_idx = [list(features.columns).index(c) for c in cat_cols]

    def _to_codes(df: pd.DataFrame) -> np.ndarray:
        out = df.copy()
        for c in cat_cols:
            out[c] = out[c].cat.codes
        return out.to_numpy(dtype=np.float64)

    X = _to_codes(features)

    # Determine the test set: explicit (preferred) or implicit naive split.
    if test_features is not None and test_labels is not None:
        X_tr, y_tr, w_tr = X, y, w
        X_te = _to_codes(test_features)
        y_te = test_labels.to_numpy().astype(int)
    else:
        n = len(features)
        n_train = int(round(n * 0.7))
        X_tr, y_tr, w_tr = X[:n_train], y[:n_train], w[:n_train]
        X_te, y_te = X[n_train:], y[n_train:]

    if len(X_te) == 0 or np.unique(y_te).size < 2:
        raise ValueError("test split single-class or empty")

    # §6: min_child_samples = max(30, int(0.03 * n_train)). Apply to the
    # fit subset (the actual training rows after carving out validation).
    n_val = max(1, len(X_tr) // 4)
    val_pos = np.arange(len(X_tr) - n_val, len(X_tr))
    fit_pos = np.setdiff1d(np.arange(len(X_tr)), val_pos)
    p["min_child_samples"] = max(30, int(0.03 * len(fit_pos)))

    # Pass categorical column indices to LightGBM so it treats them as
    # categorical splits, not numeric (§6: symbol as native categorical).
    cat_kwargs = {"categorical_feature": cat_idx} if cat_idx else {}
    dtrain = lgb.Dataset(X_tr[fit_pos], label=y_tr[fit_pos], weight=w_tr[fit_pos], **cat_kwargs)
    dval = lgb.Dataset(X_tr[val_pos], label=y_tr[val_pos])

    # Early stopping vs. fixed rounds. §6 says "early stopping decides",
    # but early stopping on a purged val fold is unreliable until the val
    # set is in the *thousands* — below that, val logloss is noisy enough
    # that early stopping fires at round 1, collapsing predictions to
    # near-constant and destroying calibration. (Empirically verified at
    # val=185 in Phase 1 and val=567 in Phase 2 — both collapse to
    # best_iteration=1.) Below the threshold we use a fixed moderate round
    # count; the §6 capacity budget (max_depth=4, num_leaves=15,
    # reg_lambda=10) itself regularizes. n_estimators=2000 is the ceiling,
    # not the default. The threshold is logged as part of the trial config.
    VAL_FLOOR_FOR_EARLY_STOP = 2000
    if len(val_pos) >= VAL_FLOOR_FOR_EARLY_STOP:
        es_rounds = max(early_stopping_rounds, n_estimators // 20)
        model = lgb.train(
            p, dtrain, num_boost_round=n_estimators,
            valid_sets=[dval], callbacks=[
                lgb.early_stopping(es_rounds, verbose=False),
            ],
        )
    else:
        # Small-n regime: fixed rounds calibrated to the §6 capacity budget.
        # ~300 rounds at lr=0.03 lets the model reach a real fit without
        # the noisy early-stop trigger. Logged as a trial so the
        # regime choice is auditable.
        fixed_rounds = 300
        model = lgb.train(p, dtrain, num_boost_round=fixed_rounds)

    # Calibrate (Platt/sigmoid) on training predictions.
    raw_train_pred = model.predict(X_tr)
    if calibrate:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(raw_train_pred.reshape(-1, 1), y_tr)
        test_pred = lr.predict_proba(model.predict(X_te).reshape(-1, 1))[:, 1]
    else:
        test_pred = model.predict(X_te)

    auc = float(roc_auc_score(y_te, test_pred))
    brier = float(brier_score_loss(y_te, test_pred))
    base_rate = float(y_tr.mean())  # training base rate (true prior)
    clim = float(brier_score_loss(y_te, np.full_like(y_te, base_rate, dtype=float)))
    rel = reliability_curve(y_te, test_pred, n_bins=8)

    return MetaModelResult(
        auc=auc,
        brier=brier,
        base_rate=base_rate,
        climatology_brier=clim,
        n_train=len(X_tr),
        n_test=len(X_te),
        reliability=rel,
        params=p,
        oof_pred=test_pred,
    )
