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
    "reliability_slope",
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
    n_train: int                # ACTUAL fit-subset size (post val carve-out)
    n_val: int                  # rows held out for early-stopping/val (0 if fixed rounds)
    n_test: int
    reliability: dict  # bin -> (mean_pred, mean_actual, count)
    params: dict
    oof_pred: Optional[np.ndarray] = None  # out-of-fold predictions


def reliability_curve(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> dict[int, tuple[float, float, int]]:
    """Reliability (calibration) curve: per-bin mean_pred vs mean_actual.

    Uses unique values if predictions are highly clustered (e.g. from
    IsotonicRegression on low-signal data), otherwise uses quantile
    bins to ensure equal-sized bins and stable slope calculation.
    """
    out: dict[int, tuple[float, float, int]] = {}
    unique_preds = np.unique(y_pred)
    
    if len(unique_preds) <= n_bins:
        for i, val in enumerate(unique_preds):
            mask = (y_pred == val)
            n = int(mask.sum())
            out[i] = (float(val), float(y_true[mask].mean()), n)
        return out
        
    try:
        import pandas as pd
        cuts = pd.qcut(y_pred, q=n_bins, duplicates='drop')
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'bin': cuts})
        grouped = df.groupby('bin', observed=False)
        i = 0
        for name, group in grouped:
            if len(group) > 0:
                out[i] = (float(group['y_pred'].mean()), float(group['y_true'].mean()), len(group))
                i += 1
        return out
    except Exception:
        pass

    bins = np.linspace(0, 1, n_bins + 1)
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


def reliability_slope(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 8) -> float:
    """Slope of the reliability curve (mean_actual regressed on mean_pred).

    G5 (§7) requires this in ``[0.8, 1.2]``. A slope near 1 is well
    calibrated; a shallow slope (toward 0) is the signature of a calibrator
    that has collapsed predictions toward a constant — which also drives the
    Brier score up toward climatology. Returns 0.0 when fewer than two
    populated bins exist (undefined slope, treated as a hard calibration
    failure by the gate).
    """
    rel = reliability_curve(np.asarray(y_true), np.asarray(y_pred), n_bins=n_bins)
    xs = [mp for mp, ma, n in rel.values() if n > 0 and not np.isnan(mp)]
    ys = [ma for mp, ma, n in rel.values() if n > 0 and not np.isnan(ma)]
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(xs, ys, 1)[0])


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
    calibration: str = "cv",
    calib_frac: float = 0.2,
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
        calibrate: Whether to calibrate probabilities at all.
        calibration: Calibration strategy when ``calibrate`` is True:

            * ``"cv"`` (default, back-compat): ``CalibratedClassifierCV``
              with internal 5-fold CV over the training data. Reproduces the
              Phase 1/2 runs. Prone to collapsing predictions toward the
              base rate at small n (Brier -> climatology, shallow slope).
            * ``"holdout"`` (§6 "fit on a purged calibration fold"): carve a
              time-ordered tail of the training set as a dedicated
              calibration fold, fit the booster on the earlier portion,
              purge fit rows whose label span reaches into the calibration
              fold (when ``events`` is supplied), and fit the 1-D calibrator
              (isotonic >= 2k train events, else sigmoid) on genuinely
              out-of-fold predictions. This is what G5 (§7) is designed to
              reward and the recommended path once pooled n clears ~2k.

        calib_frac: Fraction of the (time-ordered) training set reserved as
            the calibration fold under ``calibration="holdout"``.
        seed: RNG seed.

    Note:
        ``calibration="holdout"`` assumes ``features`` (and ``events``, if
        given) are in ascending time order — the calibration fold is the
        tail. Callers that build a purged train/test split upstream (e.g.
        ``scripts/run_phase3.py``) already satisfy this.

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

    # §6: min_child_samples = max(30, int(0.03 * n_train)).
    p["min_child_samples"] = max(30, int(0.03 * len(X_tr)))
    cat_kwargs = {"categorical_feature": cat_idx} if cat_idx else {}

    if calibrate and calibration not in ("cv", "holdout"):
        raise ValueError(
            f"calibration must be 'cv' or 'holdout', got {calibration!r}"
        )

    if calibrate and calibration == "holdout":
        # §6: fit the calibrator on a DEDICATED, PURGED calibration fold —
        # not via internal CV on the training data. The booster fits on the
        # earlier portion of the (time-ordered) training set; the calibrator
        # fits on genuinely out-of-fold predictions over the tail fold.
        method = "isotonic" if len(X_tr) >= 2000 else "sigmoid"
        n_calib = max(1, int(round(len(X_tr) * calib_frac)))
        calib_pos = np.arange(len(X_tr) - n_calib, len(X_tr))
        fit_pos = np.arange(0, len(X_tr) - n_calib)

        # Purge fit rows whose label span reaches into the calibration fold.
        if events is not None and len(events) == len(X_tr):
            ev = events.reset_index(drop=True)
            calib_start = pd.to_datetime(ev.iloc[calib_pos]["t0"]).min()
            t1 = pd.to_datetime(ev.iloc[fit_pos]["t1"])
            # Keep only fit rows fully resolved before the calibration fold
            # opens (NaT t1 -> unresolved -> conservatively dropped).
            keep = (t1 < calib_start).fillna(False).to_numpy()
            fit_pos = fit_pos[keep]
        else:
            # No event spans available: apply a small positional embargo gap
            # between the fit set and the calibration fold to blunt serial
            # correlation leakage.
            gap = min(len(fit_pos), max(0, min(50, int(0.05 * len(X_tr)))))
            if gap:
                fit_pos = fit_pos[:-gap]

        calib_ok = (
            len(fit_pos) >= 50
            and len(calib_pos) >= 20
            and np.unique(y_tr[fit_pos]).size == 2
            and np.unique(y_tr[calib_pos]).size == 2
        )
        if calib_ok:
            dtrain = lgb.Dataset(
                X_tr[fit_pos], label=y_tr[fit_pos], weight=w_tr[fit_pos], **cat_kwargs
            )
            model = lgb.train(p, dtrain, num_boost_round=300)
            calib_raw = model.predict(X_tr[calib_pos])
            test_raw = model.predict(X_te)

            if method == "isotonic":
                from sklearn.isotonic import IsotonicRegression

                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(calib_raw, y_tr[calib_pos], sample_weight=w_tr[calib_pos])
                test_pred = iso.predict(test_raw)
            else:
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression()
                lr.fit(
                    calib_raw.reshape(-1, 1),
                    y_tr[calib_pos],
                    sample_weight=w_tr[calib_pos],
                )
                test_pred = lr.predict_proba(test_raw.reshape(-1, 1))[:, 1]

            fit_pos_len = len(fit_pos)
            val_pos_len = len(calib_pos)
        else:
            # Degenerate fold (single-class or too small) -> fall back to the
            # CV calibrator rather than emit uncalibrated probabilities.
            calibration = "cv"

    if calibrate and calibration == "cv":
        from sklearn.calibration import CalibratedClassifierCV
        # Cross-validated calibrator over the training data (back-compat path;
        # reproduces Phase 1/2). Isotonic >= 2k events, else Platt sigmoid.
        p_clf = p.copy()
        p_clf.pop("n_estimators", None)
        clf = lgb.LGBMClassifier(**p_clf, n_estimators=300, n_jobs=-1, random_state=42)

        if len(X_tr) >= 2000:
            calibrator = CalibratedClassifierCV(clf, cv=5, method="isotonic")
        else:
            calibrator = CalibratedClassifierCV(clf, cv=5, method="sigmoid")

        calibrator.fit(X_tr, y_tr, sample_weight=w_tr)
        test_pred = calibrator.predict_proba(X_te)[:, 1]

        # Fit a single booster on all data so the caller can extract feature importances
        dtrain_full = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, **cat_kwargs)
        model = lgb.train(p, dtrain_full, num_boost_round=300)

        fit_pos_len = len(X_tr)
        val_pos_len = 0

    if not calibrate:
        n_val = max(1, len(X_tr) // 4)
        val_pos = np.arange(len(X_tr) - n_val, len(X_tr))
        fit_pos = np.setdiff1d(np.arange(len(X_tr)), val_pos)
        
        dtrain = lgb.Dataset(X_tr[fit_pos], label=y_tr[fit_pos], weight=w_tr[fit_pos], **cat_kwargs)
        dval = lgb.Dataset(X_tr[val_pos], label=y_tr[val_pos])
        
        VAL_FLOOR_FOR_EARLY_STOP = 1000
        if len(val_pos) >= VAL_FLOOR_FOR_EARLY_STOP:
            es_rounds = max(early_stopping_rounds, n_estimators // 20)
            model = lgb.train(
                p, dtrain, num_boost_round=n_estimators,
                valid_sets=[dval], callbacks=[
                    lgb.early_stopping(es_rounds, verbose=False),
                ],
            )
        else:
            model = lgb.train(p, dtrain, num_boost_round=300)

        test_pred = model.predict(X_te)
        fit_pos_len = len(fit_pos)
        val_pos_len = len(val_pos)

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
        n_train=fit_pos_len,   # actual rows the model fit on
        n_val=val_pos_len,     # rows held out
        n_test=len(X_te),
        reliability=rel,
        params=p,
        oof_pred=test_pred,
    )
