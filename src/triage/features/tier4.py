"""Tier 4 cross-sectional features + MDA-based selection (§5, Phase 3).

ARCHITECTURE.md §5: "Tier 4 cross-sectional features enter at Phase 3
only, gated by MDA evidence." And the selection method: "correlation-
prune first (drop one of any pair with |ρ| > 0.85 — this addresses the
substitution effect directly), then MDA under purged CV: permute each
surviving feature within OOS folds, measure the drop in strategy-
relevant metric (mean OOS trade EV, not raw AUC). Features whose MDA
confidence interval spans zero after two phases get cut. MDI/gain
importances are diagnostics only, never selection criteria."

This module implements:
1. :func:`correlation_prune` — drop one of any |ρ|>0.85 pair.
2. :func:`mda_selection` — Mean-Decreased-AUC/EV under purged CV; keep
   features whose permutation drop is significantly > 0.
3. Tier 4 cross-sectional feature columns (computed at the basket level,
   not per-symbol — they capture relative positioning across instruments).

Tier 4 features (cross-sectional, by definition computed across the
pooled basket at each timestamp):
  - ``cs_atr_pct_rank``  this event's ATR% rank within all same-bar events
  - ``cs_rsi_rank``      RSI rank across the basket at this timestamp
  - ``cs_roc_rank``      6-bar ROC rank across the basket

These are the only Phase 3 additions; they enter the feature matrix
*only if* MDA clears them.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "TIER4_COLUMNS",
    "CORRELATION_THRESHOLD",
    "correlation_prune",
    "mda_selection",
    "attach_tier4_cross_sectional",
]


#: Tier 4 cross-sectional feature columns. Added to the feature matrix
#: only if MDA clears them (§5).
TIER4_COLUMNS: list[str] = [
    "cs_atr_pct_rank",
    "cs_rsi_rank",
    "cs_roc_rank",
]

#: §5 correlation-prune threshold: "drop one of any pair with |ρ| > 0.85".
CORRELATION_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# Cross-sectional feature construction.
# ---------------------------------------------------------------------------


def attach_tier4_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Add Tier 4 cross-sectional rank features to the pooled event frame.

    Cross-sectional = ranking each event's per-symbol feature *against
    all other symbols' events at the same H4 bar close*. This captures
    relative positioning (e.g. "EURUSD is the most volatile of the 3
    majors right now").

    ``df`` must have ``t0``, ``symbol``, and the base features
    (``atr_pct``, ``rsi_14``, ``roc_6``). Events are grouped by t0 and
    ranked within each timestamp; ranks normalized to [0,1].

    For timestamps with only one symbol present (sparse periods), the
    rank is 0.5 (neutral) — there's no cross-section to rank against.
    """
    out = df.copy()
    base = {"cs_atr_pct_rank": "atr_pct", "cs_rsi_rank": "rsi_14", "cs_roc_rank": "roc_6"}
    for cs_col, src in base.items():
        if src not in out.columns:
            out[cs_col] = 0.5
            continue
        # Rank within each t0 group, normalized to [0,1].
        # rank(pct=True) gives the percentile rank; for single-row groups
        # it returns 1.0, so we override those to 0.5 (no cross-section).
        grp_sizes = out.groupby("t0")[src].transform("size")
        ranks = out.groupby("t0")[src].rank(pct=True)
        out[cs_col] = np.where(grp_sizes > 1, ranks, 0.5)
    return out


# ---------------------------------------------------------------------------
# §5 selection: correlation prune, then MDA under purged CV.
# ---------------------------------------------------------------------------


def correlation_prune(
    X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD
) -> list[str]:
    """Return the column subset after dropping one of any |ρ|>threshold pair.

    §5: "drop one of any pair with |ρ| > 0.85 — this addresses the
    substitution effect directly." When a pair collides, the *second*
    column (by input order) is dropped. Returns the kept column list.
    """
    if X.shape[1] < 2:
        return list(X.columns)
    corr = X.corr().abs()
    kept = list(X.columns)
    # Walk the upper triangle; drop the later-indexed column of any clash.
    cols = list(X.columns)
    dropped: set[str] = set()
    for i in range(len(cols)):
        if cols[i] in dropped:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in dropped:
                continue
            c = corr.iloc[i, j]
            if pd.notna(c) and c > threshold:
                dropped.add(cols[j])
    return [c for c in kept if c not in dropped]


def mda_selection(
    X: pd.DataFrame,
    y: pd.Series,
    events: pd.DataFrame,
    *,
    n_splits: int = 5,
    metric: str = "auc",
    seed: int = 0,
    model_params: dict | None = None,
) -> pd.DataFrame:
    """Mean-Decreased-AUC (or EV) under purged K-fold CV (§5 MDA).

    For each feature column, permute it *within each OOS fold* and
    measure the drop in the chosen metric vs. the unpermuted baseline.
    A feature whose MDA confidence interval spans zero is cut.

    This is the §5 selection method — explicitly NOT MDI/gain importance
    (which the doc says are "diagnostics only, never selection criteria").

    Args:
        X: Feature matrix (post-correlation-prune).
        y: Binary labels.
        events: DataFrame with ``t0``/``t1`` for purging.
        n_splits: Number of purged time-ordered CV folds.
        metric: ``"auc"`` (default) or ``"ev"`` (mean trade EV).
        seed: RNG seed for permutation.
        model_params: LightGBM params override.

    Returns:
        DataFrame with columns ``feature, baseline, permuted, mda_drop,
        kept`` — sorted by MDA drop descending. ``kept`` is True when the
        drop's sign is consistently positive across folds (CI doesn't
        span zero).
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from ..validation import purged_train_idx, DEFAULT_EMBARGO_BARS

    if metric not in ("auc", "ev"):
        raise ValueError(f"metric must be 'auc' or 'ev', got {metric!r}")

    # §5 "mean OOS trade EV" needs realized returns, not predicted
    # probabilities. Pull r_realized (+ per-symbol cost) straight off the
    # aligned ``events`` frame. The historical bug was that the "ev" branch
    # returned mean predicted probability — a quantity that has nothing to do
    # with trade profitability and would "keep" whatever feature most shifts
    # p-hat, regardless of whether that shift is toward money.
    ev_r_all = None
    ev_cost_all = None
    tau0_ev = None
    if metric == "ev":
        from ..gates import cost_for_symbol, MAJOR_COST_R
        from ..sizing import default_tau0

        if "r_realized" not in events.columns:
            raise ValueError(
                "metric='ev' requires an 'r_realized' column on events "
                "(realized R-multiple per event) to compute trade EV (§5)."
            )
        ev_r_all = pd.to_numeric(events["r_realized"], errors="coerce").fillna(0.0).to_numpy()
        if "symbol" in events.columns:
            ev_cost_all = np.array([cost_for_symbol(str(s)) for s in events["symbol"]])
        else:
            ev_cost_all = np.full(len(events), MAJOR_COST_R)
        tau0_ev = default_tau0(float(np.median(ev_cost_all)))

    def _captured_ev(pred: np.ndarray, r_te: np.ndarray, cost_te: np.ndarray) -> float:
        """Mean captured trade EV: (r - cost) over taken trades, else 0.

        Taken = calibrated-conviction filter at tau0 (the strategy's own
        breakeven+margin threshold, §8). If tau0 selects too few in a fold
        to be informative, fall back to the top-40%-by-conviction slice so
        the fold still discriminates the feature's ranking contribution.
        """
        taken = pred >= tau0_ev
        if int(taken.sum()) < 5:
            thr = np.quantile(pred, 0.6)
            taken = pred >= thr
        return float(np.mean(np.where(taken, r_te - cost_te, 0.0)))

    rng = np.random.default_rng(seed)
    df_idx = np.arange(len(X))
    # Time-ordered fold boundaries.
    fold_size = len(X) // n_splits
    folds = [(i * fold_size, (i + 1) * fold_size) for i in range(n_splits)]
    folds[-1] = (folds[-1][0], len(X))

    p = dict(
        objective="binary", metric="binary_logloss", max_depth=4,
        num_leaves=15, learning_rate=0.05, verbose=-1, seed=seed,
        min_child_samples=30,
    )
    if model_params:
        p.update(model_params)

    features = list(X.columns)
    # mda_drops[feature] = list of per-fold (baseline - permuted) deltas.
    mda_drops: dict[str, list[float]] = {f: [] for f in features}

    for fi, (lo, hi) in enumerate(folds):
        test_pos = np.arange(lo, hi)
        train_pos = np.setdiff1d(df_idx, test_pos)
        # Purge training events overlapping the test span.
        ev = events.iloc[train_pos][["t0", "t1"]].copy()
        ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
        test_start = pd.Timestamp(events.iloc[lo]["t0"])
        test_end = pd.Timestamp(events.iloc[hi - 1]["t1"]) if pd.notna(events.iloc[hi - 1]["t1"]) else test_start + pd.Timedelta(hours=96)
        keep_idx = purged_train_idx(
            ev.set_index(np.arange(len(ev)))[["t0", "t1"]],
            test_start, test_end, embargo_bars=DEFAULT_EMBARGO_BARS,
        )
        keep_pos = train_pos[np.array([i for i in keep_idx], dtype=int)]
        if len(keep_pos) < 50:
            keep_pos = train_pos  # fallback if purge too aggressive

        Xn = X.to_numpy()
        yn = y.to_numpy().astype(int)
        X_tr, y_tr = Xn[keep_pos], yn[keep_pos]
        X_te, y_te = Xn[test_pos], yn[test_pos]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        model = lgb.train(p, lgb.Dataset(X_tr, label=y_tr), num_boost_round=100)
        base_pred = model.predict(X_te)
        if metric == "auc":
            baseline = roc_auc_score(y_te, base_pred)
            r_te = cost_te = None
        else:
            r_te = ev_r_all[test_pos]
            cost_te = ev_cost_all[test_pos]
            baseline = _captured_ev(base_pred, r_te, cost_te)

        for fi_idx, feat in enumerate(features):
            X_te_perm = X_te.copy()
            perm = rng.permutation(len(X_te_perm))
            X_te_perm[:, fi_idx] = X_te_perm[perm, fi_idx]
            pred_p = model.predict(X_te_perm)
            if metric == "auc":
                val_p = roc_auc_score(y_te, pred_p)
            else:
                val_p = _captured_ev(pred_p, r_te, cost_te)
            mda_drops[feat].append(baseline - val_p)  # positive = feature helped

    # Summarize: mean drop + CI, keep if CI doesn't span zero.
    rows = []
    for feat in features:
        drops = np.array(mda_drops[feat])
        if len(drops) == 0:
            rows.append({"feature": feat, "mda_mean": 0.0, "mda_std": 0.0,
                         "ci_low": 0.0, "ci_high": 0.0, "kept": False})
            continue
        mean = float(drops.mean())
        std = float(drops.std()) if len(drops) > 1 else 0.0
        # 95% CI (normal approx).
        ci_low = mean - 1.96 * std / max(np.sqrt(len(drops)), 1)
        ci_high = mean + 1.96 * std / max(np.sqrt(len(drops)), 1)
        # §5: "Features whose MDA confidence interval spans zero get cut."
        kept = ci_low > 0  # CI entirely above zero -> feature helps
        rows.append({"feature": feat, "mda_mean": mean, "mda_std": std,
                     "ci_low": ci_low, "ci_high": ci_high, "kept": bool(kept)})
    out = pd.DataFrame(rows).sort_values("mda_mean", ascending=False).reset_index(drop=True)
    return out
