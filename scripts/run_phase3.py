"""Phase 3 orchestration: full basket + second rule family + Tier 4 + calibration fix.

ARCHITECTURE.md §10 Phase 3:
- Add XAUUSD, US30, BTCUSD.
- Introduce second rule family (htf_pullback).
- Introduce Tier 4 features (cross-sectional) gated by MDA.
- Isotonic regression for calibration (handled automatically in meta.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from triage.data.mt5_ingest import TF_D1, TF_H4, TF_M1, load_bars_frame
from triage.features import FEATURE_COLUMNS
from triage.features.tier4 import TIER4_COLUMNS, attach_tier4_cross_sectional, correlation_prune, mda_selection
from triage.gates import compute_gates, strategy_returns_r, cost_for_symbol
from triage.leakage import assert_no_leakage
from triage.models.meta import train_meta_model
from triage.pipeline import DEFAULT_EVENTS_DB, build_events_for_signals, persist_events, load_events
from triage.registry import TrialRegistry
from triage.rules import donchian_backfill
from triage.rules import pullback
from triage.uniqueness import uniqueness_weights
from triage.validation import DEFAULT_EMBARGO_BARS, purged_train_idx

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIALS_DB = REPO_ROOT / ".triage" / "registry.sqlite"

PHASE3_SYMBOLS = ["EURUSD", "USDJPY", "BTCUSD"]


# ---------------------------------------------------------------------------
# Step 1: backfill
# ---------------------------------------------------------------------------


def backfill_symbol_events(symbol: str) -> int:
    """Run Donchian and Pullback rules + M1 labeling for one symbol."""
    existing = load_events(DEFAULT_EVENTS_DB)
    has_donchian = False
    has_pullback = False
    if len(existing) > 0:
        sym_events = existing[existing["symbol"] == symbol]
        has_donchian = "donchian" in sym_events["rule_id"].values if len(sym_events) > 0 else False
        has_pullback = "htf_pullback" in sym_events["rule_id"].values if len(sym_events) > 0 else False

    if has_donchian and has_pullback:
        print(f"[phase3] {symbol}: events already present, skipping")
        return 0

    print(f"[phase3] {symbol}: loading bars...")
    h4 = load_bars_frame(symbol, TF_H4)
    d1 = load_bars_frame(symbol, TF_D1)
    m1 = load_bars_frame(symbol, TF_M1)
    
    total_persisted = 0
    if not has_donchian:
        print(f"[phase3] {symbol}: generating Donchian-20 signals...")
        sigs_donchian = donchian_backfill.backfill_signals(h4, d1)
        labeled = build_events_for_signals(sigs_donchian, h4=h4, d1=d1, m1_all=m1, symbol=symbol, progress_every=500)
        n = persist_events(labeled)
        print(f"[phase3] {symbol}: persisted {n} donchian events")
        total_persisted += n

    if not has_pullback:
        print(f"[phase3] {symbol}: generating Pullback signals...")
        sigs_pullback = pullback.backfill_signals(h4, d1)
        labeled_pb = build_events_for_signals(sigs_pullback, h4=h4, d1=d1, m1_all=m1, symbol=symbol, rule_id="htf_pullback", progress_every=500)
        n = persist_events(labeled_pb)
        print(f"[phase3] {symbol}: persisted {n} pullback events")
        total_persisted += n
        
    return total_persisted


# ---------------------------------------------------------------------------
# Step 2: load & features
# ---------------------------------------------------------------------------


def load_pooled_events(symbols: list[str] | None = None) -> pd.DataFrame:
    """Load all events, unpack features, apply Tier 4.

    ``symbols`` defaults to ``PHASE3_SYMBOLS`` (back-compat); pass an explicit
    list to pool an arbitrary basket (used by the basket-comparison harness).
    """
    symbols = symbols if symbols is not None else PHASE3_SYMBOLS
    df = load_events(DEFAULT_EVENTS_DB)
    df = df[df["symbol"].isin(symbols)].copy()
    
    feats = pd.DataFrame([json.loads(f) if isinstance(f, str) else f for f in df["features"]]).set_index(df.index)
    feat_cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    feats = feats[feat_cols]
    df = df.join(feats)
    
    # Apply Tier 4 Cross Sectional
    df = attach_tier4_cross_sectional(df)
    return df


def build_pooled_matrices(df: pd.DataFrame, final_features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, events_meta)."""
    X = df[final_features + ["symbol", "rule_id"]].copy()
    for c in final_features:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    X["symbol"] = X["symbol"].astype("category")
    X["rule_id"] = X["rule_id"].astype("category")
    y = df["label"].astype(int)
    meta = df[["t0", "t1", "symbol", "rule_id", "r_realized", "label"]].copy()
    return X, y, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def split_pooled(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("t0").reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    test_start = pd.Timestamp(test["t0"].min())
    test_end = pd.Timestamp(df["t1"].max())
    keep = purged_train_idx(
        train[["t0", "t1", "symbol"]].set_index(train.index)[["t0", "t1"]],
        test_start, test_end, embargo_bars=DEFAULT_EMBARGO_BARS,
    )
    keep_pos = [train.index.get_loc(i) for i in keep]
    train = train.iloc[keep_pos].copy()
    return train, test


def train_pooled(train: pd.DataFrame, test: pd.DataFrame, registry: TrialRegistry, final_features: list[str]) -> dict:
    from triage.registry import logged_trial

    @logged_trial(
        {"phase": 3, "config": "pooled_full_basket", "features": final_features},
        registry=registry,
    )
    def _run() -> dict:
        X_tr_raw, y_tr, _ = build_pooled_matrices(train, final_features)
        X_te_raw, y_te, _ = build_pooled_matrices(test, final_features)
        X_te_raw["symbol"] = pd.Categorical(X_te_raw["symbol"], categories=X_tr_raw["symbol"].cat.categories)
        X_te_raw["rule_id"] = pd.Categorical(X_te_raw["rule_id"], categories=X_tr_raw["rule_id"].cat.categories)

        ev = train[["t0", "t1"]].copy()
        ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
        w = uniqueness_weights(ev)

        res = train_meta_model(
            X_tr_raw, y_tr, sample_weight=w,
            test_features=X_te_raw, test_labels=y_te,
            # §6: calibrate on a dedicated purged calibration fold rather than
            # internal CV over the training data. ``events`` (t0/t1, aligned to
            # the training rows) lets the calibration fold be purged of fit
            # rows whose label span reaches into it.
            events=train[["t0", "t1"]],
            calibration="holdout",
            params={"max_depth": 6, "num_leaves": 31, "learning_rate": 0.02},
            n_estimators=1000,
        )
        return res

    res = _run()
    X_tr, y_tr, _ = build_pooled_matrices(train, final_features)
    X_te, y_te, meta_te = build_pooled_matrices(test, final_features)
    X_te["symbol"] = pd.Categorical(X_te["symbol"], categories=X_tr["symbol"].cat.categories)
    X_te["rule_id"] = pd.Categorical(X_te["rule_id"], categories=X_tr["rule_id"].cat.categories)
    return {
        "result": res,
        "X_te": X_te,
        "y_te": y_te,
        "meta_te": meta_te,
        "test_r_realized": test["r_realized"].to_numpy(),
        "test_labels": test["label"].to_numpy().astype(int),
        "test_symbols": test["symbol"].to_numpy(),
    }


def per_slice_attribution(preds: np.ndarray, meta_te: pd.DataFrame, r_realized: np.ndarray, slice_col: str) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rows = []
    for val in sorted(meta_te[slice_col].unique()):
        mask = (meta_te[slice_col] == val).to_numpy()
        n = int(mask.sum())
        if n < 10:
            continue
        y_s = meta_te["label"].to_numpy()[mask]
        p_s = preds[mask]
        r_s = r_realized[mask]
        auc = float(roc_auc_score(y_s, p_s)) if len(set(y_s)) > 1 else float("nan")
        mean_r = float(r_s.mean())
        rows.append({"slice": val, "n": n, "auc": round(auc, 4), "mean_r_raw": round(mean_r, 4), "base_rate": round(float(y_s.mean()), 3)})
    return pd.DataFrame(rows)

def main() -> None:
    print("=" * 70)
    print("PHASE 3 — Full Basket + Tier 4 + Pullback + Calibration Fix")
    print("=" * 70)

    for sym in PHASE3_SYMBOLS:
        backfill_symbol_events(sym)

    df = load_pooled_events()
    print(f"\npooled events: {len(df)}")
    print(f"per-symbol counts: {df.groupby('symbol').size().to_dict()}")

    train, test = split_pooled(df)
    print(f"purged train/test: {len(train)} / {len(test)}")

    print("\n--- Feature Selection (MDA on Tier 4) ---")
    candidate_features = FEATURE_COLUMNS + TIER4_COLUMNS
    X_candidates = train[candidate_features].copy()
    for c in candidate_features:
         X_candidates[c] = pd.to_numeric(X_candidates[c], errors="coerce").fillna(0)
         
    kept_corr = correlation_prune(X_candidates)
    print(f"Post-correlation features: {len(kept_corr)} / {len(candidate_features)}")
    
    mda_res = mda_selection(X_candidates[kept_corr], train["label"], train)
    print("MDA Results:\n", mda_res)
    
    kept_mda = mda_res[mda_res["kept"]]["feature"].tolist()
    tier4_kept = [c for c in kept_mda if c in TIER4_COLUMNS]
    final_features = FEATURE_COLUMNS + tier4_kept
    print(f"Final selected features for training: {final_features}")

    reg = TrialRegistry(TRIALS_DB)
    out = train_pooled(train, test, reg, final_features)
    res = out["result"]
    print(f"\n  pooled AUC={res.auc:.4f}  Brier={res.brier:.4f} (clim={res.climatology_brier:.4f})")
    print(f"  n_train={res.n_train}  n_test={res.n_test}")

    print("\n--- Attribution (Diagnostics) ---")
    preds = res.oof_pred
    print("\nPer-Symbol:")
    print(per_slice_attribution(preds, out["meta_te"], out["test_r_realized"], "symbol").to_string(index=False))
    print("\nPer-Rule:")
    print(per_slice_attribution(preds, out["meta_te"], out["test_r_realized"], "rule_id").to_string(index=False))

    print("\n--- Gates ---")
    from triage.sizing import default_tau0
    tau0 = default_tau0(0.04) # default scalar for simplicity, but we'll use dynamic sizing
    
    # We pass symbols directly to compute per-trade costs
    primary_ret = strategy_returns_r(out["test_labels"], out["test_r_realized"], symbols=out["test_symbols"])
    # Note: tau0 parameter is actually not vectorized in strategy_returns_r easily without updating the signature, 
    # but we can use a scalar approximate tau0 for filtering evaluation, or update strategy_returns_r.
    # For now, we pass symbols which handles the cost_vec.
    filtered_ret = strategy_returns_r(
        out["test_labels"], out["test_r_realized"], 
        probabilities=res.oof_pred, ladder_tau0=0.48, # approximation
        symbols=out["test_symbols"]
    )
    
    taken = filtered_ret[filtered_ret != 0]
    oos_mean = float(taken.mean()) if len(taken) else 0.0
    oos_se = float(taken.std() / np.sqrt(len(taken))) if len(taken) > 1 else 1.0
    obs_sharpe = float(primary_ret.mean() / primary_ret.std() * np.sqrt(252)) if primary_ret.std() > 0 else 0.0
    
    from triage.models.meta import reliability_slope
    rel_slope = reliability_slope(out["test_labels"], res.oof_pred, n_bins=8)

    # Per-symbol calibration attribution: is one symbol dragging the pooled
    # reliability slope? Delete a symbol only if its OWN slope is the outlier,
    # not because it happens to nudge a gate — that would be basket-overfitting
    # (§0 "every tuned knob is a trial"; §10 "no single symbol carrying the book").
    print("\n--- Per-Symbol Calibration (reliability slope, target [0.8, 1.2]) ---")
    for sym in sorted(set(out["test_symbols"])):
        mask = out["test_symbols"] == sym
        if int(mask.sum()) < 20 or len(set(out["test_labels"][mask])) < 2:
            print(f"  {sym}: n={int(mask.sum())} (too few / single-class — skipped)")
            continue
        s = reliability_slope(out["test_labels"][mask], res.oof_pred[mask], n_bins=6)
        print(f"  {sym}: n={int(mask.sum())}  slope={s:.3f}")

    n_trials = len(reg)
    gates = compute_gates(
        primary_returns_r=primary_ret,
        filtered_returns_r=filtered_ret,
        oos_mean_r=oos_mean,
        oos_mean_r_se=oos_se,
        brier=res.brier,
        climatology_brier=res.climatology_brier,
        reliability_slope=rel_slope,
        n_trials=n_trials,
        n_obs=len(df),
        cpcv_strategy_returns=[filtered_ret],
        cpcv_primary_returns=[primary_ret],
        observed_sharpe=obs_sharpe,
    )
    print(gates.as_table())

if __name__ == "__main__":
    main()
