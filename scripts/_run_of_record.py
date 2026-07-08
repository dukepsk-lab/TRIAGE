"""Full-basket run of record — heavy gates (§3 step 6, §7).

Full 6-symbol basket, holdout calibration, Tier 4 gated by MDA (metric='ev').
Produces:
  * headline meta-model on a purged 70/30 split (AUC/Brier/clim, EV-vs-tau),
  * G1-G5 with a REAL G3 PBO from full CPCV (N=6, k=2 -> 15 splits, 5 paths)
    with per-path model refits,
  * G2 DSR at the honest registry trial count,
  * per-symbol attribution (AUC + reliability slope).

Logged as exactly ONE trial (the run-of-record config). The CPCV refits are
the same config evaluated on different folds, not new trials — logging each
would dishonestly inflate the DSR/PBO trial count (§0/§4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from triage.features import FEATURE_COLUMNS
from triage.features.tier4 import TIER4_COLUMNS, correlation_prune, mda_selection
from triage.gates import (
    GateResults,
    compute_gates,
    cost_for_symbol,
    dsr,
    pbo_from_cpcv,
    strategy_returns_r,
)
from triage.leakage import assert_no_leakage
from triage.models.meta import reliability_slope, train_meta_model
from triage.registry import TrialRegistry, logged_trial
from triage.sizing import default_tau0
from triage.uniqueness import uniqueness_weights
from triage.validation import backtest_paths, cpcv_splits

from run_phase3 import (
    TRIALS_DB,
    build_pooled_matrices,
    load_pooled_events,
    split_pooled,
)

FULL6 = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "BTCUSD"]
LADDER_TAU0 = round(default_tau0(0.04), 3)
MODEL_PARAMS = {"max_depth": 6, "num_leaves": 31, "learning_rate": 0.02}
CALIBRATION = "holdout"


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feats: list[str]):
    """Train (holdout-calibrated) on train_df, predict on test_df. Returns
    (result, oof_pred aligned to test_df rows)."""
    X_tr, y_tr, _ = build_pooled_matrices(train_df, feats)
    X_te, y_te, _ = build_pooled_matrices(test_df, feats)
    for col in ("symbol", "rule_id"):
        X_te[col] = pd.Categorical(X_te[col], categories=X_tr[col].cat.categories)
    ev = train_df[["t0", "t1"]].copy()
    ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
    w = uniqueness_weights(ev)
    res = train_meta_model(
        X_tr, y_tr, sample_weight=w,
        test_features=X_te, test_labels=y_te,
        events=train_df[["t0", "t1"]],
        calibration=CALIBRATION, params=MODEL_PARAMS, n_estimators=1000,
    )
    return res, res.oof_pred


def _ev_vs_tau(labels, r, preds, symbols):
    cost = np.array([cost_for_symbol(s) for s in symbols])
    rows = []
    for tau in np.arange(0.30, 0.72, 0.025):
        taken = preds >= tau
        n = int(taken.sum())
        mr = float(np.mean(r[taken] - cost[taken])) if n else 0.0
        rows.append((round(float(tau), 3), n, round(mr, 4)))
    return rows


def main() -> None:
    print("=" * 74)
    print("FULL-BASKET RUN OF RECORD — heavy gates (CPCV per-path PBO + DSR)")
    print("=" * 74)

    df = load_pooled_events(FULL6)
    df = df.sort_values("t0")
    print(f"\npooled events: {len(df)}")
    print("per-symbol/rule counts:")
    print(df.groupby(["symbol", "rule_id"]).size().to_string())
    print("per-symbol base rate:", df.groupby("symbol")["label"].mean().round(3).to_dict())
    print("pooled base rate:", round(float(df["label"].mean()), 4))

    # ---- Leakage battery on the pooled REAL feature matrix (§4) ----
    print("\n--- Leakage battery (shuffled-label) ---")
    Xfull = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    res_leak = assert_no_leakage(Xfull, df["label"].astype(int))
    print("shuffled-label AUCs:", [round(r.auc, 4) for r in res_leak],
          "-> all within tol:", all(r.passed for r in res_leak))

    # ---- Feature selection (corr-prune + MDA on EV), on train portion ----
    train, test = split_pooled(df)
    print(f"\npurged train/test: {len(train)} / {len(test)}")
    cand = FEATURE_COLUMNS + TIER4_COLUMNS
    Xc = train[cand].apply(pd.to_numeric, errors="coerce").fillna(0)
    kept_corr = correlation_prune(Xc)
    mda = mda_selection(Xc[kept_corr], train["label"].astype(int), train, metric="ev")
    print("MDA (metric=ev):")
    print(mda.to_string(index=False))
    tier4_kept = [c for c in mda[mda["kept"]]["feature"].tolist() if c in TIER4_COLUMNS]
    final_features = FEATURE_COLUMNS + tier4_kept
    print("Tier4 kept by MDA:", tier4_kept)
    print("final features:", final_features)

    reg = TrialRegistry(TRIALS_DB)

    # ---- Headline model on the purged 70/30 split (logged as ONE trial) ----
    @logged_trial(
        {"phase": 3, "config": "run_of_record", "basket": FULL6,
         "calibration": CALIBRATION, "params": MODEL_PARAMS,
         "features": final_features},
        registry=reg,
    )
    def _headline() -> dict:
        res, pred = _fit_predict(train, test, final_features)
        return {"auc": res.auc, "brier": res.brier, "clim": res.climatology_brier}

    _headline()
    res, pred = _fit_predict(train, test, final_features)

    test_labels = test["label"].to_numpy().astype(int)
    test_r = test["r_realized"].to_numpy()
    test_syms = test["symbol"].to_numpy()
    print(f"\nheadline: AUC={res.auc:.4f} Brier={res.brier:.4f} clim={res.climatology_brier:.4f}"
          f" n_train={res.n_train} n_test={res.n_test}")

    print("\nEV-vs-tau (mean R after per-symbol cost):")
    for tau, n, mr in _ev_vs_tau(test_labels, test_r, pred, test_syms):
        print(f"  tau={tau:.3f}  n={n:4d}  mean_r={mr:+.4f}")

    primary = strategy_returns_r(test_labels, test_r, symbols=test_syms)
    filtered = strategy_returns_r(test_labels, test_r, probabilities=pred,
                                  ladder_tau0=LADDER_TAU0, symbols=test_syms)
    taken = filtered[filtered != 0]
    oos_mean = float(taken.mean()) if len(taken) else 0.0
    oos_se = float(taken.std() / np.sqrt(len(taken))) if len(taken) > 1 else 1.0
    obs_sharpe = float(primary.mean() / primary.std() * np.sqrt(252)) if primary.std() > 0 else 0.0
    rel_slope = reliability_slope(test_labels, pred, n_bins=8)

    # ---- Per-symbol attribution ----
    from sklearn.metrics import roc_auc_score
    print("\n--- Per-symbol attribution (headline test fold) ---")
    print(f"{'symbol':<8}{'n':>6}{'auc':>8}{'mean_r':>9}{'slope':>8}{'base':>7}")
    per_symbol = {}
    for s in sorted(set(test_syms)):
        m = test_syms == s
        n = int(m.sum())
        if n < 20 or len(set(test_labels[m])) < 2:
            print(f"{s:<8}{n:>6}   (too few / single-class)")
            per_symbol[s] = None
            continue
        auc_s = float(roc_auc_score(test_labels[m], pred[m]))
        mr_s = float(np.mean(test_r[m] - np.array([cost_for_symbol(s)] * n)))
        sl_s = reliability_slope(test_labels[m], pred[m], n_bins=6)
        br_s = float(test_labels[m].mean())
        per_symbol[s] = {"n": n, "auc": round(auc_s, 4), "mean_r": round(mr_s, 4),
                         "slope": round(sl_s, 3), "base_rate": round(br_s, 3)}
        print(f"{s:<8}{n:>6}{auc_s:>8.4f}{mr_s:>9.4f}{sl_s:>8.3f}{br_s:>7.3f}")

    # ---- Full CPCV with per-path refits for a REAL PBO (G3) ----
    print("\n--- CPCV (N=6, k=2 -> 15 splits, 5 paths, per-path refits) ---")
    ev_idx = df[["t0", "t1"]].copy()
    splits = cpcv_splits(ev_idx, n_groups=6, k_test=2)
    pair_to_ret: dict[tuple, dict] = {}
    for sp in splits:
        tr_df = df.loc[sp.train_idx].sort_values("t0")
        te_df = df.loc[sp.test_idx].sort_values("t0")
        if len(te_df) < 20 or te_df["label"].nunique() < 2 or tr_df["label"].nunique() < 2:
            continue
        try:
            _, p_te = _fit_predict(tr_df, te_df, final_features)
        except Exception as e:
            print(f"  split {sp.split_id} groups {sp.test_groups}: refit failed ({e})")
            continue
        yl = te_df["label"].to_numpy().astype(int)
        rr = te_df["r_realized"].to_numpy()
        sy = te_df["symbol"].to_numpy()
        prim = strategy_returns_r(yl, rr, symbols=sy)
        filt = strategy_returns_r(yl, rr, probabilities=p_te, ladder_tau0=LADDER_TAU0, symbols=sy)
        pair_to_ret[sp.test_groups] = {"prim": prim, "filt": filt}
        print(f"  split {sp.split_id} groups {sp.test_groups}: n_test={len(te_df)} "
              f"filt_mean={filt.mean():+.4f} prim_mean={prim.mean():+.4f}")

    paths = backtest_paths(6, 2)
    path_filt, path_prim = [], []
    for pth in paths:
        f = np.concatenate([pair_to_ret[pr]["filt"] for pr in pth if pr in pair_to_ret]) if any(pr in pair_to_ret for pr in pth) else np.array([])
        p = np.concatenate([pair_to_ret[pr]["prim"] for pr in pth if pr in pair_to_ret]) if any(pr in pair_to_ret for pr in pth) else np.array([])
        if len(f):
            path_filt.append(f)
            path_prim.append(p)
    pbo = pbo_from_cpcv(path_filt, path_prim)
    # CPCV path-level uplift diagnostics.
    def _sh(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if len(x) and x.std() > 0 else 0.0
    print(f"\nCPCV paths: {len(path_filt)}   PBO={pbo:.3f}")
    for i, (f, p) in enumerate(zip(path_filt, path_prim)):
        print(f"  path {i}: n={len(f)} filt_sharpe={_sh(f):.3f} prim_sharpe={_sh(p):.3f} "
              f"filt_mean_r={f.mean():+.4f}")

    n_trials = len(reg)
    g2 = dsr(obs_sharpe, n_trials=n_trials, n_obs=len(df))

    gates = compute_gates(
        primary_returns_r=primary, filtered_returns_r=filtered,
        oos_mean_r=oos_mean, oos_mean_r_se=oos_se,
        brier=res.brier, climatology_brier=res.climatology_brier,
        reliability_slope=rel_slope, n_trials=n_trials, n_obs=len(df),
        cpcv_strategy_returns=path_filt, cpcv_primary_returns=path_prim,
        observed_sharpe=obs_sharpe,
    )
    print("\n--- GATES (run of record) ---")
    print(gates.as_table())
    print(f"\nHonest trial count (len registry): {n_trials}")
    print(f"pooled reliability slope: {rel_slope:.3f}")

    # Dump a machine-readable summary for the writeup.
    summary = {
        "n_events": len(df),
        "per_symbol_counts": {k: int(v) for k, v in df.groupby("symbol").size().items()},
        "per_symbol_rule_counts": {f"{a}/{b}": int(v) for (a, b), v in df.groupby(["symbol", "rule_id"]).size().items()},
        "per_symbol_base_rate": {k: round(float(v), 3) for k, v in df.groupby("symbol")["label"].mean().items()},
        "pooled_base_rate": round(float(df["label"].mean()), 4),
        "shuffled_aucs": [round(r.auc, 4) for r in res_leak],
        "final_features": final_features,
        "tier4_kept": tier4_kept,
        "mda": mda.to_dict(orient="records"),
        "headline": {"auc": res.auc, "brier": res.brier, "clim": res.climatology_brier,
                     "n_train": res.n_train, "n_test": res.n_test},
        "ev_vs_tau": _ev_vs_tau(test_labels, test_r, pred, test_syms),
        "per_symbol": per_symbol,
        "gates": {
            "g1": {"delta_sharpe": gates.g1_uplift_delta_sharpe, "pvalue": gates.g1_uplift_pvalue, "pass": gates.g1_pass},
            "g2_dsr": {"value": gates.g2_dsr, "pass": gates.g2_pass},
            "g3_pbo": {"value": gates.g3_pbo, "pass": gates.g3_pass},
            "g4_tstat": {"value": gates.g4_tstat_mean_r, "pass": gates.g4_pass},
            "g5": {"brier": gates.g5_brier, "clim": gates.g5_climatology_brier,
                   "slope": gates.g5_reliability_slope, "pass": gates.g5_pass},
        },
        "cpcv": {"n_paths": len(path_filt), "pbo": pbo,
                 "path_filt_sharpe": [_sh(f) for f in path_filt],
                 "path_prim_sharpe": [_sh(p) for p in path_prim]},
        "rel_slope_pooled": rel_slope,
        "n_trials": n_trials,
        "obs_sharpe": obs_sharpe,
    }
    out = Path(".triage") / "run_of_record_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
