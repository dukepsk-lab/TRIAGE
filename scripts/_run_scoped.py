"""Scoped run of record — heavy gates on an arbitrary (symbols, rules) slice.

Same machinery as scripts/_run_of_record.py (leakage battery, corr-prune + MDA
on realized trade EV, headline meta-model, full CPCV N=6/k=2 with per-path
refits for a REAL PBO, DSR at the honest registry count, per-symbol attribution)
but parameterized so a *scoped* run of record can be taken through the full
stack — e.g. the majors x donchian slice that PHASE3_RESULTS.md identifies as
the only slice with replicated edge.

Usage:
  python scripts/_run_scoped.py --symbols EURUSD,GBPUSD,USDJPY --rules donchian20 \
      --cap d4/15 --tag majors_donchian

Logged as exactly ONE trial (the scoped run-of-record config). CPCV refits are
the same config on different folds, not new trials (§0/§4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from triage.features import FEATURE_COLUMNS
from triage.features.tier4 import TIER4_COLUMNS, correlation_prune, mda_selection
from triage.gates import compute_gates, cost_for_symbol, dsr, pbo_from_cpcv, strategy_returns_r
from triage.leakage import assert_no_leakage
from triage.models.meta import reliability_slope, train_meta_model
from triage.registry import TrialRegistry, logged_trial
from triage.sizing import default_tau0
from triage.uniqueness import uniqueness_weights
from triage.validation import backtest_paths, cpcv_splits

from run_phase3 import TRIALS_DB, build_pooled_matrices, load_pooled_events, split_pooled

LADDER_TAU0 = round(default_tau0(0.04), 3)
CAPS = {
    "d4/15": {"max_depth": 4, "num_leaves": 15, "learning_rate": 0.03},
    "d6/31": {"max_depth": 6, "num_leaves": 31, "learning_rate": 0.02},
}
CALIBRATION = "holdout"


def _fit_predict(train_df, test_df, feats, params):
    X_tr, y_tr, _ = build_pooled_matrices(train_df, feats)
    X_te, y_te, _ = build_pooled_matrices(test_df, feats)
    for col in ("symbol", "rule_id"):
        X_te[col] = pd.Categorical(X_te[col], categories=X_tr[col].cat.categories)
    ev = train_df[["t0", "t1"]].copy()
    ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
    w = uniqueness_weights(ev)
    res = train_meta_model(
        X_tr, y_tr, sample_weight=w, test_features=X_te, test_labels=y_te,
        events=train_df[["t0", "t1"]], calibration=CALIBRATION,
        params=params, n_estimators=1000,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY")
    ap.add_argument("--rules", default="donchian20")
    ap.add_argument("--cap", default="d4/15", choices=list(CAPS))
    ap.add_argument("--tag", default="majors_donchian")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    params = CAPS[args.cap]

    print("=" * 74)
    print(f"SCOPED RUN OF RECORD — {args.tag}  symbols={symbols} rules={rules} cap={args.cap}")
    print("=" * 74)

    df = load_pooled_events(symbols)
    df = df[df["rule_id"].isin(rules)].sort_values("t0")
    print(f"\nevents: {len(df)}")
    print("per-symbol/rule counts:", df.groupby(["symbol", "rule_id"]).size().to_dict())
    print("per-symbol base rate:", df.groupby("symbol")["label"].mean().round(3).to_dict())
    print("pooled base rate:", round(float(df["label"].mean()), 4))

    print("\n--- Leakage battery (shuffled-label) ---")
    Xfull = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    res_leak = assert_no_leakage(Xfull, df["label"].astype(int))
    print("shuffled-label AUCs:", [round(r.auc, 4) for r in res_leak],
          "-> all within tol:", all(r.passed for r in res_leak))

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

    reg = TrialRegistry(TRIALS_DB)

    @logged_trial(
        {"phase": 3, "config": f"run_of_record_{args.tag}", "basket": symbols,
         "rules": rules, "calibration": CALIBRATION, "cap": args.cap,
         "features": final_features},
        registry=reg,
    )
    def _headline():
        res, _ = _fit_predict(train, test, final_features, params)
        return {"auc": res.auc, "brier": res.brier, "clim": res.climatology_brier}

    _headline()
    res, pred = _fit_predict(train, test, final_features, params)

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

    print("\n--- CPCV (N=6, k=2 -> 15 splits, 5 paths, per-path refits) ---")
    ev_idx = df[["t0", "t1"]].copy()
    splits = cpcv_splits(ev_idx, n_groups=6, k_test=2)
    pair_to_ret = {}
    for sp in splits:
        tr_df = df.loc[sp.train_idx].sort_values("t0")
        te_df = df.loc[sp.test_idx].sort_values("t0")
        if len(te_df) < 20 or te_df["label"].nunique() < 2 or tr_df["label"].nunique() < 2:
            continue
        try:
            _, p_te = _fit_predict(tr_df, te_df, final_features, params)
        except Exception as e:
            print(f"  split {sp.split_id} {sp.test_groups}: refit failed ({e})")
            continue
        yl = te_df["label"].to_numpy().astype(int)
        rr = te_df["r_realized"].to_numpy()
        sy = te_df["symbol"].to_numpy()
        prim = strategy_returns_r(yl, rr, symbols=sy)
        filt = strategy_returns_r(yl, rr, probabilities=p_te, ladder_tau0=LADDER_TAU0, symbols=sy)
        pair_to_ret[sp.test_groups] = {"prim": prim, "filt": filt}
        print(f"  split {sp.split_id} {sp.test_groups}: n={len(te_df)} "
              f"filt_mean={filt.mean():+.4f} prim_mean={prim.mean():+.4f}")

    paths = backtest_paths(6, 2)
    path_filt, path_prim = [], []
    for pth in paths:
        present = [pr for pr in pth if pr in pair_to_ret]
        if not present:
            continue
        path_filt.append(np.concatenate([pair_to_ret[pr]["filt"] for pr in present]))
        path_prim.append(np.concatenate([pair_to_ret[pr]["prim"] for pr in present]))
    pbo = pbo_from_cpcv(path_filt, path_prim)

    def _sh(x):
        return float(x.mean() / x.std() * np.sqrt(252)) if len(x) and x.std() > 0 else 0.0
    print(f"\nCPCV paths: {len(path_filt)}   PBO={pbo:.3f}")
    for i, (f, p) in enumerate(zip(path_filt, path_prim)):
        print(f"  path {i}: n={len(f)} filt_sharpe={_sh(f):.3f} prim_sharpe={_sh(p):.3f} "
              f"filt_mean_r={f.mean():+.4f}")

    n_trials = len(reg)
    gates = compute_gates(
        primary_returns_r=primary, filtered_returns_r=filtered,
        oos_mean_r=oos_mean, oos_mean_r_se=oos_se,
        brier=res.brier, climatology_brier=res.climatology_brier,
        reliability_slope=rel_slope, n_trials=n_trials, n_obs=len(df),
        cpcv_strategy_returns=path_filt, cpcv_primary_returns=path_prim,
        observed_sharpe=obs_sharpe,
    )
    print("\n--- GATES (scoped run of record) ---")
    print(gates.as_table())
    print(f"\nHonest trial count (len registry): {n_trials}")
    print(f"pooled reliability slope: {rel_slope:.3f}")

    summary = {
        "tag": args.tag, "symbols": symbols, "rules": rules, "cap": args.cap,
        "n_events": len(df),
        "per_symbol_counts": {k: int(v) for k, v in df.groupby("symbol").size().items()},
        "per_symbol_base_rate": {k: round(float(v), 3) for k, v in df.groupby("symbol")["label"].mean().items()},
        "pooled_base_rate": round(float(df["label"].mean()), 4),
        "shuffled_aucs": [round(r.auc, 4) for r in res_leak],
        "final_features": final_features, "tier4_kept": tier4_kept,
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
        "rel_slope_pooled": rel_slope, "n_trials": n_trials, "obs_sharpe": obs_sharpe,
    }
    out = Path(".triage") / f"run_of_record_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
