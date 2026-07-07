"""Decision-tree diagnostic (§6): swap the rule, and check capacity.

The full6/holdout run-of-record cell has AUC~0.515, G1 FAIL, flat p-hat (§11
signature). Before declaring falsification, §6 mandates: swap the RULE (try the
other family pooled), not just the model; a coarse capacity grid on FIXED folds
is also sanctioned (§5). This isolates:
  - rule family: donchian-only vs pullback-only vs both (does one carry edge?)
  - capacity: depth4/15 (the §1/§6 tiny prior) vs depth6/31 (§5 upper bound)
All on the full basket, holdout calibration, one fixed purged split. Every
config is logged as an honest trial (§0).

Also reproduces the Phase 2 cell (3 majors, donchian, depth4/15) as a sanity
anchor — that cell should still show the AUC~0.56 edge Phase 2 reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from triage.features import FEATURE_COLUMNS
from triage.gates import compute_gates, cost_for_symbol, strategy_returns_r
from triage.models.meta import reliability_slope, train_meta_model
from triage.registry import TrialRegistry, logged_trial
from triage.sizing import default_tau0
from triage.uniqueness import uniqueness_weights

from run_phase3 import TRIALS_DB, build_pooled_matrices, load_pooled_events, split_pooled

FULL6 = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "BTCUSD"]
MAJORS3 = ["EURUSD", "GBPUSD", "USDJPY"]
LADDER_TAU0 = round(default_tau0(0.04), 3)
CAPS = {"d4/15": {"max_depth": 4, "num_leaves": 15, "learning_rate": 0.03},
        "d6/31": {"max_depth": 6, "num_leaves": 31, "learning_rate": 0.02}}


def run_cell(df_all, symbols, rules, params, cap_name, calib, reg):
    df = df_all[df_all["symbol"].isin(symbols) & df_all["rule_id"].isin(rules)].copy()
    df = df.sort_values("t0")
    if len(df) < 200:
        return None
    train, test = split_pooled(df)
    feats = FEATURE_COLUMNS
    X_tr, y_tr, _ = build_pooled_matrices(train, feats)
    X_te, y_te, _ = build_pooled_matrices(test, feats)
    for col in ("symbol", "rule_id"):
        X_te[col] = pd.Categorical(X_te[col], categories=X_tr[col].cat.categories)
    ev = train[["t0", "t1"]].copy()
    ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
    w = uniqueness_weights(ev)

    cfg = {"exp": "diag", "symbols": symbols, "rules": sorted(rules),
           "cap": cap_name, "calibration": calib}

    @logged_trial(cfg, registry=reg)
    def _t():
        r = train_meta_model(X_tr, y_tr, sample_weight=w, test_features=X_te,
                             test_labels=y_te, events=train[["t0", "t1"]],
                             calibration=calib, params=params, n_estimators=1000)
        return {"auc": r.auc, "brier": r.brier}
    _t()
    res = train_meta_model(X_tr, y_tr, sample_weight=w, test_features=X_te,
                          test_labels=y_te, events=train[["t0", "t1"]],
                          calibration=calib, params=params, n_estimators=1000)
    tl = test["label"].to_numpy().astype(int)
    tr_r = test["r_realized"].to_numpy()
    ts = test["symbol"].to_numpy()
    prim = strategy_returns_r(tl, tr_r, symbols=ts)
    filt = strategy_returns_r(tl, tr_r, probabilities=res.oof_pred, ladder_tau0=LADDER_TAU0, symbols=ts)
    taken = filt[filt != 0]
    oos_mean = float(taken.mean()) if len(taken) else 0.0
    oos_se = float(taken.std() / np.sqrt(len(taken))) if len(taken) > 1 else 1.0
    obs_sharpe = float(prim.mean() / prim.std() * np.sqrt(252)) if prim.std() > 0 else 0.0
    slope = reliability_slope(tl, res.oof_pred, 8)
    g = compute_gates(primary_returns_r=prim, filtered_returns_r=filt, oos_mean_r=oos_mean,
                      oos_mean_r_se=oos_se, brier=res.brier, climatology_brier=res.climatology_brier,
                      reliability_slope=slope, n_trials=len(reg), n_obs=len(df),
                      cpcv_strategy_returns=[filt], cpcv_primary_returns=[prim], observed_sharpe=obs_sharpe)
    # per-symbol AUC
    psym = {}
    for s in sorted(set(ts)):
        m = ts == s
        if int(m.sum()) >= 20 and len(set(tl[m])) > 1:
            psym[s] = round(float(roc_auc_score(tl[m], res.oof_pred[m])), 3)
    return {"n": len(df), "auc": res.auc, "brier": res.brier, "clim": res.climatology_brier,
            "slope": slope, "g1_ds": g.g1_uplift_delta_sharpe, "g1_p": g.g1_uplift_pvalue,
            "g1": g.g1_pass, "g4_t": g.g4_tstat_mean_r, "g5": g.g5_pass, "psym": psym}


def main():
    reg = TrialRegistry(TRIALS_DB)
    df_all = load_pooled_events(FULL6)
    cells = [
        ("majors3/donch", MAJORS3, ["donchian20"], "d4/15", "holdout"),
        ("full6/donch", FULL6, ["donchian20"], "d4/15", "holdout"),
        ("full6/donch", FULL6, ["donchian20"], "d6/31", "holdout"),
        ("full6/pull", FULL6, ["htf_pullback"], "d4/15", "holdout"),
        ("full6/both", FULL6, ["donchian20", "htf_pullback"], "d4/15", "holdout"),
        ("full6/both", FULL6, ["donchian20", "htf_pullback"], "d6/31", "holdout"),
    ]
    print(f"{'cell':<16}{'cap':<7}{'n':>6}{'AUC':>8}{'Brier':>8}{'clim':>8}{'slope':>8}"
          f"{'G1dS':>8}{'G1p':>7}{'G4t':>7}  G1 G5")
    rows = []
    for name, syms, rules, cap, calib in cells:
        r = run_cell(df_all, syms, rules, CAPS[cap], cap, calib, reg)
        if r is None:
            print(f"{name:<16}{cap:<7}  (skipped)")
            continue
        rows.append((name, cap, r))
        print(f"{name:<16}{cap:<7}{r['n']:>6}{r['auc']:>8.4f}{r['brier']:>8.4f}{r['clim']:>8.4f}"
              f"{r['slope']:>8.3f}{r['g1_ds']:>8.3f}{r['g1_p']:>7.3f}{r['g4_t']:>7.2f}  "
              f"{'Y' if r['g1'] else '.'}  {'Y' if r['g5'] else '.'}")
    print("\nPer-symbol AUC:")
    for name, cap, r in rows:
        print(f"  {name}/{cap}: {r['psym']}")
    print(f"\nregistry trial count now: {len(reg)}")


if __name__ == "__main__":
    main()
