"""§6 calibration sweep for the majors x donchian slice (G5 lever).

G1 passes / G5 fails on majors x donchian (see docs/MAJORS_DONCHIAN_RESULTS.md).
§6 says the lever is the CALIBRATOR, not the basket. This sweeps the three
sanctioned knobs — isotonic<->sigmoid, calib_frac, pooled vs per-symbol — over
one FIXED feature set and one FIXED purged split (only the calibrator moves).
Every config is logged as an honest trial (§0).

Question answered: does any sanctioned calibrator get the pooled reliability
slope into [0.8, 1.2] with Brier < climatology at this n, or is G5 data-bound?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from triage.features import FEATURE_COLUMNS
from triage.features.tier4 import TIER4_COLUMNS, correlation_prune, mda_selection
from triage.gates import block_bootstrap_uplift, cost_for_symbol, strategy_returns_r
from triage.models.meta import reliability_slope, train_meta_model
from triage.registry import TrialRegistry, logged_trial
from triage.sizing import default_tau0
from triage.uniqueness import uniqueness_weights

from run_phase3 import TRIALS_DB, build_pooled_matrices, load_pooled_events, split_pooled

MAJORS3 = ["EURUSD", "GBPUSD", "USDJPY"]
RULES = ["donchian20"]
PARAMS = {"max_depth": 4, "num_leaves": 15, "learning_rate": 0.03}
LADDER_TAU0 = round(default_tau0(0.04), 3)

# (calib_frac, method, per_symbol) — covers all three §6 levers economically.
GRID = [
    (0.20, "sigmoid", False),   # baseline (reproduces the scoped run)
    (0.20, "isotonic", False),
    (0.35, "sigmoid", False),
    (0.35, "isotonic", False),
    (0.20, "isotonic", True),   # per-symbol
    (0.35, "isotonic", True),
]


def main():
    df = load_pooled_events(MAJORS3)
    df = df[df["rule_id"].isin(RULES)].sort_values("t0")
    train, test = split_pooled(df)

    # Fixed feature set (corr-prune + MDA on realized EV) — computed ONCE so the
    # only thing that varies across the sweep is the calibrator.
    cand = FEATURE_COLUMNS + TIER4_COLUMNS
    Xc = train[cand].apply(pd.to_numeric, errors="coerce").fillna(0)
    kept_corr = correlation_prune(Xc)
    mda = mda_selection(Xc[kept_corr], train["label"].astype(int), train, metric="ev")
    tier4_kept = [c for c in mda[mda["kept"]]["feature"].tolist() if c in TIER4_COLUMNS]
    feats = FEATURE_COLUMNS + tier4_kept
    print(f"events={len(df)}  train/test={len(train)}/{len(test)}  tier4_kept={tier4_kept}")

    X_tr, y_tr, _ = build_pooled_matrices(train, feats)
    X_te, y_te, _ = build_pooled_matrices(test, feats)
    for col in ("symbol", "rule_id"):
        X_te[col] = pd.Categorical(X_te[col], categories=X_tr[col].cat.categories)
    ev = train[["t0", "t1"]].copy()
    ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
    w = uniqueness_weights(ev)

    tl = test["label"].to_numpy().astype(int)
    tr_r = test["r_realized"].to_numpy()
    ts = test["symbol"].to_numpy()
    clim_ref = None
    reg = TrialRegistry(TRIALS_DB)

    print(f"\n{'calib_frac':>10}{'method':>10}{'per_sym':>8}{'AUC':>8}{'Brier':>8}{'clim':>8}"
          f"{'slope':>8}{'G5':>4}{'G1dS':>8}{'G1p':>7}  per-symbol slopes")
    rows = []
    for frac, method, psym in GRID:
        cfg = {"exp": "calib_sweep", "symbols": MAJORS3, "rules": RULES,
               "calib_frac": frac, "calib_method": method, "per_symbol_calib": psym}

        @logged_trial(cfg, registry=reg)
        def _t():
            r = train_meta_model(
                X_tr, y_tr, sample_weight=w, test_features=X_te, test_labels=y_te,
                events=train[["t0", "t1"]], calibration="holdout",
                calib_frac=frac, calib_method=method, per_symbol_calib=psym,
                params=PARAMS, n_estimators=1000,
            )
            return {"auc": r.auc, "brier": r.brier, "slope": reliability_slope(y_te.to_numpy(), r.oof_pred, 8)}
        _t()
        res = train_meta_model(
            X_tr, y_tr, sample_weight=w, test_features=X_te, test_labels=y_te,
            events=train[["t0", "t1"]], calibration="holdout",
            calib_frac=frac, calib_method=method, per_symbol_calib=psym,
            params=PARAMS, n_estimators=1000,
        )
        pred = res.oof_pred
        slope = reliability_slope(tl, pred, 8)
        g5 = (res.brier < res.climatology_brier) and (0.8 <= slope <= 1.2)
        prim = strategy_returns_r(tl, tr_r, symbols=ts)
        filt = strategy_returns_r(tl, tr_r, probabilities=pred, ladder_tau0=LADDER_TAU0, symbols=ts)
        g1_ds, g1_p = block_bootstrap_uplift(prim, filt)
        psym_sl = {}
        for s in sorted(set(ts)):
            m = ts == s
            if int(m.sum()) >= 20 and len(set(tl[m])) > 1:
                psym_sl[s] = round(reliability_slope(tl[m], pred[m], 6), 3)
        rows.append({"frac": frac, "method": method, "per_sym": psym, "auc": res.auc,
                     "brier": res.brier, "clim": res.climatology_brier, "slope": slope,
                     "g5": g5, "g1_ds": g1_ds, "g1_p": g1_p, "psym": psym_sl})
        print(f"{frac:>10.2f}{method:>10}{str(psym):>8}{res.auc:>8.4f}{res.brier:>8.4f}"
              f"{res.climatology_brier:>8.4f}{slope:>8.3f}{'Y' if g5 else '.':>4}"
              f"{g1_ds:>8.3f}{g1_p:>7.3f}  {psym_sl}")

    best = max(rows, key=lambda r: (r["g5"], -abs(r["slope"] - 1.0)))
    print(f"\nregistry trial count now: {len(reg)}")
    print(f"best-by-G5: frac={best['frac']} method={best['method']} per_sym={best['per_sym']} "
          f"slope={best['slope']:.3f} G5={'PASS' if best['g5'] else 'FAIL'}")
    Path(".triage/calib_sweep.json").write_text(json.dumps(rows, indent=2, default=str))
    print("wrote .triage/calib_sweep.json")


if __name__ == "__main__":
    main()
