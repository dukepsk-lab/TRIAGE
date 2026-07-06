"""Head-to-head: fix the calibrator vs. drop the symbol.

Answers the question raised during architecture validation — is the G5
calibration failure fixable by the calibrator (§6 "purged calibration fold")
on the FULL basket, or does it require dropping symbols (which §0/§10 warn is
basket-overfitting)?

Runs a 2x2 matrix over the same pooled event set:

    basket       x   calibration
    ---------------------------------
    full 6       x   {cv, holdout}
    curated 3    x   {cv, holdout}

For each cell it prints G1-G5 and a per-symbol reliability-slope breakdown,
holding features (Tier1-3 + all Tier4) and model params fixed so the ONLY
things that vary are basket composition and calibration method. Requires a
populated bar store (run ``scripts/ingest_csv.py`` first) and reuses the
Phase 3 backfill/split/matrix helpers so labels are identical across cells.

    uv run python scripts/run_basket_comparison.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from triage.features import FEATURE_COLUMNS
from triage.features.tier4 import TIER4_COLUMNS
from triage.gates import compute_gates, strategy_returns_r
from triage.models.meta import reliability_slope, train_meta_model
from triage.registry import TrialRegistry, logged_trial
from triage.sizing import default_tau0
from triage.uniqueness import uniqueness_weights

from run_phase3 import (  # reuse identical label/split logic across cells
    TRIALS_DB,
    backfill_symbol_events,
    build_pooled_matrices,
    load_pooled_events,
    split_pooled,
)

FULL6 = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "BTCUSD"]
CURATED3 = ["EURUSD", "USDJPY", "BTCUSD"]

# Fixed across all cells so basket + calibration are the only variables.
FINAL_FEATURES = FEATURE_COLUMNS + TIER4_COLUMNS
MODEL_PARAMS = {"max_depth": 6, "num_leaves": 31, "learning_rate": 0.02}
LADDER_TAU0 = round(default_tau0(0.04), 3)  # scalar filter threshold for G1


def _run_cell(
    symbols: list[str], calibration: str, registry: TrialRegistry
) -> dict:
    """Train + gate one (basket, calibration) cell. Returns metrics + slopes."""
    df = load_pooled_events(symbols)
    train, test = split_pooled(df)

    X_tr, y_tr, _ = build_pooled_matrices(train, FINAL_FEATURES)
    X_te, y_te, meta_te = build_pooled_matrices(test, FINAL_FEATURES)
    # Align test categoricals to the training category levels.
    for col in ("symbol", "rule_id"):
        X_te[col] = pd.Categorical(X_te[col], categories=X_tr[col].cat.categories)

    ev = train[["t0", "t1"]].copy()
    ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
    w = uniqueness_weights(ev)

    @logged_trial(
        {"exp": "basket_comparison", "symbols": symbols, "calibration": calibration},
        registry=registry,
    )
    def _train() -> dict:
        res = train_meta_model(
            X_tr, y_tr, sample_weight=w,
            test_features=X_te, test_labels=y_te,
            events=train[["t0", "t1"]],
            calibration=calibration,
            params=MODEL_PARAMS, n_estimators=1000,
        )
        return {"auc": res.auc, "brier": res.brier}

    _train()
    res = train_meta_model(
        X_tr, y_tr, sample_weight=w,
        test_features=X_te, test_labels=y_te,
        events=train[["t0", "t1"]],
        calibration=calibration,
        params=MODEL_PARAMS, n_estimators=1000,
    )

    test_labels = test["label"].to_numpy().astype(int)
    test_r = test["r_realized"].to_numpy()
    test_syms = test["symbol"].to_numpy()

    primary = strategy_returns_r(test_labels, test_r, symbols=test_syms)
    filtered = strategy_returns_r(
        test_labels, test_r, probabilities=res.oof_pred,
        ladder_tau0=LADDER_TAU0, symbols=test_syms,
    )
    taken = filtered[filtered != 0]
    oos_mean = float(taken.mean()) if len(taken) else 0.0
    oos_se = float(taken.std() / np.sqrt(len(taken))) if len(taken) > 1 else 1.0
    obs_sharpe = (
        float(primary.mean() / primary.std() * np.sqrt(252))
        if primary.std() > 0 else 0.0
    )
    rel_slope = reliability_slope(test_labels, res.oof_pred, n_bins=8)

    gates = compute_gates(
        primary_returns_r=primary, filtered_returns_r=filtered,
        oos_mean_r=oos_mean, oos_mean_r_se=oos_se,
        brier=res.brier, climatology_brier=res.climatology_brier,
        reliability_slope=rel_slope, n_trials=len(registry), n_obs=len(df),
        cpcv_strategy_returns=[filtered], cpcv_primary_returns=[primary],
        observed_sharpe=obs_sharpe,
    )

    per_symbol = {}
    for s in sorted(set(test_syms)):
        m = test_syms == s
        if int(m.sum()) < 20 or len(set(test_labels[m])) < 2:
            per_symbol[s] = None
        else:
            per_symbol[s] = round(reliability_slope(test_labels[m], res.oof_pred[m], 6), 3)

    return {
        "n_events": len(df), "auc": res.auc, "brier": res.brier,
        "clim": res.climatology_brier, "gates": gates,
        "rel_slope": rel_slope, "per_symbol": per_symbol,
    }


def main() -> None:
    print("=" * 74)
    print("BASKET COMPARISON — fix the calibrator vs. drop the symbol")
    print("=" * 74)

    print("\n[1] Backfilling events for the full 6-symbol basket...")
    for sym in FULL6:
        backfill_symbol_events(sym)

    reg = TrialRegistry(TRIALS_DB)
    cells = [
        ("full6", FULL6, "cv"), ("full6", FULL6, "holdout"),
        ("curated3", CURATED3, "cv"), ("curated3", CURATED3, "holdout"),
    ]
    results = {}
    for name, syms, calib in cells:
        print(f"\n[cell] basket={name}  calibration={calib}")
        try:
            results[(name, calib)] = _run_cell(syms, calib, reg)
        except Exception as e:  # keep the matrix going; report the failure
            print(f"  FAILED: {e}")
            results[(name, calib)] = None

    print("\n" + "=" * 74)
    print("SUMMARY (G5 target: Brier < clim AND slope in [0.8, 1.2])")
    print("=" * 74)
    hdr = f"{'basket':<10}{'calib':<9}{'n':>6}{'AUC':>8}{'Brier':>8}{'clim':>8}{'slope':>8}  gates"
    print(hdr)
    print("-" * len(hdr))
    for (name, calib), r in results.items():
        if r is None:
            print(f"{name:<10}{calib:<9}  (failed)")
            continue
        g = r["gates"]
        passes = "".join(
            k if p else "." for k, p in
            [("1", g.g1_pass), ("2", g.g2_pass), ("3", g.g3_pass),
             ("4", g.g4_pass), ("5", g.g5_pass)]
        )
        print(
            f"{name:<10}{calib:<9}{r['n_events']:>6}{r['auc']:>8.4f}"
            f"{r['brier']:>8.4f}{r['clim']:>8.4f}{r['rel_slope']:>8.3f}  G[{passes}]"
        )

    print("\nPer-symbol reliability slope (target [0.8, 1.2]):")
    for (name, calib), r in results.items():
        if r is None:
            continue
        parts = "  ".join(
            f"{s}={v}" if v is not None else f"{s}=n/a" for s, v in r["per_symbol"].items()
        )
        print(f"  {name}/{calib}: {parts}")

    print(
        "\nRead: if full6/holdout passes G5 (slope in band) then dropping "
        "GBPUSD/XAU/US30 was unnecessary — the fix belonged in the calibrator, "
        "not the basket (§0/§10). If only curated3 passes, and a per-symbol "
        "slope identifies the true outlier, that is a documented, principled "
        "drop rather than gate-chasing."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
