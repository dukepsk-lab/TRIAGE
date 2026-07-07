"""Paper-trading dry-run for the promotable slice: majors x donchian, filtered,
FIXED sizing (rung 2). See docs/MAJORS_DONCHIAN_RESULTS.md — G1 uplift is real
and calibrator-invariant; G5 (probability sizing, rung 3) is deferred, so the
sizer stays flat.

Fits a frozen PaperStrategy on the purged training period and walks the held-out
period through decide() under the §8 overlays (per-symbol single open position,
aggregate open-risk cap). Reports the filtered fixed-size paper result vs the
take-all baseline, with a per-symbol breakdown, and writes the frozen model +
trade log + equity curve.

NOTE: this is an OOS *simulation* on the held-out test fold (a dry-run of the
exact live decision path), not forward live paper results.

  python scripts/run_paper_majors_donchian.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from triage.features import FEATURE_COLUMNS
from triage.paper import PaperStrategy

from run_phase3 import load_pooled_events, split_pooled

MAJORS3 = ["EURUSD", "GBPUSD", "USDJPY"]
# MDA-selected feature set from the scoped run of record
# (docs/MAJORS_DONCHIAN_RESULTS.md): Tier 1-3 + cs_atr_pct_rank.
FINAL_FEATURES = FEATURE_COLUMNS + ["cs_atr_pct_rank"]
PARAMS = {"max_depth": 4, "num_leaves": 15, "learning_rate": 0.03}


def main() -> None:
    print("=" * 74)
    print("PAPER DRY-RUN — majors x donchian, filtered, FIXED sizing (rung 2)")
    print("=" * 74)

    df = load_pooled_events(MAJORS3)
    df = df[df["rule_id"] == "donchian20"].sort_values("t0")
    train, test = split_pooled(df)
    print(f"events={len(df)}  purged train/test={len(train)}/{len(test)}")

    strat = PaperStrategy(features=FINAL_FEATURES, base_risk=0.01,
                          cost_default=0.04, calib_frac=0.20, params=PARAMS)
    strat.fit(train)
    print(f"fit: booster on {strat.n_fit} rows, {strat._method} calibrator on "
          f"{strat.n_calib} rows; tau0={strat.tau0}")

    rep = strat.simulate(test)
    s, b = rep.stats, rep.baseline_stats

    print("\n--- FILTERED (fixed size) vs TAKE-ALL, on the held-out fold ---")
    print(f"{'':22}{'filtered':>12}{'take-all':>12}")
    print(f"{'n_trades':22}{s.get('n_trades',0):>12}{b.get('n_trades',0):>12}")
    print(f"{'total_R (after cost)':22}{s.get('total_r',0):>12.3f}{b.get('total_r',0):>12.3f}")
    print(f"{'mean_R/trade':22}{s.get('mean_r_after_cost',0):>12.4f}{b.get('mean_r_after_cost',0):>12.4f}")
    print(f"{'win_rate':22}{s.get('win_rate',0):>12.4f}{'-':>12}")
    print(f"{'sharpe/trade (ann)':22}{s.get('sharpe_per_trade_ann',0):>12.3f}{b.get('sharpe_per_trade_ann',0):>12.3f}")
    print(f"{'profit_factor':22}{s.get('profit_factor',0):>12.3f}")
    print(f"{'max_dd % (1% risk)':22}{s.get('max_drawdown_pct_at_1pct_risk',0):>12.3f}")
    print(f"{'final_eq (1% risk)':22}{s.get('final_equity_at_1pct_risk',0):>12.4f}")

    # Per-symbol breakdown of the taken trades.
    print("\n--- Per-symbol (taken trades) ---")
    tdf = pd.DataFrame([{"symbol": t.symbol, "pnl_r": t.pnl_r, "win": t.pnl_r > 0} for t in rep.trades])
    print(f"{'symbol':<8}{'n':>6}{'total_R':>10}{'mean_R':>9}{'win_rate':>10}")
    per_symbol = {}
    if len(tdf):
        for sym, g in tdf.groupby("symbol"):
            per_symbol[sym] = {"n": int(len(g)), "total_r": round(float(g["pnl_r"].sum()), 3),
                               "mean_r": round(float(g["pnl_r"].mean()), 4),
                               "win_rate": round(float(g["win"].mean()), 3)}
            print(f"{sym:<8}{len(g):>6}{g['pnl_r'].sum():>10.3f}{g['pnl_r'].mean():>9.4f}{g['win'].mean():>10.3f}")

    # Persist artifacts.
    outdir = Path(".triage")
    (outdir / "majors_donchian_strategy.pkl").write_bytes(pickle.dumps(strat))
    if len(rep.trades):
        pd.DataFrame([t.__dict__ for t in rep.trades]).to_csv(outdir / "paper_trades.csv", index=False)
        rep.equity_curve.to_csv(outdir / "paper_equity.csv")
    summary = {
        "slice": "majors_donchian", "sizing": "fixed_rung2", "tau0": strat.tau0,
        "n_events": len(df), "n_train": len(train), "n_test": len(test),
        "filtered": s, "take_all": b, "per_symbol": per_symbol,
        "uplift_total_r": round(s.get("total_r", 0) - b.get("total_r", 0), 3),
    }
    (outdir / "paper_majors_donchian_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nwrote .triage/majors_donchian_strategy.pkl, paper_trades.csv, "
          "paper_equity.csv, paper_majors_donchian_summary.json")


if __name__ == "__main__":
    main()
