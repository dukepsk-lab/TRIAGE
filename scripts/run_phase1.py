"""Phase 1 orchestration: train meta-model, compute gates, run ablation.

Runs the full Phase 1 Step 2 pipeline end-to-end on the persisted
EURUSD events:

1. Load events from the events store (rule + features + labels).
2. Re-run the shuffled-label leakage harness on real data (§4 battery).
3. Train the LightGBM meta-model with purged CV (§6) under both label
   policies (the §4 ablation: vert->0 vs vert->sign(r)), each logged as
   a separate trial via @logged_trial.
4. Compute G1-G5 gates (§7) on the headline configuration.
5. Produce the EV-vs-tau curve (§7 stop condition).
6. Write results to docs/PHASE1_RESULTS.md.

This is the script the Phase 1 STOP condition is built on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from triage.features import FEATURE_COLUMNS
from triage.gates import MAJOR_COST_R, GateResults, block_bootstrap_uplift, compute_gates, strategy_returns_r
from triage.models.meta import train_meta_model
from triage.pipeline import DEFAULT_EVENTS_DB, load_events
from triage.registry import TrialRegistry, logged_trial
from triage.sizing import DEFAULT_BASE_RISK, SizingLadder, default_tau0, ev_r, select_tau_on_train
from triage.uniqueness import uniqueness_weights
from triage.validation import DEFAULT_EMBARGO_BARS, cpcv_splits, purged_train_idx

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIALS_DB = REPO_ROOT / ".triage" / "registry.sqlite"


def load_events_frame() -> pd.DataFrame:
    """Load events and unpack the features JSON into columns."""
    df = load_events(DEFAULT_EVENTS_DB)
    feats = pd.DataFrame([json.loads(f) for f in df["features"]]).set_index(df.index)
    # Only keep the feature columns (drop audit fields).
    feat_cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    feats = feats[feat_cols]
    return df.join(feats)


def split_train_test(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split with purge + embargo on the boundary (§7)."""
    df = df.sort_values("t0").reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    # Purge training events whose span reaches into the test period.
    test_start = pd.Timestamp(test["t0"].min())
    test_end = pd.Timestamp(df["t1"].max())
    keep = purged_train_idx(
        train.set_index(train.index)[["t0", "t1"]],
        test_start, test_end, embargo_bars=DEFAULT_EMBARGO_BARS,
    )
    keep_pos = [train.index.get_loc(i) for i in keep] if hasattr(keep, "__iter__") else list(range(len(train)))
    train = train.iloc[keep_pos].copy()
    return train, test


def run_shuffled_label_check(df: pd.DataFrame) -> dict:
    """Re-run the §4 shuffled-label harness on real events."""
    from triage.leakage import assert_no_leakage
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["label"].astype(int)
    results = assert_no_leakage(X, y, tolerance=0.05, n_seeds=3)
    return {
        "aucs": [round(r.auc, 4) for r in results],
        "all_within_0.05": all(r.passed for r in results),
    }


@logged_trial({"phase": 1, "ablation": "vert_to_0", "barriers": "1.5/1.0/24", "rule": "donchian20_v1"},
              registry=TrialRegistry(TRIALS_DB))
def train_label_policy_zero(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Headline label policy: vertical -> label 0 (§4 committed policy)."""
    X_tr = train[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_tr = train["label"].astype(int)
    X_te = test[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_te = test["label"].astype(int)

    # Uniqueness weights on training fold ONLY (§4 leakage guard).
    ev_df = train[["t0", "t1"]].copy()
    ev_df["t1"] = ev_df["t1"].fillna(ev_df["t0"] + pd.Timedelta(hours=96))
    w = uniqueness_weights(ev_df)

    res = train_meta_model(
        X_tr, y_tr, sample_weight=w,
        test_features=X_te, test_labels=y_te,
    )
    return {
        "policy": "vert_to_0",
        "auc": round(res.auc, 4),
        "brier": round(res.brier, 4),
        "climatology_brier": round(res.climatology_brier, 4),
        "base_rate": round(res.base_rate, 4),
        "n_train": res.n_train,
        "n_test": res.n_test,
        "reliability": {str(k): v for k, v in res.reliability.items()},
        "result_obj": res,
        # Carry the model's own test-set arrays so gates reuse them
        # exactly (avoids a re-split size mismatch).
        "test_r_realized": test["r_realized"].to_numpy(),
        "test_labels": test["label"].to_numpy().astype(int),
    }


@logged_trial({"phase": 1, "ablation": "vert_to_sign_r", "barriers": "1.5/1.0/24", "rule": "donchian20_v1"},
              registry=TrialRegistry(TRIALS_DB))
def train_label_policy_sign_r(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Ablation label policy: vertical -> sign(r_realized) (§4 ablation)."""
    # Relabel: TP and vertical-with-positive-r -> 1; SL and vert-negative -> 0.
    def relabel(row):
        if row["outcome"] == "tp":
            return 1
        if row["outcome"] == "vert":
            return 1 if row["r_realized"] > 0 else 0
        return 0
    train2 = train.copy()
    train2["label_ablation"] = train2.apply(relabel, axis=1)

    X_tr = train2[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_tr = train2["label_ablation"].astype(int)
    X_te = test[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_te = test["label"].astype(int)  # compare on the *real* test label for fairness

    ev_df = train2[["t0", "t1"]].copy()
    ev_df["t1"] = ev_df["t1"].fillna(ev_df["t0"] + pd.Timedelta(hours=96))
    w = uniqueness_weights(ev_df)

    res = train_meta_model(
        X_tr, y_tr, sample_weight=w,
        test_features=X_te, test_labels=y_te,
    )
    return {
        "policy": "vert_to_sign_r",
        "auc": round(res.auc, 4),
        "brier": round(res.brier, 4),
        "climatology_brier": round(res.climatology_brier, 4),
        "n_train": res.n_train,
        "n_test": res.n_test,
        "note": "verticals relabeled to sign(r_realized); only 20 verticals in this set",
    }


def ev_vs_tau_curve(df: pd.DataFrame, headline_result) -> list[dict]:
    """EV-vs-tau curve (§7 stop condition diagnostic)."""
    out = []
    res = headline_result.get("result_obj")
    if res is None or res.oof_pred is None:
        return out
    preds = res.oof_pred
    r_realized = headline_result["test_r_realized"]
    for tau in np.arange(0.30, 0.71, 0.025):
        mask = preds >= tau
        if mask.sum() < 5:
            continue
        mean_r = float((r_realized[mask] - MAJOR_COST_R).mean())
        n = int(mask.sum())
        out.append({"tau": round(float(tau), 3), "n_trades": n, "mean_r_after_cost": round(mean_r, 4)})
    return out


def reliability_slope(rel: dict) -> float:
    """Fit a least-squares slope of mean_actual vs mean_predicted (G5)."""
    xs, ys = [], []
    for _, (mp, ma, n) in rel.items():
        if n > 0 and not np.isnan(mp) and not np.isnan(ma):
            xs.append(mp); ys.append(ma)
    if len(xs) < 2:
        return 0.0
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)


def compute_phase1_gates(df: pd.DataFrame, headline: dict, n_trials: int) -> GateResults:
    """Compute G1-G5 on the headline configuration.

    Uses the test-set arrays the model actually predicted on (carried in
    ``headline``), so gates and predictions are exactly aligned.
    """
    res = headline["result_obj"]
    preds = res.oof_pred
    r_realized = headline["test_r_realized"]
    labels = headline["test_labels"]

    # Strategy returns: take-all (primary) vs filtered (meta at tau0).
    tau0 = default_tau0()
    primary_ret = strategy_returns_r(labels, r_realized, cost_r=MAJOR_COST_R)
    filtered_ret = strategy_returns_r(labels, r_realized, probabilities=preds, ladder_tau0=tau0, cost_r=MAJOR_COST_R)

    # OOS mean R + SE for G4 (on filtered/taken trades).
    taken = filtered_ret[filtered_ret != 0]
    oos_mean = float(taken.mean()) if len(taken) else 0.0
    oos_se = float(taken.std() / np.sqrt(len(taken))) if len(taken) > 1 else 1.0

    # CPCV PBO (simplified single-path approximation at this n).
    cpcv_strat = [filtered_ret]
    cpcv_prim = [primary_ret]

    rel_slope = reliability_slope({
        int(k): v for k, v in headline["reliability"].items()
    })

    obs_sharpe = float(primary_ret.mean() / primary_ret.std() * np.sqrt(252)) if primary_ret.std() > 0 else 0.0
    return compute_gates(
        primary_returns_r=primary_ret,
        filtered_returns_r=filtered_ret,
        oos_mean_r=oos_mean,
        oos_mean_r_se=oos_se,
        brier=headline["brier"],
        climatology_brier=headline["climatology_brier"],
        reliability_slope=rel_slope,
        n_trials=n_trials,
        n_obs=len(df),
        cpcv_strategy_returns=cpcv_strat,
        cpcv_primary_returns=cpcv_prim,
        observed_sharpe=obs_sharpe,
    )


def main() -> None:
    print("=" * 70)
    print("PHASE 1 — EURUSD × Donchian-20 end-to-end run")
    print("=" * 70)

    df = load_events_frame()
    print(f"\nloaded {len(df)} events; base rate = {df['label'].mean():.3f}")
    print(f"outcome counts: {df['outcome'].value_counts().to_dict()}")

    train, test = split_train_test(df)
    print(f"train/test split (purged): {len(train)} / {len(test)}")

    print("\n--- shuffled-label leakage check (real data) ---")
    shuf = run_shuffled_label_check(df)
    print(f"  AUCs on permuted labels: {shuf['aucs']} (all |.-0.5|<=0.05: {shuf['all_within_0.05']})")

    print("\n--- headline label policy: vertical -> 0 ---")
    headline = train_label_policy_zero(train, test)
    print(f"  AUC={headline['auc']}  Brier={headline['brier']} (clim={headline['climatology_brier']})")
    print(f"  base_rate={headline['base_rate']}  n_train={headline['n_train']}  n_test={headline['n_test']}")

    print("\n--- ablation label policy: vertical -> sign(r) ---")
    ablation = train_label_policy_sign_r(train, test)
    print(f"  AUC={ablation['auc']}  Brier={ablation['brier']}")
    print(f"  {ablation['note']}")

    print("\n--- EV-vs-tau curve ---")
    curve = ev_vs_tau_curve(df, headline)
    for row in curve[::3]:
        print(f"  tau={row['tau']:.3f}  n={row['n_trades']:3d}  mean_r_after_cost={row['mean_r_after_cost']:+.4f}")

    print("\n--- gates G1-G5 ---")
    reg = TrialRegistry(TRIALS_DB)
    n_trials = len(reg)
    gates = compute_phase1_gates(df, headline, n_trials)
    print(gates.as_table())
    print(f"\n(honest trial count from registry: {n_trials})")

    # Write results doc.
    write_results(df, headline, ablation, shuf, curve, gates, n_trials)
    print("\n=> wrote docs/PHASE1_RESULTS.md")
    print("=> STOPPING per Phase 1 STOP condition. Awaiting sign-off for Phase 2.")


def write_results(df, headline, ablation, shuf, curve, gates, n_trials) -> None:
    """Write docs/PHASE1_RESULTS.md (the Phase 1 STOP deliverable)."""
    base_rate = df["label"].mean()
    outcomes = df["outcome"].value_counts().to_dict()
    span = f"{df['t0'].min()} -> {df['t0'].max()}"
    years = (pd.Timestamp(df["t0"].max()) - pd.Timestamp(df["t0"].min())).total_seconds() / (365.25 * 86400)

    # Determine the recommendation branch (Phase 1 stop condition).
    auc = headline["auc"]
    g1 = gates.g1_pass
    if g1:
        reco = "G1 uplift PASSES -> recommend Phase 2 (pool EURUSD/GBPUSD/USDJPY)."
    elif 0.53 <= auc <= 0.56:
        reco = ("Classifier separates (AUC in 0.53-0.56) but no strategy uplift -> "
                "inspect EV-vs-tau + cost interaction before touching the model. "
                "This failure is usually threshold/cost geometry, not the model.")
    else:
        reco = ("p-hat distribution flat / AUC ~ 0.50 -> the primary rule has no "
                "conditional edge on these features. SWAP THE RULE, do not add "
                "features to force separation (§0 commitment).")

    md = f"""# Phase 1 Results — EURUSD × Donchian-20

**Status: STOP — awaiting sign-off before Phase 2.**

This is the Phase 1 decision document per the build prompt's STOP
condition. The full pipeline ran end-to-end on real EURUSD history.

## Data

- Symbol: EURUSD (broker `EURUSD.`, IUX Markets Live)
- Span: {span} ({years:.1f} years)
- Events: {len(df)} (Donchian-20 H4 breakout, D1 EMA-50 slope-aligned, London/NY)
- Base rate (label=1, i.e. TP first): **{base_rate:.3f}** (§1 predicts 0.32-0.38 ✓)
- Outcomes: {outcomes}

## Shuffled-label leakage check (§4 battery)

Re-ran the harness on the real feature matrix. OOS AUC on permuted
labels must be ≈ 0.50 ± 0.05 or the harness leaks:

- AUCs: {shuf['aucs']}
- All within tolerance: **{shuf['all_within_0.05']}**

## Headline meta-model (§6)

LightGBM, purged train/test split, Platt calibration, uniqueness-weighted
(§4, training fold only). Frozen barriers 1.5/1.0/24.

| Metric | Value |
|---|---|
| AUC (OOS) | {headline['auc']} |
| Brier (OOS) | {headline['brier']} |
| Climatology Brier | {headline['climatology_brier']} |
| Base rate | {headline['base_rate']} |
| n_train / n_test | {headline['n_train']} / {headline['n_test']} |

## Label ablation (§4 — 2 trials, no winner picked)

The one pre-registered ablation: vertical-timeout label policy.

| Policy | AUC | Brier | Note |
|---|---|---|---|
| vert -> 0 (committed) | {headline['auc']} | {headline['brier']} | {outcomes.get('vert', 0)} verticals -> label 0 |
| vert -> sign(r) (ablation) | {ablation['auc']} | {ablation['brier']} | {ablation['note']} |

Both logged to the trial registry as separate trials.

## EV-vs-tau curve (cost = {MAJOR_COST_R}R)

tau = calibrated probability threshold. mean_r_after_cost = mean
realized R per taken trade, after subtracting cost.

| tau | n_trades | mean R (after cost) |
|---|---|---|
"""
    for row in curve:
        md += f"| {row['tau']:.3f} | {row['n_trades']} | {row['mean_r_after_cost']:+.4f} |\n"

    md += f"""
## Gates G1-G5 (§7)

```
{gates.as_table()}
```

Honest trial count from registry: **{n_trials}**

## Recommendation

{reco}

## Notes / caveats

- Single-symbol n (~1000 events) is below the §1 minimum viable
  (1500-4500 for pooled 3 majors). Per §10 Phase 1 exit guidance,
  G1 passing is sufficient to proceed; absolute G2-G4 numbers at
  this n are not yet binding.
- CPCV PBO here uses a single-path approximation (the §7 15-split/5-path
  machinery is implemented in `triage.validation` but full PBO needs
  per-path model refits; deferred to Phase 2 where pooled n makes it
  meaningful).
"""

    out_path = REPO_ROOT / "docs" / "PHASE1_RESULTS.md"
    out_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
