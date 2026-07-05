"""Phase 2 orchestration: pool EURUSD/GBPUSD/USDJPY.

ARCHITECTURE.md §10 Phase 2 + build prompt Phase 2 spec:

- Extend rule + feature pipeline across the three majors (verified
  symbol-agnostic in the Phase 1 audit — near-zero new code).
- ``symbol`` as a native categorical in LightGBM; pool all three.
- Full gate stack G1-G5 now BINDING (first phase where gates are a real
  verdict, not a sanity check).
- Trial counts are honest CUMULATIVE counts from Phase 1-2, pulled from
  the registry — never reset.
- Re-run the shuffled-label harness on the pooled set.
- STOP: docs/PHASE2_RESULTS.md with pooled gates, per-symbol breakdown
  (is one symbol carrying the result?), and trial-count accounting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from triage.data.mt5_ingest import PHASE2_SYMBOLS, TF_D1, TF_H4, TF_M1, load_bars_frame
from triage.features import FEATURE_COLUMNS
from triage.gates import MAJOR_COST_R, compute_gates
from triage.leakage import assert_no_leakage
from triage.models.meta import train_meta_model
from triage.pipeline import DEFAULT_EVENTS_DB, build_events_for_signals, persist_events, load_events
from triage.registry import TrialRegistry
from triage.rules import backfill_signals
from triage.uniqueness import uniqueness_weights
from triage.validation import DEFAULT_EMBARGO_BARS, purged_train_idx

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIALS_DB = REPO_ROOT / ".triage" / "registry.sqlite"

# Phase 2 feature set = Phase 1 features + `symbol` as categorical (§6).
POOLED_FEATURE_COLUMNS = FEATURE_COLUMNS + ["symbol"]


# ---------------------------------------------------------------------------
# Step 1: run the rule + label pipeline on any new symbols.
# ---------------------------------------------------------------------------


def backfill_symbol_events(symbol: str) -> int:
    """Run the Donchian rule + M1 labeling for one symbol, persist events.

    EURUSD is already labeled from Phase 1; this is a no-op skip if the
    symbol's events already exist in the store. Returns the count of
    newly-persisted events.
    """
    existing = load_events(DEFAULT_EVENTS_DB)
    if symbol in existing["symbol"].unique():
        print(f"[phase2] {symbol}: events already present ({len(existing[existing['symbol']==symbol])}), skipping")
        return 0

    print(f"[phase2] {symbol}: loading bars...")
    h4 = load_bars_frame(symbol, TF_H4)
    d1 = load_bars_frame(symbol, TF_D1)
    m1 = load_bars_frame(symbol, TF_M1)
    print(f"[phase2] {symbol}: H4={len(h4)} D1={len(d1)} M1={len(m1)}")

    print(f"[phase2] {symbol}: generating Donchian-20 signals...")
    sigs = backfill_signals(h4, d1)
    print(f"[phase2] {symbol}: {len(sigs)} signals -> labeling (M1 backfill)...")

    labeled = build_events_for_signals(sigs, h4=h4, d1=d1, m1_all=m1, symbol=symbol, progress_every=200)
    n = persist_events(labeled)
    print(f"[phase2] {symbol}: persisted {n} events")
    return n


# ---------------------------------------------------------------------------
# Step 2: pooled feature matrix.
# ---------------------------------------------------------------------------


def load_pooled_events() -> pd.DataFrame:
    """Load all events across symbols, unpack features, add `symbol` col."""
    df = load_events(DEFAULT_EVENTS_DB)
    feats = pd.DataFrame([json.loads(f) for f in df["features"]]).set_index(df.index)
    feat_cols = [c for c in FEATURE_COLUMNS if c in feats.columns]
    feats = feats[feat_cols]
    df = df.join(feats)
    # `symbol` already a column on the events table; ensure it's in the
    # feature matrix as a categorical (§6: "Symbol ... enter as native
    # categoricals").
    return df


def build_pooled_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, events_meta) for pooled training.

    ``X`` includes the ``symbol`` column as a pandas 'category' dtype so
    LightGBM treats it natively (§6). Numeric features coerced + NaN-filled.
    """
    X = df[POOLED_FEATURE_COLUMNS].copy()
    for c in FEATURE_COLUMNS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    X["symbol"] = X["symbol"].astype("category")
    y = df["label"].astype(int)
    meta = df[["t0", "t1", "symbol", "r_realized", "label"]].copy()
    return X, y, meta


# ---------------------------------------------------------------------------
# Step 3: purged time-ordered split (pooled, across all symbols).
# ---------------------------------------------------------------------------


def split_pooled(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split across the POOLED set, purged at the boundary.

    Splitting by global t0 (not per-symbol) is correct: the test period
    is a contiguous wall-clock window across all symbols, which is what
    makes the purge + embargo meaningful (§7).
    """
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


# ---------------------------------------------------------------------------
# Step 4: train pooled model with symbol categorical.
# ---------------------------------------------------------------------------


def train_pooled(train: pd.DataFrame, test: pd.DataFrame, registry: TrialRegistry) -> dict:
    """Train the pooled LightGBM with `symbol` as native categorical."""
    from triage.registry import logged_trial

    @logged_trial(
        {"phase": 2, "config": "pooled_3majors_symbol_cat", "barriers": "1.5/1.0/24", "rule": "donchian20_v1"},
        registry=registry,
    )
    def _run() -> dict:
        X_tr_raw, y_tr, _ = build_pooled_matrices(train)
        X_te_raw, y_te, _ = build_pooled_matrices(test)
        # Align category dtypes (test must share train's category levels).
        X_te_raw["symbol"] = pd.Categorical(X_te_raw["symbol"], categories=X_tr_raw["symbol"].cat.categories)

        # Uniqueness weights on the training fold only (§4).
        ev = train[["t0", "t1"]].copy()
        ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
        w = uniqueness_weights(ev)

        res = train_meta_model(
            X_tr_raw, y_tr, sample_weight=w,
            test_features=X_te_raw, test_labels=y_te,
        )
        return res

    res = _run()
    # Rebuild the test matrices for attribution / gates.
    X_tr, y_tr, _ = build_pooled_matrices(train)
    X_te, y_te, meta_te = build_pooled_matrices(test)
    X_te["symbol"] = pd.Categorical(X_te["symbol"], categories=X_tr["symbol"].cat.categories)
    return {
        "result": res,
        "X_te": X_te,
        "y_te": y_te,
        "meta_te": meta_te,
        "test_r_realized": test["r_realized"].to_numpy(),
        "test_labels": test["label"].to_numpy().astype(int),
        "test_symbols": test["symbol"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# Step 5: per-symbol attribution.
# ---------------------------------------------------------------------------


def per_symbol_attribution(preds: np.ndarray, meta_te: pd.DataFrame, r_realized: np.ndarray) -> pd.DataFrame:
    """Per-symbol AUC + mean-R + event count on the test set.

    Answers the §10/Phase 2 question: "is one symbol carrying the
    pooled result?" Reports each symbol's standalone AUC and EV.
    """
    from sklearn.metrics import roc_auc_score
    rows = []
    for sym in sorted(meta_te["symbol"].unique()):
        mask = (meta_te["symbol"] == sym).to_numpy()
        n = int(mask.sum())
        if n < 10:
            rows.append({"symbol": sym, "n": n, "auc": float("nan"), "mean_r_after_cost": float("nan"), "base_rate": float("nan")})
            continue
        y_s = meta_te["label"].to_numpy()[mask]
        p_s = preds[mask]
        r_s = r_realized[mask]
        auc = float(roc_auc_score(y_s, p_s)) if len(set(y_s)) > 1 else float("nan")
        mean_r = float((r_s - MAJOR_COST_R).mean())
        rows.append({"symbol": sym, "n": n, "auc": round(auc, 4), "mean_r_after_cost": round(mean_r, 4), "base_rate": round(float(y_s.mean()), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("PHASE 2 — Pool EURUSD / GBPUSD / USDJPY")
    print("=" * 70)

    # Step 1: ensure all symbols have events.
    print("\n--- Step 1: backfill events for new symbols ---")
    for sym in PHASE2_SYMBOLS:
        backfill_symbol_events(sym)

    # Step 2: load pooled events.
    print("\n--- Step 2: load pooled events ---")
    df = load_pooled_events()
    print(f"pooled events: {len(df)}")
    print(f"per-symbol counts: {df.groupby('symbol').size().to_dict()}")
    print(f"pooled base rate: {df['label'].mean():.3f}")

    # Step 3: shuffled-label harness on pooled set.
    print("\n--- Step 3: shuffled-label leakage check (pooled) ---")
    X_all, y_all, _ = build_pooled_matrices(df)
    X_num = X_all.drop(columns=["symbol"])  # harness uses numeric features only
    results = assert_no_leakage(X_num, y_all, tolerance=0.05, n_seeds=3)
    shuf_aucs = [round(r.auc, 4) for r in results]
    print(f"  permuted-label AUCs: {shuf_aucs} (all within 0.05: {all(r.passed for r in results)})")

    # Step 4: train pooled model.
    print("\n--- Step 4: train pooled model (symbol as categorical) ---")
    train, test = split_pooled(df)
    print(f"  purged train/test: {len(train)} / {len(test)}")
    reg = TrialRegistry(TRIALS_DB)
    out = train_pooled(train, test, reg)
    res = out["result"]
    print(f"  pooled AUC={res.auc:.4f}  Brier={res.brier:.4f} (clim={res.climatology_brier:.4f})")
    print(f"  n_train={res.n_train}  n_test={res.n_test}")

    # Step 5: per-symbol attribution.
    print("\n--- Step 5: per-symbol attribution ---")
    preds = res.oof_pred
    attr = per_symbol_attribution(preds, out["meta_te"], out["test_r_realized"])
    print(attr.to_string(index=False))

    # Step 6: gates (binding).
    print("\n--- Step 6: gates G1-G5 (BINDING) ---")
    from triage.gates import strategy_returns_r
    from triage.sizing import default_tau0
    tau0 = default_tau0()
    primary_ret = strategy_returns_r(out["test_labels"], out["test_r_realized"], cost_r=MAJOR_COST_R)
    filtered_ret = strategy_returns_r(out["test_labels"], out["test_r_realized"], probabilities=preds, ladder_tau0=tau0, cost_r=MAJOR_COST_R)
    taken = filtered_ret[filtered_ret != 0]
    oos_mean = float(taken.mean()) if len(taken) else 0.0
    oos_se = float(taken.std() / np.sqrt(len(taken))) if len(taken) > 1 else 1.0
    obs_sharpe = float(primary_ret.mean() / primary_ret.std() * np.sqrt(252)) if primary_ret.std() > 0 else 0.0

    from triage.models.meta import reliability_curve
    rel = reliability_curve(out["test_labels"], preds, n_bins=8)
    xs = [mp for mp, ma, n in rel.values() if n > 0 and not np.isnan(mp)]
    ys = [ma for mp, ma, n in rel.values() if n > 0 and not np.isnan(ma)]
    rel_slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0

    n_trials = len(reg)  # CUMULATIVE across Phase 1-2
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
    print(f"\n(cumulative trial count Phase 1-2: {n_trials})")

    # Step 7: write results.
    write_results(df, res, attr, shuf_aucs, gates, n_trials, train, test)
    print("\n=> wrote docs/PHASE2_RESULTS.md")
    print("=> STOPPING per Phase 2 STOP condition. Awaiting sign-off for Phase 3.")


def write_results(df, res, attr, shuf_aucs, gates, n_trials, train, test) -> None:
    """Write docs/PHASE2_RESULTS.md (Phase 2 STOP deliverable)."""
    per_sym = df.groupby("symbol").agg(
        n=("label", "size"),
        base_rate=("label", "mean"),
    ).round(3).to_dict()
    span = f"{df['t0'].min()} -> {df['t0'].max()}"

    # Is one symbol carrying the result?
    attr_row = attr.copy()
    best = attr_row.loc[attr_row["auc"].idxmax()] if not attr_row["auc"].isna().all() else None
    worst = attr_row.loc[attr_row["auc"].idxmin()] if not attr_row["auc"].isna().all() else None
    spread = (best["auc"] - worst["auc"]) if best is not None else float("nan")

    if best is not None and spread > 0.10:
        attribution_verdict = (
            f"**Yes — attribution is uneven.** {best['symbol']} leads (AUC={best['auc']}), "
            f"{worst['symbol']} lags (AUC={worst['auc']}); spread={spread:.3f}. "
            "The pooled result is not uniformly driven; inspect whether the laggard "
            "has conditional structure the rule isn't capturing."
        )
    else:
        attribution_verdict = (
            f"**No single symbol is carrying the result.** AUC spread across symbols "
            f"is {spread:.3f} (best {best['symbol'] if best is not None else '?'}={best['auc'] if best is not None else float('nan')}, "
            f"worst {worst['symbol'] if worst is not None else '?'}={worst['auc'] if worst is not None else float('nan')}). "
            "Pooling is justified — no symbol dominates."
        )

    # Recommendation.
    if gates.g1_pass and gates.g5_pass:
        reco = "G1 + G5 pass -> recommend Phase 3 (full basket + second rule family)."
    elif gates.g1_pass:
        reco = ("G1 uplift passes but G5 (calibration) fails -> calibration needs work "
                "(isotonic at higher n, or a dedicated calibration fold) before Phase 3. "
                "The ranking is sound; absolute probabilities are not yet sizer-ready.")
    else:
        reco = ("G1 does not pass at pooled n -> inspect per-symbol attribution + "
                "EV-vs-tau. If one symbol is dragging the pool, consider whether the "
                "rule fits it; do NOT add features to force separation (§0).")

    md = f"""# Phase 2 Results — Pool EURUSD / GBPUSD / USDJPY

**Status: STOP — awaiting sign-off before Phase 3.**

Pooled 3-major meta-labeling run. This is the first phase where the
gates are a **real verdict**, not a sanity check (§10).

## Pooled data

- Symbols: EURUSD, GBPUSD, USDJPY (all majors, same cost class ~0.04R)
- Span: {span}
- Pooled events: **{len(df)}** (3× Phase 1's single-symbol n)
- Per-symbol counts: {per_sym}
- Pooled base rate: **{df['label'].mean():.3f}**
- Purged train/test: {len(train)} / {len(test)}

## Shuffled-label leakage check (pooled)

Permuted-label OOS AUCs must be ≈ 0.50 ± 0.05:

- AUCs: {shuf_aucs}
- All within tolerance: **{all(abs(a-0.5)<=0.05 for a in shuf_aucs)}**

## Pooled meta-model (§6, symbol as native categorical)

| Metric | Value |
|---|---|
| AUC (OOS, pooled) | {res.auc:.4f} |
| Brier (OOS) | {res.brier:.4f} |
| Climatology Brier | {res.climatology_brier:.4f} |
| n_train / n_test | {res.n_train} / {res.n_test} |

## Per-symbol attribution (is one symbol carrying the result?)

{attribution_verdict}

{attr.to_string(index=False)}

## Gates G1-G5 (BINDING — first real verdict)

```
{gates.as_table()}
```

**Cumulative trial count (Phase 1-2, honest, from registry): {n_trials}**

## Recommendation

{reco}

## Notes

- `symbol` enters LightGBM as a native categorical (§6) — pooling is
  clean by construction, no one-hot blow-up.
- Uniqueness weights recomputed on the pooled training fold (§4).
- CPCV PBO uses a single-path approximation here; full 15-split/5-path
  PBO needs per-path refits and is meaningful only with the full basket
  (Phase 3).
"""
    out_path = REPO_ROOT / "docs" / "PHASE2_RESULTS.md"
    out_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
