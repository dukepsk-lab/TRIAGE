# Autonomous Goal — Full-Basket Ingest → Train → Gate

**Paste this as your agent goal.** You are running unattended (overnight).
Your job is to take the 6-symbol basket from raw bars to a **gate verdict**,
iterating *only* within the principled moves below, and stop the moment a
**suitable model** is found or the bounded search is exhausted. The whole
point of this architecture is rejection rigor — a green gate you reached by
cheating is worse than an honest red one.

Work on branch `claude/validate-architecture-docs-xx277i`. Commit + push
frequently. **Do not open a PR.**

---

## 0. Read first (source of truth)

1. `docs/ARCHITECTURE.md` — the full design. Every decision below cites it.
2. `docs/PHASE1_RESULTS.md`, `docs/PHASE2_RESULTS.md` — prior runs + format.
3. Recent commits on this branch — the purged holdout calibration (§6 fix),
   the cost-model single-source-of-truth, and the comparison harness.

Internalize the §0 design stance: **the meta-model is a precision dial, not
an alpha source; every tuned knob is a trial; pooling across symbols is the
plan of record.** And §11 "What falsifies this design" — you must be willing
to *reject*, not force a pass.

---

## 1. Environment + data

```bash
uv sync --extra dev
uv run pytest tests/ -q          # must be green before you start
```

Verify the bar store is populated (6 symbols, D1+H4+M1, ~2015→2026, UTC):

```python
from triage.data.mt5_ingest import coverage_report, TF_D1, TF_H4, TF_M1, ALL_SYMBOLS
for s in ALL_SYMBOLS:
    for tf in (TF_D1, TF_H4, TF_M1):
        print(coverage_report(s, tf))   # expect ~11y, low gap_ratio, nonzero n
```

If empty, ingest raw files first (replace `<RAW_DATA_DIR>`):

```bash
uv run python scripts/ingest_csv.py --dir <RAW_DATA_DIR>
```

Files are named `<SYMBOL>_<TF>.{csv,parquet}` (e.g. `EURUSD_M1.parquet`).
If a `spread` column is present, note the reported per-symbol median spread.

**Sanity gate before trusting anything:** the M1 backfill is the compute-heavy
job (§6) and is embarrassingly parallel per (symbol, year) — be patient, it
may take hours. Do not shortcut barrier resolution to H4; M1 first-touch is
the correct label (§4).

---

## 2. Definition of "suitable model" (STOP conditions)

Evaluate on the **full 6-symbol basket**, post-cost, with **holdout**
calibration (`train_meta_model(..., calibration="holdout")`).

**Leakage battery (blocking — if any fails, STOP and fix the leak, do not
proceed to gates):**
- Shuffled-label OOS AUC ∈ [0.45, 0.55] on the real feature matrix
  (`triage.leakage.assert_no_leakage`).
- Feature-timestamp audit passes (`triage.features.audit.no_lookahead`).
- Donchian backfill/live parity passes (`tests/unit/test_donchian_parity.py`).

**PRIMARY success (this is the answer we are after — stop here and write up):**
- **G1 (uplift) PASS** on the full basket. This is the gate that matters most
  (§7): if the meta-layer adds no uplift over take-all, the model is worthless
  even if other numbers look good.
- **G5 (calibration) PASS** on the full basket: Brier < climatology **and**
  reliability slope ∈ [0.8, 1.2].
- **Per-symbol attribution healthy** (§10): no single symbol carrying the
  book (modest AUC spread) and no single symbol's reliability slope is the
  lone outlier propping up the pass/fail.

**FULL success (promotable):** PRIMARY **plus** G2 (DSR ≥ 0.95 at the honest
registry trial count), G3 (PBO ≤ 0.20 via CPCV 5-path), G4 (t-stat ≥ 3.0).
Do not expect all five at this n — that is fine (§10).

**Hard trial budget: ≤ 60 honest trials for the whole night.** Every config
you run logs to `.triage/registry.sqlite` via `@logged_trial` and inflates
the DSR/PBO count. More search makes G2 *harder*, not easier. When you reach
PRIMARY success, **STOP searching** — additional trials only degrade DSR.

---

## 3. Pipeline (ordered)

1. **Backfill events** for all 6 symbols — Donchian-20 (§3) and htf_pullback
   (§10 second family), M1 first-touch labeled (§4). Reuse
   `scripts/run_phase3.py::backfill_symbol_events` (idempotent — skips symbols
   already present).
2. **Leakage battery** on the pooled real feature matrix. STOP if it fails.
3. **Feature selection** — correlation-prune (|ρ|>0.85), then **MDA under
   purged CV using the EV metric** (mean OOS trade EV), per §5. Cut Tier 4
   features whose MDA CI spans zero. MDI/gain is diagnostic only.
   - ⚠️ Known issue: `triage.features.tier4.mda_selection` defaults to
     `metric="auc"`, and its `"ev"` branch currently returns *mean predicted
     probability*, not realized trade EV. If you select on EV, **fix it to use
     `r_realized`-based trade EV first**; otherwise select on AUC and
     **document the deviation from §5** in the results.
4. **The head-to-head** — run the staged 2×2 (basket × calibration):
   ```bash
   uv run python scripts/run_basket_comparison.py
   ```
   This isolates the two variables (full6 vs curated3, cv vs holdout) over
   identical labels and prints G1–G5 + per-symbol slopes per cell.
5. **HPO (capped, coarse, fixed folds)** — Optuna/Ray are NOT installed;
   use a **small manual coarse grid** over the §6 capacity-bounded params on
   **fixed purged folds shared across all trials** (re-randomizing folds per
   trial is a leak). Stay inside the trial budget.
6. **Heavy gates for the run of record** — full CPCV (N=6, k=2 → 15 splits,
   5 paths) with **per-path refits** for a real PBO, and DSR at the honest
   registry count. Use `triage.validation.cpcv_splits` + `backtest_paths`.

---

## 4. HARD CONSTRAINTS — do NOT (these are the honesty guardrails)

- **Do NOT delete, prune, reset, or rewrite `.triage/registry.sqlite`** to
  lower the trial count. The honest count feeds DSR/PBO (§0, §7, §9). Lowering
  it is the exact self-deception the framework exists to prevent.
- **Do NOT curate the basket to pass a gate.** Full 6 is the run of record.
  Drop a symbol ONLY if its own per-symbol attribution/slope independently
  identifies it as an outlier — never because removing it nudges a gate green
  (§0, §10). The 2×2 exists to distinguish these.
- **Do NOT tune the frozen constants:** barriers 1.5/1.0/24, label policy
  (label=1 iff TP), rule params (Donchian-20, D1 EMA-50), embargo 48, the
  breakeven/cost formula. These are pre-registered (§0, §3, §4). The model's
  job is conditional-edge recovery, not rule/barrier tuning.
- **Do NOT edit gate thresholds, the shuffled-label tolerance, or the EV/cost
  math to manufacture a pass.** No moving goalposts.
- **Do NOT re-randomize CV folds per trial** (§6 — a leak into the search).
- **Do NOT recompute features from a full-history frame** — use the persisted
  point-in-time snapshots (§5).
- **Do NOT downgrade to H4 barrier resolution** to save time (§4).
- Cost model exception: if `spread` data is present you MAY update
  `gates.COST_BY_SYMBOL` to realized per-symbol medians **once**, as a
  documented, registry-logged pre-registered-constant change — then freeze it.

---

## 5. Allowed search space (bounded)

- Model capacity (§1/§6): `max_depth ≤ 6`, `num_leaves ≤ 31`,
  `min_child_samples = max(30, int(0.03·n_train))`, heavy `reg_lambda`,
  `feature_fraction`/`bagging_fraction` < 1, early stopping. Tiny models only.
- Coarse HPO grid over the above, ≤ ~40 configs, fixed folds, within budget.
- Calibration: `holdout` (isotonic ≥ 2k events, else sigmoid). You may vary
  `calib_frac` and isotonic-vs-sigmoid on the holdout fold — that is the
  §6-sanctioned way to fix G5.
- Rule families: `donchian` and `htf_pullback` (both, pooled). No new rules.
- Tier 4 features: only those MDA clears.
- Symbol / rule_id / session as native categoricals (§6).

---

## 6. Decision tree (when a gate fails — mirror §10 / §11)

- **G1 fails (flat p̂, no uplift):** the primary rule has no ML-recoverable
  edge on these features. **Swap the RULE, not the model** (§10). Try the
  other rule family pooled. If G1 still fails with flat p̂ across *both*
  families → §11 falsification ("H4 setups lack recoverable structure") →
  STOP and report the rejection. Do NOT add features to force separation.
- **G1 passes, G5 fails on full6:** run the 2×2. If a single symbol's slope
  is the clear outlier *and* attribution independently flags it → document a
  principled drop. Else the calibrator is the lever: try isotonic↔sigmoid,
  larger `calib_frac`, or per-symbol calibration. **Never drop symbols to
  force G5.**
- **G1+G5 pass, G2/G4 marginal:** expected at this n (§10). Report as
  "separates + real uplift; statistical gates marginal at n" — this is
  PRIMARY success. Stop.
- **CPCV passes but you suspect regime instability:** note it as the §11
  paper-trading risk; the parity tests get audited first.

---

## 7. Deliverables (commit to the branch, push, no PR)

- `docs/PHASE3_RESULTS.md` — full-basket verdict in the Phase 1/2 format:
  data span + per-symbol counts + base rate, leakage battery, headline
  meta-model (AUC/Brier/clim), EV-vs-τ, G1–G5 table, per-symbol attribution,
  honest trial count, recommendation.
- `docs/BASKET_COMPARISON.md` — the 2×2 table + per-symbol slopes + the
  explicit verdict: *did full6+holdout clear G5 (→ symbol-dropping was
  unnecessary), or did a per-symbol outlier justify a documented drop?*
- Keep `uv run pytest tests/ -q` green throughout; fix, don't skip.
- Commit messages: what changed + which gate/section it addresses.
- **Preserve `.triage/registry.sqlite`** and report `len(registry)` as the
  honest trial count in the results.

---

## 8. Stop / escalate

- Stop and write up the moment **PRIMARY success** is reached (do not keep
  searching — it only hurts DSR).
- Stop when the bounded decision tree is exhausted; write up the best honest
  result and the §11 falsification signature you observed.
- Stop if you hit the ≤60 trial budget; report where you are.
- If the leakage battery ever goes red, STOP everything and fix the leak
  before any gate is trustworthy.
- Leave a clear final commit summarizing the verdict and, if unresolved, the
  single most useful next decision for a human to make in the morning.
