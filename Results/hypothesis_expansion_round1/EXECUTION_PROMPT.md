# TRIAGE — Hypothesis Expansion Round: Execution Prompt

**Target:** Antigravity 2.0 (or any equivalent coding agent with repo + bash access **and** a
machine where the MT5 terminal can actually run/log in). This is *not* a fresh start — it is the
execution phase (Steps 3-6) of an already-pre-registered hypothesis round. Paste this whole file
as the agent's goal.

**MT5 data fetch is authorized for this run** — the user has explicitly approved fetching fresh
MT5 data this time (unlike a default session, which must stop and ask before touching a live
broker terminal, per `docs/PHASE1_DATA_AUDIT.md`'s convention). You do not need to re-ask
permission for the ingest itself. You DO need to stop and report if coverage comes back thin (see
§1's threshold).

Repo: `dukepsk-lab/TRIAGE`, branch `claude/triage-model-wz7d2g`. Commit + push frequently. Do not
open a PR unless asked.

---

## 0. Read first (source of truth — do not skip)

1. `docs/ARCHITECTURE.md` — full design; every decision below cites it.
2. **`Results/hypothesis_expansion_round1/hypotheses_registered.md`** — the six hypotheses this
   round exists to test (H1-H6). Every trigger rule, counterparty, barrier scheme, feature set,
   and HPO search space in it is **frozen** — it was written before any backtest ran, specifically
   so results can't leak back into the design. Do not edit trigger logic, barrier constants, or
   search-space bounds in that file after seeing results. If a genuine data-availability
   constraint forces narrowing scope, add a new dated note to that file explaining the decision —
   never silently delete or rewrite a spec.
3. `docs/PHASE1_RESULTS.md`, `PHASE2_RESULTS.md`, `PHASE3_RESULTS.md`, `MAJORS_DONCHIAN_RESULTS.md`
   — prior verdicts and the "Honest trial count" sections. **The cumulative honest trial count
   going into this round is 22** (`MAJORS_DONCHIAN_RESULTS.md`), carried from a checkout where
   `.triage/` did not exist. If this checkout's `.triage/` is *also* empty at the start of this
   run, seed the registry count from 22, not from zero — don't repeat the same undercount this
   repo has already documented twice.
4. `src/triage/gates.py` — exact G1-G5 formulas, reuse verbatim, do not reimplement.
5. `docs/AUTONOMOUS_GOAL.md` — the established data-ingest / registry / HPO conventions this round
   inherits (bounded model capacity, fixed purged folds, trial-budget discipline).

Internalize §0 of `ARCHITECTURE.md`: the meta-model is a precision dial, not an alpha source;
every tuned knob is a trial; an honest red gate is a better outcome than a green one reached by
cheating.

---

## 1. Environment + data

```bash
uv sync --extra dev
uv run pytest tests/ -q          # must be green before you start
```

Ingest H4 + M1 + D1 bars, ~2015 → today, UTC, for all six symbols:
`EURUSD, GBPUSD, USDJPY, XAUUSD, US30, BTCUSD`.

```python
from triage.data.mt5_ingest import ingest_symbol, coverage_report, TF_D1, TF_H4, TF_M1
# or the existing ingest_eurusd_phase1-style helper, extended to the other 5 symbols
for s in ["EURUSD","GBPUSD","USDJPY","XAUUSD","US30","BTCUSD"]:
    for tf in (TF_D1, TF_H4, TF_M1):
        print(coverage_report(s, tf))   # expect ~10y+, low gap_ratio, nonzero n
```

If MT5 is unreachable for a given symbol/timeframe, fall back to `scripts/ingest_csv.py` only if
the user has supplied raw files — do not fabricate bars. **Stop and report** (don't proceed to
training) if any symbol comes back under ~5 years of H4 history or with a severe gap ratio — same
threshold `PHASE1_DATA_AUDIT.md` used.

**Data this round needs beyond plain OHLC — source or explicitly defer, never fake:**
- **H1, H5** — OHLC only. Fully runnable immediately once ingest completes.
- **H3** — OHLC plus a short, hand-curated, pre-registered (i.e. decided *before* seeing which
  ones "worked") static list of historically documented USDJPY intervention dates/levels (public
  record, e.g. 2022 and 2024 BoJ actions). This is a research task, not a live feed — do it once,
  freeze the list, cite the source.
- **H2** — needs an ECB/BoE scheduled-decision calendar (dates + surprise direction). Not
  available from MT5 bars. Source from a calendar provider if you have one; otherwise mark H2
  **deferred — data unavailable** in the deliverables, don't approximate event dates from price
  action.
- **H4** — needs a futures-roll/expiry calendar for the US30 CFD/futures proxy. Same treatment as
  H2 if unavailable: defer, don't guess roll dates.
- **H6** — needs BTCUSD perpetual-futures funding-rate history, which MT5 does not provide (MT5
  gives you spot/CFD OHLC, not exchange funding data). If you cannot source a real funding-rate
  series, mark H6 **deferred — data unavailable**. Do not proxy funding rate with price momentum;
  that would silently turn it back into a direction-prediction hypothesis, which is explicitly
  closed (§1.5 of the original round prompt).

Report explicitly, in the deliverables, which of H1-H6 actually ran this round and which were
deferred for data reasons — that split is itself part of an honest report, not a failure to hide.

---

## 2. Implementation (one hypothesis at a time, cheapest-data-first: H1, H5, H3, then H2/H4/H6 if sourced)

- New rule module per hypothesis under `src/triage/rules/`, following the exact conventions of
  `donchian.py` / `pullback.py`: frozen `@dataclass` config, `signal_at_bar()` / `backfill_signals()`
  point-in-time closed-bar functions, reuse `session_mask()`, a distinct `rule_version` tag per
  hypothesis so events attribute correctly.
- One shared feature-computation path — extend `src/triage/features/service.py` (or a sibling
  module it imports) with the new "Tier 5" hypothesis-specific columns named in
  `hypotheses_registered.md` (e.g. `sweep_depth_atr`, `days_since_event`,
  `distance_to_intervention_level_atr`, `overnight_gap_atr`, `funding_rate_zscore`, ...). Do not
  fork a parallel feature pipeline — backtest and any future live path must share this code.
- Correlation-prune + MDA-gate every new Tier-5 feature exactly like `tier4.py::mda_selection`
  before trusting it (permute within OOS purged folds, keep only if the CI on the drop in mean
  OOS trade EV excludes zero).
- Unit test per new rule mirroring `tests/unit/test_donchian_parity.py` (synthetic bars with a
  known, hand-computed expected trigger) *before* running it against real ingested data. Keep
  `uv run pytest tests/ -q` green throughout.

---

## 3. Validation

For each hypothesis that has real data (per §1's split):

1. Backfill events (idempotent — skip already-backfilled symbol/rule pairs), M1 first-touch
   labeled per that hypothesis's frozen triple-barrier scheme from `hypotheses_registered.md`.
2. **Leakage battery, blocking:** shuffled-label OOS AUC ∈ [0.45, 0.55]
   (`triage.leakage.assert_no_leakage` if present, else replicate the shuffled-label harness
   pattern in `tests/unit/test_shuffled_label.py`), plus `features/audit.no_lookahead`. If this
   fails for a hypothesis, **stop and fix the leak for that hypothesis** before trusting any gate
   on it — do not proceed to gates on a leaking feature set.
3. **HPO** — use the exact bounded grid already frozen per-hypothesis in
   `hypotheses_registered.md` (≤ 40 configs, fixed purged folds shared across all trials of that
   hypothesis — re-randomizing folds per trial is a leak). If Optuna gets installed, it may run
   the same frozen bounds; do not widen them because a nicer optimizer is available.
4. **Gates** — `triage.gates.compute_gates` per hypothesis × instrument-arm, full CPCV (N=6, k=2 →
   15 splits, 5 paths) for a real PBO, DSR at the **honest cumulative registry count** (≥ 22 + every
   trial this round adds — every HPO config, every calibration variant, every hypothesis attempt,
   pass or fail, is one trial; log it).
5. Every run — pass, fail, or error — goes through `logged_trial` / `trial_context` against
   `.triage/registry.sqlite`. Never delete, prune, or reset it to lower the count.

---

## 4. Five Sins audit (Step 5 — orthogonal to the gates, run regardless of gate outcome)

For every hypothesis that reaches a gate verdict, check explicitly and report even if gates
passed:
- **Look-ahead** — implausibly smooth equity curve, suspiciously high hit rate vs. the base-rate
  geometry in `ARCHITECTURE.md` §1.
- **Survivorship** — did the instrument/date range selection quietly exclude inconvenient periods?
- **Narrative-without-mechanism** — does the counterparty story in `hypotheses_registered.md` still
  hold up against the actual data, or did the edge come from somewhere the story doesn't explain?
- **Proxy-objective optimization** — did HPO/feature selection optimize AUC/accuracy in a way that
  drifts from the strategy-relevant metric (mean OOS trade EV)?
- **Unrealistic costs** — re-check `COST_BY_SYMBOL` against this hypothesis's actual expected
  holding pattern (e.g. H2/H3's multi-day holds may cross more spread/swap events than the
  existing per-symbol cost constants assume — flag if so, don't silently under-cost).

A flagged sin with passing gates → note prominently, lean toward kill, same as the base prompt's
instruction.

---

## 5. Update the trial registry (Step 6)

Every hypothesis attempted this round — deferred, killed, or promoted — gets a permanent,
never-deleted entry. Deferred-for-data-reasons hypotheses (H2/H4/H6 if unresolved) are **not**
"trials" in the DSR/PBO sense (nothing was fit or gated) — record them as such explicitly in the
written report, but do not log a fake `logged_trial` row for work that never ran.

---

## 6. Deliverables

In `Results/hypothesis_expansion_round1/` (continue this round's folder — do not start a new one):

- `hypotheses_registered.md` — already committed; leave the specs untouched, only append a dated
  note if scope had to narrow for data reasons.
- `gate_report.md` — one table per hypothesis × instrument-arm: all gate values, verdict, 2-3
  sentence interpretation. Include a row for each deferred hypothesis explaining why.
- `gate_metrics.csv` — machine-readable version of the same.
- `five_sins_audit.md` — flags per surviving candidate.
- `trial_registry_updated.csv` — full registry dump after this round, with the new cumulative N.
- Plain-language top-level summary in your final reply: which hypotheses ran, which were deferred
  and why, which failed which gate and on what metric, which passed and are ready for MQL5 parity
  check / paper trading. Report kills as prominently as passes — do not cherry-pick a clean-looking
  summary.

---

## 7. Hard constraints (do NOT — same honesty guardrails as the rest of this repo)

- Do NOT delete, prune, or reset `.triage/registry.sqlite` to lower the trial count.
- Do NOT edit any already-frozen constant — Donchian/pullback's 1.5/1.0/24 barriers, `COST_BY_SYMBOL`,
  or any H1-H6 barrier/threshold already fixed in `hypotheses_registered.md` — to manufacture a
  pass. A genuinely new constant is a new pre-registered trial, documented, not a silent edit.
- Do NOT widen an HPO search space, add features, or extend a vertical barrier after seeing OOS
  results.
- Do NOT fabricate, approximate, or "reasonably estimate" data for H2/H4/H6 if a real source isn't
  available — defer them, explicitly, in the report.
- Do NOT curate which instruments/hypotheses appear in the final report to make it look cleaner.
- Do NOT re-randomize CV folds per trial.
- Do NOT recompute features from a full-history frame instead of the persisted point-in-time
  snapshots.

## 8. Stop / escalate

- Stop and write up a hypothesis the moment its gate verdict (pass or fail) is reached — extra
  trials on an already-decided hypothesis only degrade DSR/PBO for everyone else in the registry.
- Stop and report immediately if the leakage battery fails and can't be fixed quickly.
- Stop and report if MT5 coverage is thin (<~5y) for any symbol before training on it.
- Leave a clear final commit summarizing, per hypothesis: ran / deferred / killed / promoted, and
  if anything is unresolved, the single most useful next decision for a human to make.
