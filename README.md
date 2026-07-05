# TRIAGE — Triple-Barrier Meta-Label Architecture

A rule-based primary layer + ML trade filter for H4 forex/metals/index
instruments. The full design is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
— **that document is the source of truth.** If anything in this repo
conflicts with it, the doc wins.

## Phase 0 — harness correctness (current)

This phase implements *only* the correctness harness described in
ARCHITECTURE.md §4 ("Correctness battery (Phase 0, blocking)") and §10
("Phase 0 — Harness correctness"). No rule engine, no real data, no
meta-model, no sizing, no execution — those are stubbed modules with
docstrings pointing at the relevant architecture section.

Phase 0 ships:

- **Synthetic M1 path generators** (`tests/synthetic/paths.py`) with
  analytically known first-touch outcomes, so labeler tests assert
  against hand-derived ground truth.
- **Triple-barrier labeler** (`src/triage/labeling/`) implementing
  `resolve()` from §4 exactly, including the same-bar → SL tie-break.
- **Purged K-fold + embargo + CPCV** (`src/triage/validation/`) per §7,
  with `purged_train_idx()` matching the doc's code sketch and CPCV at
  N=6, k=2 → 15 splits / 5 backtest paths.
- **Feature-timestamp audit** (`src/triage/features/`) — the
  no-look-ahead invariant enforced as an assertable function, not a
  comment.
- **Trial registry** (`src/triage/registry/`) — SQLite store +
  `@logged_trial(config)` decorator; every run writes a row regardless
  of pass/fail.
- **Shuffled-label leakage harness**
  (`tests/unit/shuffled_label_harness.py`) — a reusable function that
  trains plain LightGBM on permuted labels and asserts OOS AUC ≈ 0.50.

Barrier constants are **frozen** for this phase (TP = 1.5×ATR, SL =
1.0×ATR, vertical = 24 H4 bars). They are not free parameters; they
will be owned by `configs/rules/<rule>.yaml` in Phase 1 and remain
pre-registered there.

## Quick start

```bash
uv sync                       # create venv, install deps
uv run pytest                 # run the full correctness battery
```

Requires Python ≥ 3.11 and `uv`.

## Repo layout

```
docs/ARCHITECTURE.md             # source of truth (do not modify)
configs/rules/                   # empty until Phase 1
src/triage/
  data/                          # Bar/Side types (synthetic only in Phase 0)
  labeling/                      # resolve() + label policy + r_realized  (§4)
  validation/                    # purged K-fold + embargo + CPCV         (§7)
  features/                      # no_lookahead() audit (real) + stub     (§5)
  registry/                      # SQLite trial store + @logged_trial     (§0)
  rules/ models/ sizing/ execution/   # stubs only this phase
tests/
  synthetic/paths.py             # M1 generators with known outcomes
  unit/                          # pytest battery
```

Phase 0 exit criteria and open questions are in
[`docs/PHASE0_NOTES.md`](docs/PHASE0_NOTES.md).
