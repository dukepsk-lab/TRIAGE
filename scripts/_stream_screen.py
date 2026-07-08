"""Per-stream G1 screen — does each Phase 3 ADDITION clear G1 on its own?

PHASE3_RESULTS.md recommends: make each added stream clear G1 alone before
pooling. This screens the additions that had not been individually tested —
{XAU, US30, BTC} x donchian, and the pullback rule on the majors — against the
already-known anchors (majors x donchian pooled: G1 p=0.016; full6 x pullback:
G1 p=0.075). Headline meta-model, holdout, d4/15, one fixed purged split each.
Every stream is logged as an honest trial (§0).

The informative metric here is AUC (ranking edge) as much as G1: single-stream n
is small, so G1 is under-powered — but AUC>~0.53 vs ~0.50 cleanly separates
"has edge to pool" from "does not".
"""
from __future__ import annotations

from triage.registry import TrialRegistry
from run_phase3 import TRIALS_DB, load_pooled_events
from _diag import CAPS, run_cell

FULL6 = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "BTCUSD"]
MAJORS3 = ["EURUSD", "GBPUSD", "USDJPY"]

STREAMS = [
    ("XAUUSD/donchian", ["XAUUSD"], ["donchian20"]),
    ("US30/donchian", ["US30"], ["donchian20"]),
    ("BTCUSD/donchian", ["BTCUSD"], ["donchian20"]),
    ("majors3/pullback", MAJORS3, ["htf_pullback"]),
]


def main():
    reg = TrialRegistry(TRIALS_DB)
    df_all = load_pooled_events(FULL6)
    print(f"{'stream':<18}{'n':>6}{'AUC':>8}{'slope':>8}{'G1dS':>8}{'G1p':>7}  G1  per-symbol AUC")
    for name, syms, rules in STREAMS:
        r = run_cell(df_all, syms, rules, CAPS["d4/15"], "d4/15", "holdout", reg)
        if r is None:
            print(f"{name:<18}  (skipped — too few events)")
            continue
        print(f"{name:<18}{r['n']:>6}{r['auc']:>8.4f}{r['slope']:>8.3f}"
              f"{r['g1_ds']:>8.3f}{r['g1_p']:>7.3f}  {'Y' if r['g1'] else '.'}  {r['psym']}")
    print(f"\nregistry trial count now: {len(reg)}")


if __name__ == "__main__":
    main()
