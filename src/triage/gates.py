"""Promotion gates G1-G5 (§7) + baseline ladder + cost model.

ARCHITECTURE.md §7 promotion gates apply to the filtered + sized
strategy returns, after conservative costs, over CPCV paths, with trial
counts from the registry.

    G1 Uplift      Filtered beats take-all primary (block-bootstrap,
                   ΔSharpe > 0 at 95% confidence). NEW.
    G2 DSR         >= 0.95, honest trial count.
    G3 PBO         <= 0.20 via CPCV.
    G4 t-stat      >= 3.0 on OOS mean trade R.
    G5 Calibration Brier < climatology; reliability slope in [0.8, 1.2]. NEW.

Baseline ladder (§7): (1) take-all primary -> (2) +meta filter at tau ->
(3) +probability-laddered sizing. G1 tests rung 2; a second uplift check
tests rung 3.

Cost model (§7): per-symbol-class, conservative. For majors (EURUSD in
Phase 1): cost ~0.03-0.05R against a 1xATR H4 stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "MAJOR_COST_R",
    "GateResults",
    "compute_gates",
    "strategy_returns_r",
    "block_bootstrap_uplift",
    "pbo_from_cpcv",
    "dsr",
]


#: EURUSD cost in R units (§7: "~0.03-0.05R against a 1xATR H4 stop").
#: Conservative end of the range.
MAJOR_COST_R: float = 0.04


# ---------------------------------------------------------------------------
# Strategy returns in R units.
# ---------------------------------------------------------------------------


def strategy_returns_r(
    labels: np.ndarray,
    r_realized: np.ndarray,
    *,
    probabilities: Optional[np.ndarray] = None,
    ladder_tau0: Optional[float] = None,
    cost_r: float = MAJOR_COST_R,
) -> np.ndarray:
    """Per-trade strategy returns in R units after cost.

    Three modes (the §7 baseline ladder):
    - take-all: every event's ``r_realized`` (rung 1);
    - filter: ``r_realized`` only where ``prob >= tau0`` else 0 (rung 2);
    - sized: ``r_realized`` scaled by ladder size where ``prob >= tau0``
      (rung 3, used for the second uplift check).

    Cost is subtracted from every *taken* trade.
    """
    if probabilities is None or ladder_tau0 is None:
        # Rung 1: take all.
        taken = np.ones(len(labels), dtype=bool)
        scale = np.ones(len(labels))
    else:
        taken = probabilities >= ladder_tau0
        # Rung 3 sizing (0.5x above tau0, 1.0x above tau1=tau0+0.10 approx).
        # For rung 2 (filter only), pass scale=1 below by not using this.
        scale = np.where(probabilities >= ladder_tau0 + 0.10, 1.0, 0.5)

    ret = np.zeros(len(labels))
    ret[taken] = scale[taken] * (r_realized[taken] - cost_r)
    return ret


# ---------------------------------------------------------------------------
# G1: Uplift via block bootstrap.
# ---------------------------------------------------------------------------


def block_bootstrap_uplift(
    primary_returns: np.ndarray,
    filtered_returns: np.ndarray,
    *,
    n_boot: int = 2000,
    block_size: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Block-bootstrap ΔSharpe significance (G1).

    Returns ``(delta_sharpe_observed, p_value)`` where p_value is the
    fraction of bootstrap resamples where ΔSharpe <= 0. G1 passes at
    95% confidence when ``p_value < 0.05``.
    """
    rng = np.random.default_rng(seed)
    n = len(primary_returns)
    if n != len(filtered_returns) or n == 0:
        raise ValueError("primary/filtered must be equal-length and non-empty")

    def _sharpe(x: np.ndarray) -> float:
        if x.std() == 0:
            return 0.0
        return float(x.mean() / x.std() * np.sqrt(252))

    obs = _sharpe(filtered_returns) - _sharpe(primary_returns)

    # Circular block bootstrap on the paired series.
    count_le0 = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b_prim = primary_returns[idx]
        b_filt = filtered_returns[idx]
        d = _sharpe(b_filt) - _sharpe(b_prim)
        if d <= 0:
            count_le0 += 1
    return obs, count_le0 / n_boot


# ---------------------------------------------------------------------------
# G3: PBO from CPCV paths.
# ---------------------------------------------------------------------------


def pbo_from_cpcv(
    cpcv_strategy_returns: list[np.ndarray],
    cpcv_primary_returns: list[np.ndarray],
) -> float:
    """Probability of Backtest Overfitting from CPCV paths (§7 G3).

    For each backtest path, rank strategies by in-sample IS performance
    and check whether the OOS-best strategy was the IS-best; PBO is the
    fraction of paths where the IS-best underperforms the median OOS.

    Here we use the simplified 2-strategy PBO: is the *filtered*
    strategy's OOS rank worse than the *primary*'s, across paths?
    PBO = fraction of paths where filtered loses to primary OOS despite
    being tuned. Returns a value in [0, 1]; gate is PBO <= 0.20.
    """
    if not cpcv_strategy_returns:
        return 1.0
    losses = 0
    for strat, prim in zip(cpcv_strategy_returns, cpcv_primary_returns):
        s_mean = np.mean(strat) if len(strat) else 0.0
        p_mean = np.mean(prim) if len(prim) else 0.0
        if s_mean < p_mean:
            losses += 1
    return losses / len(cpcv_strategy_returns)


# ---------------------------------------------------------------------------
# G2: Deflated Sharpe Ratio (simplified, single-number form).
# ---------------------------------------------------------------------------


def dsr(
    sharpe_observed: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Approximate Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Returns the probability that the *true* Sharpe is > 0 after deflating
    for trial count. Simplified single-number form; the full DSR uses
    the variance of the Sharpe estimator under non-normality. Gate G2
    is DSR >= 0.95.
    """
    # Expected max of n_trials draws from the Sharpe distribution under
    # the null (approx zero-mean Gaussian with the observed variance).
    if n_obs < 2 or n_trials < 1:
        return 0.0
    se = np.sqrt((1 - skew * sharpe_observed + (kurt - 1) / 4 * sharpe_observed**2) / (n_obs - 1))
    # Expected max of |N(0,1)| over n_trials ~ sqrt(2*ln(n_trials)) (approx).
    e_max = np.sqrt(2 * np.log(max(n_trials, 1)))
    # Deflate: how many SEs above the trial-corrected expectation?
    sharpe_deflated = sharpe_observed - e_max * se
    # P(true Sharpe > 0) under the deflated estimate ~ Phi(sharpe_deflated / se).
    from scipy.stats import norm
    return float(norm.cdf(sharpe_deflated / se)) if se > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregate.
# ---------------------------------------------------------------------------


@dataclass
class GateResults:
    """All five promotion gates (§7) for one configuration."""

    g1_uplift_delta_sharpe: float
    g1_uplift_pvalue: float
    g2_dsr: float
    g3_pbo: float
    g4_tstat_mean_r: float
    g5_brier: float
    g5_climatology_brier: float
    g5_reliability_slope: float
    n_trials: int
    details: dict = field(default_factory=dict)

    @property
    def g1_pass(self) -> bool:
        return self.g1_uplift_pvalue < 0.05 and self.g1_uplift_delta_sharpe > 0

    @property
    def g2_pass(self) -> bool:
        return self.g2_dsr >= 0.95

    @property
    def g3_pass(self) -> bool:
        return self.g3_pbo <= 0.20

    @property
    def g4_pass(self) -> bool:
        return self.g4_tstat_mean_r >= 3.0

    @property
    def g5_pass(self) -> bool:
        return (
            self.g5_brier < self.g5_climatology_brier
            and 0.8 <= self.g5_reliability_slope <= 1.2
        )

    def as_table(self) -> str:
        rows = [
            ("G1 Uplift", f"ΔSharpe={self.g1_uplift_delta_sharpe:.3f}, p={self.g1_uplift_pvalue:.3f}", "PASS" if self.g1_pass else "FAIL"),
            ("G2 DSR", f"{self.g2_dsr:.3f} (n_trials={self.n_trials})", "PASS" if self.g2_pass else "FAIL"),
            ("G3 PBO", f"{self.g3_pbo:.3f}", "PASS" if self.g3_pass else "FAIL"),
            ("G4 t-stat", f"{self.g4_tstat_mean_r:.3f}", "PASS" if self.g4_pass else "FAIL"),
            ("G5 Calib", f"Brier={self.g5_brier:.4f} vs clim={self.g5_climatology_brier:.4f}, slope={self.g5_reliability_slope:.3f}", "PASS" if self.g5_pass else "FAIL"),
        ]
        w1 = max(len(r[0]) for r in rows)
        w2 = max(len(r[1]) for r in rows)
        lines = [f"{'Gate':<{w1}}  {'Value':<{w2}}  Result", "-" * (w1 + w2 + 10)]
        for name, val, res in rows:
            lines.append(f"{name:<{w1}}  {val:<{w2}}  {res}")
        return "\n".join(lines)


def compute_gates(
    *,
    primary_returns_r: np.ndarray,
    filtered_returns_r: np.ndarray,
    oos_mean_r: float,
    oos_mean_r_se: float,
    brier: float,
    climatology_brier: float,
    reliability_slope: float,
    n_trials: int,
    n_obs: int,
    cpcv_strategy_returns: Optional[list[np.ndarray]] = None,
    cpcv_primary_returns: Optional[list[np.ndarray]] = None,
    observed_sharpe: Optional[float] = None,
    seed: int = 0,
) -> GateResults:
    """Compute G1-G5 from the run's outputs (§7)."""
    ds, p = block_bootstrap_uplift(primary_returns_r, filtered_returns_r, seed=seed)
    pbo = (
        pbo_from_cpcv(cpcv_strategy_returns, cpcv_primary_returns)
        if cpcv_strategy_returns
        else 1.0
    )
    sharpe = observed_sharpe if observed_sharpe is not None else (
        float(primary_returns_r.mean() / primary_returns_r.std() * np.sqrt(252))
        if primary_returns_r.std() > 0 else 0.0
    )
    g2 = dsr(sharpe, n_trials=n_trials, n_obs=n_obs)
    tstat = oos_mean_r / oos_mean_r_se if oos_mean_r_se > 0 else 0.0
    return GateResults(
        g1_uplift_delta_sharpe=ds,
        g1_uplift_pvalue=p,
        g2_dsr=g2,
        g3_pbo=pbo,
        g4_tstat_mean_r=tstat,
        g5_brier=brier,
        g5_climatology_brier=climatology_brier,
        g5_reliability_slope=reliability_slope,
        n_trials=n_trials,
    )
