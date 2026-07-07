"""Paper-trading harness for a frozen filtered strategy (§8/§9).

Turns a promotable meta-labeling result into a deployable artifact: a frozen
strategy (booster + calibrator + threshold + sizing config) whose ``decide()``
maps one point-in-time feature snapshot to a trade decision, and a ``simulate()``
that walks an out-of-sample event stream through the SAME ``decide()`` under the
§8 live-execution overlays. The exact function that drives the paper simulation
is the one that would drive the live loop — no train/serve skew by construction.

Sizing is **fixed** (rung 2 of the §7 baseline ladder — filter at tau0, constant
size): the G5 calibration gate has not cleared for the promotable slice
(majors x donchian), so probability-laddered sizing (rung 3) is deliberately NOT
used. The filter's G1 uplift is what is bankable; the sizer stays flat until the
probabilities are trustworthy (§8 "the sizer consumes calibrated p ... revisit
continuous sizing only after G5 has passed").

This is an OOS *simulation* on held-out data (a paper dry-run), not forward live
results — labelled as such wherever it reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .gates import cost_for_symbol
from .sizing import DEFAULT_BASE_RISK, default_tau0
from .uniqueness import uniqueness_weights

__all__ = ["Decision", "Trade", "PaperReport", "PaperStrategy"]


# ---------------------------------------------------------------------------
# Value types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A trade decision for one candidate event (the live/paper output)."""

    take: bool
    p_win: float
    size_fraction: float          # fraction of equity risked (0 if not taken)
    reason: str                   # 'take' | 'below_tau0' | 'symbol_busy' | 'risk_cap'
    symbol: str = ""
    side: int = 0


@dataclass(frozen=True)
class Trade:
    """A taken paper trade + its resolved outcome."""

    symbol: str
    t0: pd.Timestamp
    t1: pd.Timestamp
    side: int
    p_win: float
    size_fraction: float
    r_realized: float
    cost_r: float

    @property
    def pnl_r(self) -> float:
        """Realized R after cost (size-independent; R is risk-normalized)."""
        return float(self.r_realized - self.cost_r)


@dataclass
class PaperReport:
    """Result of a paper simulation."""

    trades: list[Trade]
    equity_curve: pd.Series          # 1%-risk fixed-fractional equity, indexed by exit time
    cum_r: pd.Series                 # cumulative R after cost, indexed by exit time
    stats: dict = field(default_factory=dict)
    baseline_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frozen strategy.
# ---------------------------------------------------------------------------


class PaperStrategy:
    """A frozen filtered strategy: booster + calibrator + threshold + sizing.

    Fit on a (time-ordered, purged) training set via the §6 holdout calibration
    recipe (booster on the earlier fit portion, 1-D calibrator on a purged tail
    fold). Then ``decide`` / ``simulate`` consume it read-only.
    """

    def __init__(
        self,
        *,
        features: list[str],
        base_risk: float = DEFAULT_BASE_RISK,
        cost_default: float = 0.04,
        calib_frac: float = 0.20,
        calib_method: Optional[str] = None,
        params: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        self.features = list(features)
        self.cat_cols = ["symbol", "rule_id"]
        self.base_risk = float(base_risk)
        self.tau0 = round(default_tau0(cost_default), 3)
        self.calib_frac = float(calib_frac)
        self._calib_method = calib_method
        self.params = params or {"max_depth": 4, "num_leaves": 15, "learning_rate": 0.03}
        self.seed = int(seed)
        # Fit artifacts (populated by fit()).
        self._booster = None
        self._calibrator = None
        self._method: Optional[str] = None
        self._feature_names: list[str] = []
        self._cat_levels: dict[str, object] = {}

    # -- feature/categorical coding ------------------------------------------

    def _build_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature frame with numeric features + categorical symbol/rule_id —
        the same shape scripts build for pooled training."""
        X = df[self.features + self.cat_cols].copy()
        for c in self.features:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        for c in self.cat_cols:
            X[c] = X[c].astype("category")
        return X

    def _coded_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Reproduce train_meta_model's `_to_codes` with the captured category
        levels: column order = feature_names, cat cols -> integer codes."""
        X = df[self._feature_names].copy()
        for c in self._feature_names:
            if c in self._cat_levels:
                X[c] = pd.Categorical(X[c], categories=self._cat_levels[c]).codes
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        return X.to_numpy(dtype=np.float64)

    # -- fit -----------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "PaperStrategy":
        """Fit via train_meta_model (single source of truth) and capture the
        fitted booster + calibrator — so the frozen strategy IS the promotable
        model, not a re-implementation that can silently drift."""
        from .models.meta import train_meta_model

        df = train_df.sort_values("t0").reset_index(drop=True)
        X = self._build_matrix(df)
        y = df["label"].astype(int)
        ev = df[["t0", "t1"]].copy()
        ev["t1"] = ev["t1"].fillna(ev["t0"] + pd.Timedelta(hours=96))
        w = uniqueness_weights(ev)

        cap: dict = {}
        # The throwaway test set (the full train) only satisfies the API — its
        # predictions are discarded; the captured booster/calibrator are fit on
        # the internal purged fit/calib folds, independent of the test arg.
        train_meta_model(
            X, y, sample_weight=w, test_features=X, test_labels=y,
            events=df[["t0", "t1"]], calibration="holdout",
            calib_frac=self.calib_frac, calib_method=self._calib_method,
            params=self.params, n_estimators=1000, _capture=cap, seed=self.seed,
        )
        if "booster" not in cap:
            raise RuntimeError("holdout calibration degenerate — strategy not fit")
        self._booster = cap["booster"]
        self._calibrator = cap["calibrator"]
        self._method = cap["method"]
        self._feature_names = cap["feature_names"]
        self._cat_levels = cap["cat_levels"]
        self.n_fit = cap.get("fit_n", 0)
        self.n_calib = cap.get("calib_n", 0)
        return self

    # -- inference -----------------------------------------------------------

    def predict_p(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated P(win) for each row (the exact path live inference uses)."""
        if self._booster is None:
            raise RuntimeError("PaperStrategy not fit")
        from .models.meta import _apply_1d_calibrator

        raw = self._booster.predict(self._coded_matrix(df))
        return _apply_1d_calibrator(self._calibrator, self._method, raw)

    def decide(self, row: pd.Series | pd.DataFrame, *, symbol_busy: bool = False,
               open_risk: float = 0.0, max_open_risk_mult: float = 3.0) -> Decision:
        """Map one candidate event to a trade decision (fixed sizing).

        ``row`` is a single event's features (Series or 1-row frame). The §8
        overlays are passed in by the caller: ``symbol_busy`` (a position is
        already open on this symbol) and ``open_risk`` (aggregate risk already
        committed, in equity fractions) against ``max_open_risk_mult * base_risk``.
        """
        one = row.to_frame().T if isinstance(row, pd.Series) else row
        p = float(self.predict_p(one)[0])
        sym = str(one.iloc[0].get("symbol", ""))
        side = int(one.iloc[0].get("side", 0)) if "side" in one.columns else 0
        if symbol_busy:
            return Decision(False, p, 0.0, "symbol_busy", sym, side)
        if p < self.tau0:
            return Decision(False, p, 0.0, "below_tau0", sym, side)
        if open_risk + self.base_risk > max_open_risk_mult * self.base_risk + 1e-12:
            return Decision(False, p, 0.0, "risk_cap", sym, side)
        return Decision(True, p, self.base_risk, "take", sym, side)

    # -- simulation ----------------------------------------------------------

    def simulate(self, test_df: pd.DataFrame, *, max_open_risk_mult: float = 3.0) -> PaperReport:
        """Walk an OOS event stream through decide() under §8 overlays.

        Enforces per-symbol single open position and an aggregate open-risk cap.
        Returns a PaperReport with the trade log, equity curve, and stats — plus
        a take-all baseline (same overlays, no p filter) for the uplift read.
        """
        df = test_df.sort_values("t0").reset_index(drop=True)
        preds = self.predict_p(df)

        trades: list[Trade] = []
        busy_until: dict[str, pd.Timestamp] = {}
        open_positions: list[tuple[pd.Timestamp, float]] = []  # (t1, risk)

        def _open_risk(now: pd.Timestamp) -> float:
            nonlocal open_positions
            open_positions = [(t1, r) for (t1, r) in open_positions if t1 > now]
            return sum(r for _, r in open_positions)

        for i, r in df.iterrows():
            sym = str(r["symbol"])
            t0 = pd.Timestamp(r["t0"])
            t1 = pd.Timestamp(r["t1"]) if pd.notna(r["t1"]) else t0 + pd.Timedelta(hours=96)
            busy = sym in busy_until and t0 < busy_until[sym]
            dec = self.decide(r, symbol_busy=busy, open_risk=_open_risk(t0),
                              max_open_risk_mult=max_open_risk_mult)
            # Override the decision's cached p with the batch prediction (identical
            # model, avoids a second single-row predict for speed/consistency).
            if dec.take:
                busy_until[sym] = t1
                open_positions.append((t1, self.base_risk))
                trades.append(Trade(
                    symbol=sym, t0=t0, t1=t1, side=int(r.get("side", 0)),
                    p_win=float(preds[i]), size_fraction=self.base_risk,
                    r_realized=float(r["r_realized"]), cost_r=cost_for_symbol(sym),
                ))

        stats, eq, cumr = self._summarize(trades)
        base_stats = self._baseline(df)
        return PaperReport(trades=trades, equity_curve=eq, cum_r=cumr,
                           stats=stats, baseline_stats=base_stats)

    # -- reporting -----------------------------------------------------------

    def _summarize(self, trades: list[Trade]):
        if not trades:
            empty = pd.Series(dtype=float)
            return {"n_trades": 0}, empty, empty
        ts = pd.to_datetime([t.t1 for t in trades])
        pnl_r = np.array([t.pnl_r for t in trades])
        order = np.argsort(ts.values)
        ts, pnl_r = ts[order], pnl_r[order]
        cum_r = pd.Series(np.cumsum(pnl_r), index=ts, name="cum_r")
        eq = pd.Series(np.cumprod(1.0 + self.base_risk * pnl_r), index=ts, name="equity")
        wins = pnl_r > 0
        n = len(pnl_r)
        gross_win = pnl_r[pnl_r > 0].sum()
        gross_loss = -pnl_r[pnl_r < 0].sum()
        peak = np.maximum.accumulate(eq.to_numpy())
        max_dd = float(np.max((peak - eq.to_numpy()) / peak)) if len(eq) else 0.0
        stats = {
            "n_trades": int(n),
            "win_rate": round(float(wins.mean()), 4),
            "mean_r_after_cost": round(float(pnl_r.mean()), 4),
            "total_r": round(float(pnl_r.sum()), 3),
            "sharpe_per_trade_ann": round(float(pnl_r.mean() / pnl_r.std() * np.sqrt(252)), 3) if pnl_r.std() > 0 else 0.0,
            "profit_factor": round(float(gross_win / gross_loss), 3) if gross_loss > 0 else float("inf"),
            "max_drawdown_pct_at_1pct_risk": round(max_dd * 100, 3),
            "final_equity_at_1pct_risk": round(float(eq.iloc[-1]), 4),
        }
        return stats, eq, cum_r

    def _baseline(self, df: pd.DataFrame) -> dict:
        """Take-all baseline (every candidate, same fixed size) for the uplift."""
        r = df["r_realized"].to_numpy()
        cost = np.array([cost_for_symbol(s) for s in df["symbol"]])
        pnl = r - cost
        if len(pnl) == 0 or pnl.std() == 0:
            return {"n_trades": int(len(pnl)), "total_r": round(float(pnl.sum()), 3)}
        return {
            "n_trades": int(len(pnl)),
            "mean_r_after_cost": round(float(pnl.mean()), 4),
            "total_r": round(float(pnl.sum()), 3),
            "sharpe_per_trade_ann": round(float(pnl.mean() / pnl.std() * np.sqrt(252)), 3),
        }
