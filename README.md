# TRIAGE: Multi-Strategy Portfolio Allocation & Equity Curve Gating

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Portfolio](https://img.shields.io/badge/Method-Risk%20Parity%20%7C%20HRP-success.svg)]()
[![Risk Engine](https://img.shields.io/badge/Gating-Equity%20Curve%20Momentum-purple.svg)]()

> **Quantitative portfolio allocation framework implementing Hierarchical Risk Parity (HRP), dynamic capital triage, and equity-curve momentum gating to eliminate portfolio tail drawdowns.**

---

## 1. Portfolio Philosophy

Even the best quantitative strategies experience multi-month drawdowns. **TRIAGE** manages risk at the meta-portfolio level:
- **Hierarchical Risk Parity (HRP):** Allocates capital based on graph-theory tree clustering of strategy return correlations, avoiding the instability of classic Markowitz mean-variance optimization.
- **Equity Curve Momentum Gating:** Evaluates the moving average of each strategy's equity curve. Capital is automatically down-scaled or parked in cash when a strategy drops below its equity moving average, and re-allocated as positive momentum returns.

```
                          TRIAGE PORTFOLIO ALLOCATION
 ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │ Candidate Strategy 1 │     │ Candidate Strategy 2 │     │ Candidate Strategy N │
 └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
            │                            │                            │
            └────────────────────────────┼────────────────────────────┘
                                         ▼
                           ┌────────────────────────────┐
                           │ Hierarchical Risk Parity   │
                           │ (Correlation Clustering)   │
                           └────────────────────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
                           │ Equity Curve Gating        │
                           │ (Momentum vs Moving Avg)   │
                           └────────────────────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
                           │ Dynamic Capital Allocation │
                           └────────────────────────────┘
```

---

## 2. Usage

```bash
# Allocate Portfolio Capital across Strategy Streams
python run_portfolio_triage.py --strategies-dir ../strategies/ --method hrp
```
