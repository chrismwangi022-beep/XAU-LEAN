# XAU-LEAN

A research, backtesting, risk-management, and execution framework for
systematic XAU/USD algorithmic trading using QuantConnect LEAN.

## Project Status

**Phase:** 1.1 — Project Structure  
**Status:** Foundation setup  
**Asset:** XAU/USD (Gold)  
**Engine:** QuantConnect LEAN  
**Language:** Python  
**Environment:** Ubuntu / WSL2 + Docker  
**Repository:** XAU-LEAN

---

## Objectives

The project is designed to:

1. Acquire and validate historical XAU/USD market data.
2. Develop systematic trading strategies.
3. Backtest strategies using QuantConnect LEAN.
4. Perform robust out-of-sample and walk-forward testing.
5. Apply strict risk-management rules.
6. Evaluate strategies against FTMO-style trading constraints.
7. Run paper trading before any live deployment.
8. Produce reproducible research and backtest results.
9. Prevent unvalidated strategies from reaching execution.

---

## Architecture

```text
XAU-LEAN/
│
├── src/
│   └── xau_lean/
│       ├── algorithm/
│       ├── data/
│       ├── indicators/
│       ├── strategy/
│       ├── risk/
│       ├── execution/
│       └── monitoring/
│
├── config/
│   ├── strategy/
│   └── risk/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── backtests/
│   ├── configs/
│   ├── results/
│   └── reports/
│
├── research/
│   ├── notebooks/
│   └── experiments/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── scripts/
├── logs/
│
├── .gitignore
├── README.md
└── Commands.txt