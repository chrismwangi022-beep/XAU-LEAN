# XAU-LEAN

Systematic XAU/USD research, backtesting, risk-management, and execution framework using QuantConnect LEAN.

## Project Status

- **Current Phase:** Phase 6 — Timeframe Research
- **Phase 5:** COMPLETE — PASS
- **Asset:** XAU/USD
- **Canonical Data:** Dukascopy
- **Resolution:** 1-minute BID + ASK
- **Coverage:** January 2010 → August 2026
- **Engine:** QuantConnect LEAN
- **Environment:** Ubuntu / WSL2 + Docker Engine
- **Language:** Python

## Completed Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Environment Audit | COMPLETE |
| 1 | LEAN CLI Installation | COMPLETE |
| 2 | LEAN Workspace | COMPLETE |
| 3 | Docker Architecture | COMPLETE |
| 4 | Market Data Acquisition | COMPLETE |
| 5 | XAUUSD Data Engineering & Validation | COMPLETE — PASS |
| 6 | Timeframe Research | NEXT |

## Phase 5 — Data Engineering & Validation

Dukascopy is the canonical historical provider for XAU/USD.

Raw data:

    data/external/dukascopy/Market-Data-Lab-main/xauusd/

The canonical dataset contains monthly BID and ASK M1 files covering January 2010 through August 2026.

### Full-History Validation

- Expected months: **200**
- BID files: **200**
- ASK files: **200**
- Missing BID months: **NONE**
- Missing ASK months: **NONE**
- BID rows: **8,731,133**
- ASK rows: **8,731,134**
- Combined rows: **17,462,267**
- Invalid BID rows: **0**
- Invalid ASK rows: **0**
- Duplicate timestamps: **0**
- Out-of-order rows: **0**
- Invalid OHLC observations: **0**
- Negative spreads: **0**
- Aligned timestamps: **8,731,133**
- ASK-only observations: **1**

The single ASK-only observation occurs at **2026-08-16 23:59:00 UTC** and is retained as an isolated source anomaly.

### Spread Policy

Spread is treated as a real execution cost.

- BUY uses ASK
- SELL uses BID
- Extreme spreads are not deleted
- Stale/weekend observations remain in raw data
- Session/tradability logic determines whether an observation is usable

### Production Data Layer

    src/xau_lean/data/
    ├── __init__.py
    ├── dukascopy.py
    └── session.py

The production adapter provides controlled monthly BID/ASK access, UTC normalization, alignment, and bar streaming without modifying the canonical raw dataset.

The session layer classifies observations as TRADABLE, WEEKEND, STALE, ABNORMAL_SPREAD, or MISSING_ASK.

### LEAN Integration

Dukascopy integration was validated through:

- direct LEAN container file access
- synthetic PythonData testing
- real BID ingestion
- 1,000-row testing
- 10,000-row testing
- full-month testing
- BID/ASK alignment
- multi-month testing
- 2010 validation

Representative result:

    May 2014 raw aligned observations: 44,640
    May–August 2014 aligned observations: 133,920
    January 2010 aligned test: 1,000

The long-range dynamic LEAN GetSource(date) architecture was tested and rejected for production use because of poor long-range source-discovery behaviour. Controlled monthly access is the production approach.

## Data Integrity Policy

The canonical Dukascopy raw dataset is immutable.

No interpolation, fabricated prices, deletion of observations, weekend fill-forward, timestamp modification, OHLC modification, spread removal, or anomaly deletion is permitted in the raw layer.

Any research transformation must occur downstream and remain reproducible.

## Research Philosophy

XAU-LEAN follows a falsification-first approach.

The objective is not to maximize historical returns. Strategies must survive realistic costs, out-of-sample testing, walk-forward testing, robustness testing, Monte Carlo analysis, and risk constraints.

Future leakage, unrealistic fills, arbitrary data cleaning, and undocumented parameter tuning are prohibited.

## Phase 6 — Timeframe Research

Candidate timeframes:

- H6
- H4
- H1
- M15
- M5

No timeframe is assumed to be optimal.

Phase 6 will empirically evaluate market structure, volatility, spread impact, noise, trade frequency, returns, data quality, execution realism, robustness, and suitability for later FTMO-style risk constraints.

**Next exact objective: Phase 6 — Timeframe Research.**

Large historical datasets remain outside Git. Small deterministic test fixtures may be committed for reproducibility.
