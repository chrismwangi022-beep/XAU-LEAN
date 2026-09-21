# Phase 6.3D — Direction × Volatility Temporal OOS Validation

## Status

**COMPLETE**

## Research question

> Does the fixed Phase 6.3C persistence/reversal interaction survive chronological out-of-sample testing?

## Fixed specification

Phase 6.3D evaluates the **exact Phase 6.3B direction × volatility interaction statistic** using the fixed Phase 6.3C persistence/reversal classification.

The temporal OOS validation was conducted without:

- candidate selection
- parameter optimization
- threshold optimization
- strategy modification
- post-hoc hypothesis modification
- period selection based on observed results

The purpose of the exercise was diagnostic validation of temporal stability.

---

## Method

### Timeframe

M15 XAUUSD.

### Horizons

- H1
- H3
- H5

### Persistence classification

The fixed Phase 6.3C classification was retained:

- persistent
- reversal

No observations were missing the persistence classification in any completed window.

### Statistical procedure

The canonical Phase 6.3B direction × volatility interaction statistic was used unchanged.

Bootstrap inference:

- 5,000 replicates
- seed: 63063

The chronological windows were predefined before interpretation.

---

# Chronological OOS results

## Window 1 — 2010–2014

This window was used as a historical measurement window rather than a genuine forward OOS test because no preceding training period was included in the requested scope.

| Horizon | Δ Accuracy | Signed Interaction |
|---|---:|---:|
| H1 | +4.1046 pp | +0.008602% |
| H3 | +4.5363 pp | +0.023031% |
| H5 | +5.9264 pp | +0.026290% |

Observed interaction was positive across all three horizons.

This result is historical measurement evidence only and is not treated as independent OOS validation.

---

## Window 2 — 2015–2019

The chronological OOS component was 2019.

| Horizon | OOS Δ Accuracy | OOS Signed Interaction |
|---|---:|---:|
| H1 | +2.7585 pp | +0.005249% |
| H3 | +5.0000 pp | +0.010636% |
| H5 | +5.9787 pp | +0.016845% |

The fixed interaction remained positive across all three horizons during the 2019 OOS period.

However, the bootstrap confidence intervals were generally broad and crossed zero. Therefore this result does not establish a statistically stable trading edge.

---

## Window 3 — 2020–2021

| Horizon | OOS Δ Accuracy | OOS Signed Interaction |
|---|---:|---:|
| H1 | +1.1160 pp | +0.002612% |
| H3 | +4.2658 pp | +0.020776% |
| H5 | +5.6671 pp | +0.031112% |

The interaction remained positive across all three horizons.

The observed effect was weaker at H1 and stronger at H3/H5, but the corresponding bootstrap uncertainty remained substantial.

---

## Window 4 — 2022–2023

| Horizon | OOS Δ Accuracy | OOS Signed Interaction |
|---|---:|---:|
| H1 | -4.6141 pp | -0.002866% |
| H3 | -4.0776 pp | -0.006154% |
| H5 | -4.4075 pp | -0.012617% |

The interaction changed sign and became negative across all three horizons.

The reversal component was particularly negative:

| Horizon | Reversal Δ Accuracy |
|---|---:|
| H1 | -8.9549 pp |
| H3 | -9.1069 pp |
| H5 | -4.6917 pp |

For H1 and H3, the bootstrap accuracy confidence intervals for the reversal component excluded zero:

- H1: [-14.7713 pp, -3.1090 pp]
- H3: [-17.0644 pp, -0.5374 pp]

This demonstrates that the deterioration in 2022–2023 was not confined to a single month.

---

## Window 5 — 2024–2026

Requested end: 2026-08-22.

Observed data extended through 2026-08-21.

| Horizon | OOS Δ Accuracy | OOS Signed Interaction |
|---|---:|---:|
| H1 | -0.4636 pp | +0.007186% |
| H3 | -0.6335 pp | +0.017253% |
| H5 | -0.2067 pp | +0.018656% |

The interaction returned toward approximately zero.

All accuracy confidence intervals crossed zero, and the signed-return confidence intervals also crossed zero.

The result therefore does not provide evidence of a stable positive or negative interaction during this period.

---

# Temporal summary

| Period | H1 | H3 | H5 |
|---|---:|---:|---:|
| 2010–2014 | +4.10 pp | +4.54 pp | +5.93 pp |
| 2019 OOS | +2.76 pp | +5.00 pp | +5.98 pp |
| 2020–2021 OOS | +1.12 pp | +4.27 pp | +5.67 pp |
| 2022–2023 OOS | -4.61 pp | -4.08 pp | -4.41 pp |
| 2024–2026 OOS | -0.46 pp | -0.63 pp | -0.21 pp |

---

# Reconstruction and reconciliation

All completed temporal windows reconstructed the canonical Phase 6.3B statistic.

### Reconstruction

- Missing persistence observations: **0**
- Unclassified observations: **0**
- Canonical and scoped observation counts reconciled exactly.

### Reconciliation

Every reported OOS/train cell checked by the script reconciled exactly against the canonical Phase 6.3B statistic.

Reported reconciliation results:

- Window 1: **9/9 PASS**
- Window 2: **12/12 PASS**
- Window 3: **9/9 PASS**
- Window 4: **6/6 PASS**
- Window 5: **3/3 PASS**

No numerical discrepancies were reported.

---

# Interpretation

The fixed Phase 6.3C interaction does **not** demonstrate temporal stability.

The observed temporal sequence is:

1. Positive interaction during earlier historical periods.
2. Positive interaction during the 2019 and 2020–2021 OOS periods.
3. A material negative interaction during 2022–2023.
4. A return toward approximately neutral interaction during 2024–2026.

This behavior is inconsistent with treating the fixed interaction as a stable cross-period trading edge.

The 2022–2023 deterioration is particularly informative because it persists across a two-year OOS window and is concentrated substantially in the reversal component.

The 2024–2026 result is also important: the interaction does not remain strongly negative. Instead, it moves back toward approximately zero.

Therefore the evidence is better characterized as **temporal instability / regime dependence** than as a persistent directional trading edge.

---

# Decision

## Phase 6.3C interaction: DO NOT PROMOTE TO TRADING RULE

The fixed persistence/reversal direction × volatility interaction is **not promoted into a strategy rule** based on Phase 6.3D.

Reason:

> The effect does not survive chronological testing as a stable cross-period interaction.

This is a research conclusion, not a claim that no relationship exists under any market condition.

---

# What this result does NOT establish

Phase 6.3D does not establish:

- that the market has no directional structure
- that persistence/reversal contains no information
- that volatility contains no useful information
- that the relationship can never become useful under another hypothesis
- that a profitable strategy is impossible

It establishes only that the **specific fixed Phase 6.3C interaction**, tested using the exact Phase 6.3B statistic under chronological OOS validation, did not demonstrate sufficient temporal stability to justify promotion as a fixed trading rule.

---

# Research integrity

The following were deliberately held constant:

- statistic
- persistence/reversal definition
- horizons
- chronological boundaries
- bootstrap methodology
- bootstrap seed
- data methodology

No strategy was modified after observing unfavorable results.

No favorable periods were selected for promotion while unfavorable periods were discarded.

No parameter optimization was performed.

---

# Final Phase 6.3D conclusion

**Phase 6.3D is COMPLETE.**

The fixed Phase 6.3C direction × volatility interaction shows historical variation but fails the required temporal-stability test.

**Conclusion: reject the interaction as a stable strategy-grade edge under the tested specification.**

The result should be retained as a documented negative/diagnostic research finding rather than discarded.

No trading strategy should be constructed from this interaction without a new, independently specified hypothesis and a fresh validation protocol.

