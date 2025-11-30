# Honors Thesis Replication Repository: ARIMA-Based Bootstrap for IV–Local Projections

This repository accompanies **Yesui Nyamchuluun’s Honors Thesis (University of Minnesota, 2025)** titled  
> *“State-Dependent Fiscal Policy Effects: ARIMA-Based Bootstrap Meets Local Projections.”*

The project replicates and extends the econometric framework of **Jordà, Schularick, and Taylor (2020)** (*Journal of Monetary Economics*), translating their Stata-based IV–Local Projection (IV–LP) design into a fully transparent, reproducible **Python implementation**.  
It introduces an **ARIMA-based bootstrap** procedure to evaluate finite-sample inference robustness in estimating fiscal multipliers across different business-cycle regimes.

---
## Methodological Summary

### 1. Data Preparation  
- Combines fiscal, JST, and CAPB datasets at the country–year level.  
- Constructs growth, fiscal consolidation, and cumulative response variables.  
- Applies the **Hodrick–Prescott filter (λ = 400)** to extract cyclical components of GDP and define **boom** and **slump** states.  

### 2. ARIMA Model Selection  
- Fits ARIMA(p,0,q) models to detrended GDP residuals.  
- Selects a single common specification minimizing **AIC/BIC**, saved in `common_arima_order.json`.

### 3. Instrumented Local Projections (IV–LP)

The estimated regression takes the form:
<p align="center">
  <img src="equation_ivlp.png" alt="IV–LP equation: Δ_h y_{i,t+h} = α_i + β_h dCAPB_{i,t} + X'_{i,t} γ_h + ε_{i,t+h}" width="650"/>
</p>
  using 2SLS regressions instrumenting *dCAPB* with within-country demeaned *size*.  
- Includes controls (_x₁–x₆_) consistent with the baseline JST specification.

### 4. ARIMA-Based Bootstrap  
- Simulates residual paths under the fitted ARIMA process.  
- Recomputes IV–LP regressions for each draw to obtain empirical χ² distributions.  
- Provides **joint** and **horizon-specific** bootstrap p-values under finite-sample uncertainty.

---

## Key Outputs

| Subsample       | Observed χ² | Standard χ² (95%) for df = 5 | Bootstrap 95th Percentile | Bootstrap p-value | Decision             |
|------------------|-------------|--------------------|----------------------------|-------------------|----------------------|
| Full Sample      | 19.454      | 11.070             | 23.080                     | 0.069*            | Do not reject H₀     |
| Boom             | 58.085      | 11.070             | 15.524                     | 0.000***          | Reject H₀            |
| Slump            | 17.552      | 11.070             | 25.071                     | 0.099*            | Do not reject H₀     |

All results are automatically exported to  
`output/bootstrap_joint_chi2_summary.csv`.

---

### Yesui Nyamchuluun advised by Prof. Hannes Malmberg
yesui.nyamchuluun@gmail.com

University of Minnesota, Department of Economics, Class of 2025. 
