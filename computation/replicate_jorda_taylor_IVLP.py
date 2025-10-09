"""
----------------------
Replication Disclaimer
----------------------

This script reproduces the empirical results of Jordà and Taylor (2024)
using Python rather than Stata. All variables, transformations, and
estimation procedures are defined from scratch in Python to ensure full
transparency and reproducibility.

The econometric framework, data sources, and identification strategy
strictly follow the original study; however, every computation here is
implemented with open-source Python libraries (pandas, numpy, statsmodels,
linearmodels, etc.) instead of relying on the Stata commands used in the
authors’ replication package.

The objective of this reimplementation is methodological: to replicate
each stage of the instrumented local-projection (IV–LP) analysis—data
construction, HP filtering, cyclical classification, IV regression, and
finite-sample inference—entirely within Python. This provides a transparent,
fully scriptable environment for verifying, extending, and understanding
the mechanics of the original Jordà–Taylor results.

By building the workflow directly in Python, we can also implement a more
flexible and relevant environment for the ARIMA-based bootstrap procedure,
allowing realistic modeling of serial dependence and small-sample dynamics.

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.iv import IV2SLS
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import chi2

# =========================== LOAD AND MERGE DATA =============================

fiscal_data = pd.read_stata("input/fiscal_consolidation_v032023.dta")
jst_data = pd.read_stata("input/JSTdatasetR6.dta")
dcapb_data = pd.read_stata("input/dcapb.dta")

# merge on country (ifs) and year identifiers
merged_data = fiscal_data.merge(jst_data, on=['ifs', 'year'], how='inner')
merged_data = merged_data.merge(dcapb_data, on=['ifs', 'year'], how='inner')
merged_data = merged_data.sort_values(by=['ifs', 'year'])


# =========================== CREATE KEY VARIABLES =============================

# fiscal consolidation variable (change in cyclically adjusted primary balance)
merged_data['dCAPB'] = merged_data['dnlgxqa']

# log real GDP per capita ×100 (scaling for interpretability)
merged_data['y'] = np.log(merged_data['rgdpbarro']) * 100

# first difference of log real GDP per capita
merged_data['Dy'] = merged_data.groupby('ifs')['y'].diff()


# ================== CREATE CUMULATIVE OUTPUT RESPONSES =======================

# D(h)y = y_{t+h} - y_{t-1}, for horizons h = 0,...,10
for h in range(11):
    merged_data[f'D{h}y'] = merged_data.groupby('ifs')['y'].shift(-h) - merged_data.groupby('ifs')['y'].shift(1)

# S(h)y = cumulative sum of output changes up to horizon h
for h in range(11):
    merged_data[f'S{h}y'] = merged_data[[f'D{i}y' for i in range(h+1)]].sum(axis=1, skipna=False)


# ==================CREATE CUMULATIVE FISCAL CONSOLIDATION MEASURES==================

# dCAPB(h): recursively sum dCAPB shifts forward
merged_data['dCAPB0'] = merged_data['dCAPB']
for h in range(1, 11):
    merged_data[f'dCAPB{h}'] = merged_data['dCAPB'].shift(-h) + merged_data[f'dCAPB{h-1}']

# SdCAPB(h): cumulative fiscal consolidation up to horizon h
for h in range(11):
    merged_data[f'SdCAPB{h}'] = merged_data[[f'dCAPB{i}' for i in range(h+1)]].sum(axis=1, min_count=1)


# ================ APPLY HODRICK–PRESCOTT FILTER =====================

# purpose: separate trend and cycle in log real GDP per capita (y)
merged_data['y_hpcyc'] = np.nan
merged_data['y_hptrend'] = np.nan

for country in merged_data['ifs'].unique():
    mask = merged_data['ifs'] == country
    temp_y = merged_data.loc[mask, 'y'].bfill().ffill()  # Fill missing values for HP filtering
    if temp_y.notna().sum() > 0:
        cycle, trend = hpfilter(temp_y, lamb=400)  # λ=400 for annual data
        merged_data.loc[mask, 'y_hpcyc'] = temp_y - trend
        merged_data.loc[mask, 'y_hptrend'] = trend


# ================= DEFINE BOOM AND SLUMP INDICATORS ===================

# boom if lagged HP-cycle > 0; Slump otherwise
merged_data['boom'] = (merged_data.groupby('ifs')['y_hpcyc'].shift(1) > 0).astype(int)
merged_data['slump'] = (merged_data.groupby('ifs')['y_hpcyc'].shift(1) <= 0).astype(int)


# ================= CREATE CONTROL VARIABLES ==================

# common macro controls and lags
merged_data['_x1'] = merged_data.groupby('ifs')['Dy'].shift(1)         # Lag 1 of Δy
merged_data['_x2'] = merged_data.groupby('ifs')['Dy'].shift(2)         # Lag 2 of Δy
merged_data['_x3'] = merged_data.groupby('ifs')['dCAPB'].shift(1)      # Lag 1 of dCAPB
merged_data['_x4'] = merged_data.groupby('ifs')['dCAPB'].shift(2)      # Lag 2 of dCAPB
merged_data['_x5'] = merged_data.groupby('ifs')['y_hpcyc'].shift(1)    # Lag 1 of HP-cycle
merged_data['_x6'] = merged_data.groupby('ifs')['debtgdp'].diff().shift(1)  # Lag 1 of ΔDebt/GDP

# sample identifiers
merged_data['full'] = 1
merged_data['exp'] = (merged_data['Dy'] > 1.5).astype(int)   # Expansion phase
merged_data['rec'] = (merged_data['Dy'] <= 1.5).astype(int)  # Recession phase


# =============== HANDLE MISSING VALUES (STATA-STYLE) ====================

# fill NAs for cumulative sums with zeros (matching Stata’s row-sum convention)
fill_cols = ['D0y', 'dCAPB', '_x1', '_x2', '_x3', '_x4', '_x5', '_x6',
             'D1y', 'D2y', 'D3y', 'D4y', 'D5y', 'D6y', 'D7y', 'D8y', 'D9y', 'D10y',
             'dCAPB0', 'dCAPB1', 'dCAPB2', 'dCAPB3', 'dCAPB4', 'dCAPB5', 'dCAPB6',
             'dCAPB7', 'dCAPB8', 'dCAPB9', 'dCAPB10']

sum_fill_cols = [f'S{h}y' for h in range(11)] + [f'SdCAPB{h}' for h in range(11)]
merged_data[sum_fill_cols] = merged_data[sum_fill_cols].fillna(0)

iv_data = merged_data.copy()
iv_data.to_csv("input/iv_data_corrected.csv", index=False)


# ==================IV REGRESSION FUNCTION=====================

def run_iv_regression(iv_data, horizon, subsample):
    """
    Runs IV regression for D(h)y on dCAPB, instrumented by size (demeaned),
    with country-clustered SEs and horizon-specific dependent variables.
    """

    print(f"\n============== Horizon = {horizon}, Subsample = {subsample} ==============")

    # filter based on regime
    if subsample == "Boom":
        df = iv_data[iv_data['boom'] == 1].copy()
    elif subsample == "Slump":
        df = iv_data[iv_data['slump'] == 1].copy()
    else:
        df = iv_data.copy()

    y_var = f"D{horizon}y"
    exog_vars = ['_x1', '_x2', '_x3', '_x4', '_x5', '_x6']

    df = df.dropna(subset=[y_var, 'dCAPB', 'size'] + exog_vars)

    # create demeaned instrument (within-country)
    df.loc[:, 'size_transformed'] = df['size'] - df.groupby('ifs')['size'].transform('mean')
    instrument = df[['size_transformed']]

    # demean dependent and endogenous vars by country (removes FE)
    for var in [y_var, 'dCAPB']:
        df.loc[:, var] = df[var] - df.groupby('ifs')[var].transform('mean')

    # demean controls
    for var in exog_vars:
        df.loc[:, var] = df[var] - df.groupby('ifs')[var].transform('mean')

    # prepare X matrix (add constant)
    X = sm.add_constant(df[exog_vars], has_constant='add')

    # run IV regression
    iv_model = IV2SLS(
        dependent=df[y_var],
        exog=X,
        endog=df[['dCAPB']],
        instruments=instrument
    ).fit(cov_type='clustered', clusters=df['ifs'])

    # extract key statistics
    coef = iv_model.params['dCAPB']
    std_err = iv_model.std_errors['dCAPB']
    ci_low = coef - 1.96 * std_err
    ci_high = coef + 1.96 * std_err
    t_stat = iv_model.tstats['dCAPB']
    V_h = iv_model.cov.loc['dCAPB', 'dCAPB']
    chi_squared_stat = coef ** 2 / V_h

    print(f"Coefficient (β): {coef:.3f}")
    print(f"Standard Error: {std_err:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"Chi-squared: {chi_squared_stat:.3f}")

    return coef, ci_low, ci_high, V_h


# =============== RUN REGRESSIONS ACROSS SUBSAMPLES AND HORIZONS ================

subsamples = ["Full Sample", "Boom", "Slump"]
horizons = range(5)

results = {sub: {"coefs": [], "lower_bound": [], "upper_bound": [], 'var_matrix': []} for sub in subsamples}

for sub in subsamples:
    for h in horizons:
        coef, ci_low, ci_high, V_h = run_iv_regression(iv_data, h, sub)
        results[sub]["coefs"].append(coef)
        results[sub]["lower_bound"].append(ci_low)
        results[sub]["upper_bound"].append(ci_high)
        results[sub]['var_matrix'].append(V_h)


# =================== JOINT WALD TEST (ACROSS HORIZONS) =====================

for sub in subsamples:
    coefs = np.array(results[sub]["coefs"])
    var_matrix = np.diag(results[sub]["var_matrix"])
    chi2_stat = coefs.T @ np.linalg.inv(var_matrix) @ coefs
    p_value = 1 - chi2.cdf(chi2_stat, df=len(coefs))

    print(f"\nSubsample: {sub}")
    print(f"Joint Chi-Squared: {chi2_stat:.3f}, p-value: {p_value:.3f}")


# ========================== PLOT RESULTS ======================

plt.figure(figsize=(8, 5))
colors = {"Full Sample": "blue", "Boom": "green", "Slump": "red"}

# compute average β across horizons
average_coefs = {sub: np.mean(results[sub]["coefs"]) for sub in subsamples}

for sub in subsamples:
    plt.fill_between(
        horizons,
        results[sub]["lower_bound"],
        results[sub]["upper_bound"],
        color=colors[sub],
        alpha=0.2
    )
    plt.plot(
        horizons,
        results[sub]["coefs"],
        marker='o',
        linestyle='-',
        color=colors[sub],
        label=sub
    )
    plt.axhline(
        y=average_coefs[sub],
        color=colors[sub],
        linestyle='dashed',
        alpha=0.7,
        label=f"{sub} Avg: {average_coefs[sub]:.3f}"
    )

plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Horizon (Years)")
plt.ylabel("Response (log Real GDP per capita ×100)")
plt.title("Responses to 1% Fiscal Consolidation")
plt.legend()
plt.grid(True)
plt.show()