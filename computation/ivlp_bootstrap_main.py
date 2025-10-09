"""
--------------------------------
ARIMA-Based Bootstrap for IV–LP
--------------------------------
This script executes the finite-sample bootstrap for the instrumented local
projection (IV–LP) design using a single common ARIMA(p,0,q) order that is
preselected on the full sample. It reads the chosen order from JSON and uses
that same (p,0,q) within each subsample, refitting parameters per subset.

Notes:
- HP filter λ = 400 (annual data).
- IV: dCAPB instrumented by within-country demeaned 'size'.
- Controls: _x1.._x6 (with _x5 = lagged cycle), listwise deletion kept identical.
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from linearmodels.iv import IV2SLS
from scipy.stats import chi2
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed

# -------- Tunables / constants --------
DATA_PATH        = "input/iv_data_corrected.csv"
ORDER_JSON_PATH  = "output/common_arima_order.json"
HORIZONS         = range(5)   # {0,1,2,3,4}
HP_LAMBDA        = 400
NUM_BOOTSTRAPS   = 10000
COUNTRY_COL      = "ifs"
INSTR_COL        = "size"
Y_VAR_BASE       = "D"        # base name for D(h)y columns
OUTPUT_DIR       = "output"

horizons = HORIZONS

# ============================== Data Load & Residuals ==============================

merged_data = pd.read_csv(DATA_PATH).reset_index(drop=True)
merged_data["_x5"] = merged_data["y_hpcyc"].shift(1).fillna(merged_data["y_hpcyc"].mean())
merged_data["log_gdp"] = np.log(merged_data["rgdpbarro"]).fillna(merged_data["rgdpbarro"].mean())

# Country-trend residuals: OLS(log_gdp ~ 1 + year) within country
merged_data["residuals_country"] = merged_data["log_gdp"] - merged_data.groupby(COUNTRY_COL)["log_gdp"].transform(
    lambda x: sm.OLS(x, sm.add_constant(merged_data.loc[x.index, "year"]))
               .fit()
               .predict(sm.add_constant(merged_data.loc[x.index, "year"]))
)

subsamples = {
    "Full Sample": merged_data.copy(),
    "Boom":        merged_data[merged_data["boom"] == 1].copy(),
    "Slump":       merged_data[merged_data["slump"] == 1].copy(),
}

results = {
    sub: {
        "coefs": [],
        "var_matrix": [],
        "chi2_stat": None,
        "individual_chi2_stats": {h: 0.0 for h in HORIZONS}
    } for sub in subsamples
}

# ============================== Read Common ARIMA Order ==============================

with open(ORDER_JSON_PATH, "r") as f:
    _order_meta = json.load(f)
COMMON_ORDER = tuple(_order_meta["common_order"])
print(f"[INFO] Using common ARIMA order from JSON: {COMMON_ORDER}")

best_orders = {
    "Full Sample": COMMON_ORDER,
    "Boom":        COMMON_ORDER,
    "Slump":       COMMON_ORDER,
}

# ============================== Bootstrap Residuals ==============================

def bootstrap_for_subsample(subdata, num_bootstraps, arima_order):
    """
    Fit ARIMA(order) to residuals_country in this subsample and simulate
    'num_bootstraps' residual paths (length T). Returns array (B, T).
    """
    subdata = subdata.dropna(subset=["residuals_country"]).reset_index(drop=True)
    if len(subdata) < 10:
        print(f"Warning: Too few observations ({len(subdata)}) for ARIMA in bootstrap.")
        return np.zeros((num_bootstraps, len(subdata)))  # Return zero-matrix to prevent errors
    
    arima_model = ARIMA(subdata["residuals_country"], order=arima_order).fit()
    
    simulated_residuals = np.array([
        arima_model.simulate(nsimulations=len(subdata), anchor="start")
        for _ in range(num_bootstraps)
    ])
    
    print(f"Bootstrapped residuals generated for order {arima_order}: Shape {simulated_residuals.shape}")
    return simulated_residuals

# ============================== IV Regression (Observed) ==============================

def run_iv_regression(iv_data, horizon, subsample):
    if subsample == "Boom":
        df = iv_data[iv_data["boom"] == 1].copy()
    elif subsample == "Slump":
        df = iv_data[iv_data["slump"] == 1].copy()
    else:
        df = iv_data.copy()

    y_var = f"D{horizon}y"
    exog_vars = ["_x1", "_x2", "_x3", "_x4", "_x5", "_x6"]

    df = df.dropna(subset=[y_var, "dCAPB", INSTR_COL] + exog_vars)

    df["size_transformed"] = df[INSTR_COL] - df.groupby(COUNTRY_COL)[INSTR_COL].transform("mean")
    instrument = df[["size_transformed"]]

    for var in [y_var, "dCAPB"] + exog_vars:
        df[var] = df[var] - df.groupby(COUNTRY_COL)[var].transform("mean")

    X = sm.add_constant(df[exog_vars])

    iv_model = IV2SLS(df[y_var], X, df[["dCAPB"]], instrument).fit(cov_type="clustered", clusters=df[COUNTRY_COL])

    coef = iv_model.params["dCAPB"]
    V_h = iv_model.cov.loc["dCAPB", "dCAPB"]
    
    return coef, V_h

# ============================== Compute Observed Chi-Squared ==============================

for sub in subsamples:
    iv_data = subsamples[sub]

    coefs_list = []
    var_matrices = []

    for h in horizons:
        coef, V_h = run_iv_regression(iv_data, h, sub)
        coefs_list.append(coef)
        var_matrices.append(V_h)
        individual_actual_chi2_stat = coef ** 2 / V_h  # chi-squared for individual horizon
        results[sub]["individual_chi2_stats"][h] = individual_actual_chi2_stat  # store per-horizon Chi-squared

    coefs = np.array(coefs_list)
    var_matrix = np.diag(var_matrices)
    
    chi2_stat = coefs.T @ np.linalg.inv(var_matrix) @ coefs  # compute joint Chi-squared across horizons
    p_value = 1 - chi2.cdf(chi2_stat, df=len(coefs))
    results[sub]["chi2_stat"] = chi2_stat
    results[sub]["coefs"] = coefs_list
    results[sub]["var_matrix"] = var_matrices

    print(f"Subsample: {sub}")
    print(f"Joint Chi-Squared: {chi2_stat:.3f}, p-value: {p_value:.3f}")

# ============================== Bootstrap Chi-Squared (Joint) ==============================

def bootstrap_chi2_statistics_for_subsample(simulated_residuals, data, num_bootstraps):
    df_original = data.copy()
    
    def compute_bootstrap_chi2(i):
        df = df_original.copy()
        df["log_gdp_sim_residuals"] = df["residuals_country"].mean() + simulated_residuals[i]
        coefs_list = []
        var_matrices = []
        for h in range(5):  # Loop over horizons
            df[f"log_gdp_h{h}"] = df.groupby("ifs")["log_gdp_sim_residuals"].shift(-h) - df.groupby("ifs")["log_gdp_sim_residuals"].shift(1)
            df[f"log_gdp_h{h}"] = df[f"log_gdp_h{h}"].fillna(0) 

            # apply HP filter to extract cyclical component
            df[f"y_hpcyc_h{h}"] = df[f"log_gdp_h{h}"] - df.groupby("ifs")[f"log_gdp_h{h}"].transform(lambda x: hpfilter(x, lamb=400)[1])
            df[f"y_hpcyc_h{h}"] = df[f"y_hpcyc_h{h}"].fillna(0)

            df = df.dropna(subset=[f"log_gdp_h{h}", "dCAPB", "size", f"y_hpcyc_h{h}"]).copy()
            df["log_gdp_used"] = df[f"log_gdp_h{h}"]

            df["size_transformed"] = df["size"] - df.groupby("ifs")["size"].transform("mean")
            df["dCAPB"] = df["dCAPB"] - df.groupby("ifs")["dCAPB"].transform("mean")

            X = sm.add_constant(df[[f"y_hpcyc_h{h}"]])
            iv_model = IV2SLS(df["log_gdp_used"], X, df[["dCAPB"]], df[["size_transformed"]]).fit(cov_type="clustered", clusters=df["ifs"])

            coef = iv_model.params["dCAPB"]
            V_h = iv_model.cov.loc["dCAPB", "dCAPB"]

            coefs_list.append(coef)
            var_matrices.append(V_h)

        coefs = np.array(coefs_list)
        var_matrix = np.diag(var_matrices)

        return coefs.T @ np.linalg.inv(var_matrix) @ coefs
    
    boot_chi2_stats = Parallel(n_jobs=-1, backend="loky")(delayed(compute_bootstrap_chi2)(i) for i in range(num_bootstraps))
    return boot_chi2_stats

num = NUM_BOOTSTRAPS

# compute bootstrapped Chi-Squared statistics for each subsample
# generate bootstrapped residuals properly
simulated_residuals_full  = bootstrap_for_subsample(subsamples["Full Sample"], num, best_orders["Full Sample"])
simulated_residuals_boom  = bootstrap_for_subsample(subsamples["Boom"],        num, best_orders["Boom"])
simulated_residuals_slump = bootstrap_for_subsample(subsamples["Slump"],       num, best_orders["Slump"])

boot_chi2_stats_boom  = bootstrap_chi2_statistics_for_subsample(simulated_residuals_boom,  subsamples["Boom"],        num)
boot_chi2_stats_slump = bootstrap_chi2_statistics_for_subsample(simulated_residuals_slump, subsamples["Slump"],       num)
boot_chi2_stats_full  = bootstrap_chi2_statistics_for_subsample(simulated_residuals_full,  subsamples["Full Sample"], num)

bootstrap_stats = {
    "Full Sample": boot_chi2_stats_full,
    "Boom":        boot_chi2_stats_boom,
    "Slump":       boot_chi2_stats_slump
}

alpha = 0.05

print("Bootstrap Joint Chi-Squared Tests:")
for sub, boot_stats in bootstrap_stats.items():
    observed_chi2 = results[sub]["chi2_stat"]
    boot_stats = np.array(boot_stats)
    # p-value: proportion of bootstrapped chi2 stats that are greater than or equal to the observed value
    bootstrap_p = np.mean(boot_stats >= observed_chi2)
    # 95th percentile from the bootstrap distribution
    critical_value = np.percentile(boot_stats, 95)
    
    if bootstrap_p < alpha:
        decision = "Reject the null hypothesis"
    else:
        decision = "Do not reject the null hypothesis"
    
    print(f"Subsample '{sub}': Observed chi2 = {observed_chi2:.3f}, "
          f"95th Percentile = {critical_value:.3f}, Bootstrap p-value = {bootstrap_p:.3f} -> {decision}.")

# ============================== Bootstrap Individual Chi-Squared ==============================

def bootstrap_individual_chi2_statistics_for_subsample(simulated_residuals, data, num_bootstraps):
    individual_bootstrap = {h: [] for h in horizons}
    
    def compute_bootstrap_individual(i):
        df = data.copy()
        # cdd bootstrapped residuals: shift the residuals by their mean plus the simulated noise
        df["log_gdp_sim_residuals"] = df["residuals_country"].mean() + simulated_residuals[i]
        stats = {}
        for h in horizons:
            # compute the shifted series for horizon h
            df[f"log_gdp_h{h}"] = df.groupby("ifs")["log_gdp_sim_residuals"].shift(-h) - df.groupby("ifs")["log_gdp_sim_residuals"].shift(1)
            df[f"log_gdp_h{h}"] = df[f"log_gdp_h{h}"].fillna(0)
            
            # apply the HP filter to extract the cyclical component
            df[f"y_hpcyc_h{h}"] = df[f"log_gdp_h{h}"] - df.groupby("ifs")[f"log_gdp_h{h}"].transform(lambda x: hpfilter(x, lamb=400)[1])
            df[f"y_hpcyc_h{h}"] = df[f"y_hpcyc_h{h}"].fillna(0)
            
            df_h = df.dropna(subset=[f"log_gdp_h{h}", "dCAPB", "size", f"y_hpcyc_h{h}"]).copy()
            df_h["log_gdp_used"] = df_h[f"log_gdp_h{h}"]
            df_h["size_transformed"] = df_h["size"] - df_h.groupby("ifs")["size"].transform("mean")
            df_h["dCAPB"] = df_h["dCAPB"] - df_h.groupby("ifs")["dCAPB"].transform("mean")
            
            X = sm.add_constant(df_h[[f"y_hpcyc_h{h}"]])
            try:
                iv_model = IV2SLS(df_h["log_gdp_used"], X, df_h[["dCAPB"]], df_h[["size_transformed"]]).fit(cov_type="clustered", clusters=df_h["ifs"])
                coef = iv_model.params["dCAPB"]
                V_h = iv_model.cov.loc["dCAPB", "dCAPB"]
                stats[h] = coef ** 2 / V_h
            except Exception:
                stats[h] = np.nan
        return stats

    bootstrap_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_bootstrap_individual)(i) for i in range(num_bootstraps)
    )
    
    for result in bootstrap_results:
        for h in horizons:
            individual_bootstrap[h].append(result[h])
    
    for h in horizons:
        individual_bootstrap[h] = np.array(individual_bootstrap[h])
    
    return individual_bootstrap

individual_bootstrap_stats = {}
for sub in subsamples:
    simulated_residuals = bootstrap_for_subsample(subsamples[sub], num, best_orders[sub])
    individual_bootstrap_stats[sub] = bootstrap_individual_chi2_statistics_for_subsample(simulated_residuals, subsamples[sub], num)

# ============================== Plot ==============================

print("Individual Chi-Squared Test Decisions:")
subsample_names = ["Full Sample", "Boom", "Slump"]
for sub in subsample_names:
    print(f"Subsample: {sub}")
    for h in horizons:
        boot_stats = individual_bootstrap_stats[sub][h]
        boot_stats = boot_stats[~np.isnan(boot_stats)]
        critical_val = np.percentile(boot_stats, 95)
        observed_val = results[sub]["individual_chi2_stats"][h]
        decision = "Reject" if observed_val > critical_val else "Do not reject"
        print(f"  Horizon {h}: Observed = {observed_val:.3f}, 95th Percentile = {critical_val:.3f} -> {decision} the null.")
    print("\n")

fig, axes = plt.subplots(nrows=len(subsample_names), ncols=len(horizons), figsize=(20, 12), sharex=False, sharey=False)

for row, sub in enumerate(subsample_names):
    for col, h in enumerate(horizons):
        ax = axes[row, col]
        boot_stats = individual_bootstrap_stats[sub][h]
        boot_stats = boot_stats[~np.isnan(boot_stats)]
        
        sns.histplot(boot_stats, bins=30, kde=True, ax=ax, color='lightgreen')
        
        observed_val = results[sub]["individual_chi2_stats"][h]
        # bootstrap critical value (95th percentile)
        critical_val = np.percentile(boot_stats, 95)
        
        # plot the observed individual chi2 - solid black vertical line
        ax.axvline(observed_val, color='black', linestyle='solid', linewidth=2, label='Observed' if (row==0 and col==0) else "")
        # plot the bootstrap 95th percentile - dashed red vertical line
        ax.axvline(critical_val, color='red', linestyle='dashed', linewidth=2, label='95th Percentile' if (row==0 and col==0) else "")
        
        ax.set_title(f"{sub} - Horizon {h}")
        ax.set_xlabel("Chi-Squared Statistic")
        ax.set_ylabel("Frequency")
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ============================== Save Bootstrap χ² Summary (unchanged logic) ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
chi2_summary_path = os.path.join(OUTPUT_DIR, f"bootstrap_joint_chi2_summary.csv")

summary_rows = []
for name, dist in bootstrap_stats.items():
    obs = results[name]["chi2_stat"]
    dist = np.array(dist)
    p_emp = float(np.mean(dist >= obs))
    q95   = float(np.percentile(dist, 95))
    decision = "Reject the null hypothesis" if p_emp < 0.05 else "Do not reject the null hypothesis"
    summary_rows.append({
        "subsample": name,
        "observed_chi2": obs,
        "bootstrap_95th_percentile": q95,
        "bootstrap_p_value": p_emp,
        "decision": decision
    })

pd.DataFrame(summary_rows).to_csv(chi2_summary_path, index=False)
print(f"\n[SAVE] Bootstrap χ² summary exported to: {chi2_summary_path}")