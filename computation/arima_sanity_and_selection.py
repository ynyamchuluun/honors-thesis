"""
ARIMA Order Selection & Diagnostics
---------------------------------------

Select a single common ARIMA(p,0,q) order on the full-sample country-trend residuals
and run basic diagnostics. Persist the chosen order to JSON so the main bootstrap script
can reuse it.

Diagnostics

1) Construct residuals: OLS(log_gdp ~ 1 + year) within country.
2) Stationarity (ADF) on full-sample residuals.
3) Grid search over (p,q) with d=0 using BIC (default; use AIC if desired).
4) Optional: AR-root stability check (all roots outside unit circle).
5) Save the chosen order to: output/common_arima_order.json

Configuration

- Grids: p ∈ {0,1,2,3}, q ∈ {0,1,2}. Adjust as needed.
- Criterion: AIC.

Outputs

- Printed diagnostics and selected order.
- JSON file with {"common_order": [p,0,q], "criterion": "...", "score": ...}

"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# -------- Config --------
DATA_PATH   = "input/iv_data_corrected.csv"
OUT_JSON    = "output/common_arima_order.json"
COUNTRY_COL = "ifs"
P_GRID      = range(4)        # p in {0,1,2,3}
Q_GRID      = range(3)        # q in {0,1,2}
CRITERION   = "aic"           # "bic" or "aic"

# -------- Load & residuals --------
df = pd.read_csv(DATA_PATH).reset_index(drop=True)
df["log_gdp"] = np.log(df["rgdpbarro"]).fillna(df["rgdpbarro"].mean())
df["residuals_country"] = df["log_gdp"] - df.groupby(COUNTRY_COL)["log_gdp"].transform(
    lambda x: sm.OLS(x, sm.add_constant(df.loc[x.index, "year"]))
               .fit()
               .predict(sm.add_constant(df.loc[x.index, "year"]))
)

# -------- ADF stationarity on full-sample residuals --------
res = df["residuals_country"].dropna()
adf_stat, adf_p, _, _, adf_crit, _ = adfuller(res, autolag="AIC")
print("[ADF] Full-sample residuals:",
      f"stat={adf_stat:.3f}, p={adf_p:.3f}, crit={ {k: round(v,3) for k,v in adf_crit.items()} }",
      "-> reject unit root" if adf_p < 0.05 else "-> cannot reject")

# -------- Grid search for common order (full sample) --------
best_score = np.inf
best_order = None

for p in P_GRID:
    for q in Q_GRID:
        try:
            fit = ARIMA(res, order=(p,0,q)).fit()
            score = fit.bic if CRITERION.lower() == "bic" else fit.aic
            if score < best_score:
                best_score, best_order = score, (p,0,q)
        except Exception:
            continue

if best_order is None:
    raise RuntimeError("ARIMA order selection failed on the full sample.")

print(f"[SELECT] Common ARIMA order by {CRITERION.upper()}: {best_order} (score={best_score:.2f})")

# -------- Persist selection to JSON --------
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({
        "common_order": list(best_order),
        "criterion": CRITERION.upper(),
        "score": float(best_score)
    }, f, indent=2)

print(f"[WRITE] Saved common order to: {OUT_JSON}")