"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 06 - Autocorrelation (Serial Correlation)
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 6 - Autocorrelation
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Money Demand Function (Pakistan, quarterly)
           M2 = B0 + B1*GDP + B2*Interest + u
           Time-series data often exhibits AR(1) errors.
           Asteriou uses UK money demand as the base case;
           replaced here with Pakistan monetary aggregate
           data (SBP context) for practical local relevance.
============================================================
"""

# ===================================================
# 📌 Step 0: Import Libraries
# ===================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Simulate Quarterly Pakistan Money Demand
#   AR(1) error structure: u_t = rho * u_{t-1} + e_t
#   rho = 0.75 (moderately high autocorrelation)
#   This mimics Asteriou's simulated AR(1) example in Ch.6
# ===================================================
np.random.seed(101)
T   = 60                           # 15 years of quarterly data
rho = 0.75                         # AR(1) coefficient

gdp      = np.cumsum(np.random.normal(50, 5, T)) + 500   # GDP proxy (PKR Bn)
interest = 8 + np.random.normal(0, 1.5, T)               # SBP policy rate (%)

# Generate AR(1) error
e   = np.random.normal(0, 8, T)
u   = np.zeros(T)
u[0] = e[0]
for t in range(1, T):
    u[t] = rho * u[t-1] + e[t]

m2 = 10 + 0.85 * gdp - 3.5 * interest + u               # M2 (PKR Bn)

quarters = pd.date_range(start='2008Q1', periods=T, freq='QE')

df = pd.DataFrame({
    'Quarter' : quarters,
    'M2'      : m2,
    'GDP'     : gdp,
    'Interest': interest
})
df.set_index('Quarter', inplace=True)

print("=" * 58)
print("  ASTERIOU CH.6 | Autocorrelation")
print("  M2 = B0 + B1*GDP + B2*Interest + u")
print("  (u follows AR(1) process with rho = 0.75)")
print("=" * 58)

# ===================================================
# 📌 Step 2: Time Series Plots
# ===================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
df['M2'].plot(ax=axes[0], color='navy')
axes[0].set_title("M2 Money Supply (PKR Bn)", fontsize=12)

df['GDP'].plot(ax=axes[1], color='darkgreen')
axes[1].set_title("GDP Proxy (PKR Bn)", fontsize=12)

df['Interest'].plot(ax=axes[2], color='crimson')
axes[2].set_title("SBP Policy Rate (%)", fontsize=12)

plt.suptitle("Pakistan Money Demand Variables (Quarterly)", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 3: OLS Estimation (Naive, Ignoring AR)
# ===================================================
X     = sm.add_constant(df[['GDP', 'Interest']])
y     = df['M2']
model = sm.OLS(y, X).fit()

print("\n--- OLS Estimates (Ignoring Autocorrelation) ---")
print(model.summary())

df['Residuals'] = model.resid

# ===================================================
# 📌 Step 4: Residual Time Plot
#   Asteriou's first visual check: do residuals trend?
# ===================================================
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['Residuals'], color='steelblue', marker='o', markersize=3, lw=1.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("OLS Residuals Over Time\n(Persistent positive clusters signal autocorrelation)", fontsize=13)
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 5: Correlogram (ACF and PACF)
#   Asteriou Ch.6: examine up to 12 lags for quarterly
# ===================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(df['Residuals'],  ax=axes[0], lags=20, title="ACF of Residuals")
plot_pacf(df['Residuals'], ax=axes[1], lags=20, title="PACF of Residuals")
plt.suptitle("Correlogram: Residual Autocorrelation Structure", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 6: Formal Tests for Autocorrelation
# ===================================================

# 6A: Durbin-Watson Test (Asteriou Ch.6, most cited test)
dw = durbin_watson(df['Residuals'])
print(f"\n--- Durbin-Watson Test ---")
print(f"  DW Statistic: {dw:.4f}")
print(f"  Rule of thumb: DW near 2 = no autocorrelation")
print(f"  DW < 1.5 suggests positive autocorrelation")
print(f"  DW > 2.5 suggests negative autocorrelation")
if dw < 1.5:
    print("  Decision: Positive autocorrelation likely present.")
elif dw > 2.5:
    print("  Decision: Negative autocorrelation likely present.")
else:
    print("  Decision: No strong evidence of autocorrelation.")

# 6B: Breusch-Godfrey LM Test (Asteriou's preferred test)
bg_test   = acorr_breusch_godfrey(model, nlags=4)
bg_labels = ['LM Stat', 'LM p-value', 'F Stat', 'F p-value']
bg_dict   = dict(zip(bg_labels, bg_test))
print(f"\n--- Breusch-Godfrey LM Test (4 lags) ---")
for k, v in bg_dict.items():
    print(f"  {k}: {v:.4f}")
if bg_dict['LM p-value'] < 0.05:
    print("  Decision: Reject H0 — Serial correlation detected.")
else:
    print("  Decision: Fail to Reject H0 — No serial correlation.")

# 6C: Ljung-Box Test
lb = acorr_ljungbox(df['Residuals'], lags=[4, 8, 12], return_df=True)
print(f"\n--- Ljung-Box Test ---")
print(lb.round(4))

# ===================================================
# 📌 Step 7: Cochrane-Orcutt Correction
#   Asteriou Ch.6 primary correction method
#   Iteratively estimates rho and re-runs GLS
# ===================================================
from statsmodels.regression.linear_model import GLSAR

model_co = GLSAR(y, X, rho=1).iterative_fit(maxiter=100)
print("\n--- Cochrane-Orcutt (GLSAR) Corrected Estimates ---")
print(model_co.summary())

# ===================================================
# 📌 Step 8: Newey-West HAC Standard Errors
#   Asteriou Ch.6 Section: robust SEs without GLS
# ===================================================
model_hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print("\n--- OLS with Newey-West HAC Standard Errors ---")
print(model_hac.summary())

# ===================================================
# 📌 Step 9: Comparison Table
# ===================================================
print("\n--- Comparison: OLS vs. Cochrane-Orcutt vs. HAC ---")
comp = pd.DataFrame({
    'OLS Coef'   : model.params.round(4),
    'OLS SE'     : model.bse.round(4),
    'CO Coef'    : model_co.params.round(4),
    'CO SE'      : model_co.bse.round(4),
    'HAC SE'     : model_hac.bse.round(4),
})
print(comp)

print("\n--- Analysis Complete ---")
