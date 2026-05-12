"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 10 - Cointegration
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 10 - Cointegration and Error Correction Models
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Monetary Approach to Exchange Rate
           Log(M2) and Log(Exchange Rate) — Pakistan
           If both are I(1) and cointegrated, a long-run
           equilibrium relationship exists between money
           supply and exchange rate, consistent with
           monetary model of the exchange rate.
           Asteriou uses log GDP and log consumption (UK);
           replaced with Pakistan monetary variables.
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
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.stattools import adfuller

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Simulate Cointegrated Series
#   Both series share a common stochastic trend (I(1))
#   but their linear combination is I(0) — cointegrated
# ===================================================
np.random.seed(77)
T = 120

# Common stochastic trend
common_trend = np.cumsum(np.random.normal(0, 0.02, T))

# Log M2 drifts upward with common trend
log_m2 = 10 + 0.5 * np.arange(T) * 0.01 + common_trend + np.random.normal(0, 0.05, T)

# Log Exchange Rate shares the trend but with a long-run coefficient
log_er = 3.5 + 0.75 * common_trend + np.random.normal(0, 0.03, T)

dates = pd.date_range(start='2014-01', periods=T, freq='ME')
df    = pd.DataFrame({'Log_M2': log_m2, 'Log_ER': log_er}, index=dates)

print("=" * 60)
print("  ASTERIOU CH.10 | Cointegration & ECM")
print("  Log(M2) and Log(Exchange Rate) — Pakistan Analog")
print("=" * 60)

# ===================================================
# 📌 Step 2: Time Series Plots
# ===================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
df['Log_M2'].plot(ax=axes[0], color='navy', lw=1.5)
axes[0].set_title("Log M2 Money Supply (Pakistan)", fontsize=12)
axes[0].set_ylabel("Log M2")

df['Log_ER'].plot(ax=axes[1], color='crimson', lw=1.5)
axes[1].set_title("Log PKR/USD Exchange Rate", fontsize=12)
axes[1].set_ylabel("Log Exchange Rate")

plt.suptitle("Two Potentially Cointegrated I(1) Series", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 3: Pre-Test — Confirm Both Series Are I(1)
# ===================================================
def adf_summary(series, name):
    result = adfuller(series, regression='c', autolag='AIC')
    print(f"  {name}: ADF = {result[0]:.4f}, p = {result[1]:.4f}", end=' ')
    print("=> I(1) likely" if result[1] > 0.05 else "=> I(0) likely")

print("\n--- Pre-Test: ADF on Levels ---")
adf_summary(df['Log_M2'], "Log_M2")
adf_summary(df['Log_ER'], "Log_ER")

print("\n--- Pre-Test: ADF on First Differences ---")
adf_summary(np.diff(df['Log_M2']), "d(Log_M2)")
adf_summary(np.diff(df['Log_ER']), "d(Log_ER)")

# ===================================================
# 📌 Step 4: Engle-Granger Two-Step Method
#   Step 1: Regress y on x (long-run equation)
#   Step 2: Test residuals for stationarity
#   Asteriou Ch.10 covers this as the baseline method
# ===================================================
print("\n--- Engle-Granger Step 1: Long-Run OLS ---")
X_lr  = sm.add_constant(df['Log_M2'])
y_lr  = df['Log_ER']
model_lr = sm.OLS(y_lr, X_lr).fit()
print(model_lr.summary())

df['ECT'] = model_lr.resid   # Error Correction Term (residuals)

print("\n--- Engle-Granger Step 2: ADF on Residuals ---")
ect_adf = adfuller(df['ECT'], regression='n', autolag='AIC')
print(f"  ADF on ECT: t-stat = {ect_adf[0]:.4f}, p-value = {ect_adf[1]:.4f}")
crit = ect_adf[4]
print(f"  Critical Values: 1%={crit['1%']:.3f}  5%={crit['5%']:.3f}  10%={crit['10%']:.3f}")

# Note: Engle-Granger critical values for residuals differ from standard ADF
# (MacKinnon 1991 adjusted values)
print("\n  Note: Use MacKinnon (1991) adjusted critical values for residual-based tests.")
print("  Approximate adjusted 5% CV for 2 variables, T=120: ~ -3.37")
if ect_adf[0] < -3.37:
    print("  Decision: Residuals are I(0) — Cointegration confirmed.")
else:
    print("  Decision: Residuals not clearly I(0) — Cointegration not confirmed.")

# ===================================================
# 📌 Step 5: Engle-Granger Convenience Function
# ===================================================
eg_stat, eg_pval, _ = coint(df['Log_ER'], df['Log_M2'])
print(f"\n--- statsmodels coint() Test ---")
print(f"  EG t-stat: {eg_stat:.4f}")
print(f"  p-value  : {eg_pval:.4f}")
if eg_pval < 0.05:
    print("  Decision: Cointegration confirmed (5% level).")
else:
    print("  Decision: No cointegration detected.")

# ===================================================
# 📌 Step 6: Johansen Cointegration Test
#   Asteriou Ch.10 prefers Johansen for multivariate
#   Trace statistic and Max Eigenvalue statistic
# ===================================================
print("\n--- Johansen Cointegration Test ---")
data_johansen = df[['Log_ER', 'Log_M2']].values
result_johansen = coint_johansen(data_johansen, det_order=1, k_ar_diff=2)

print("\n  Trace Statistic:")
print(f"  r=0  : stat={result_johansen.lr1[0]:.4f}  CVs (90/95/99%): {result_johansen.cvt[0]}")
print(f"  r<=1 : stat={result_johansen.lr1[1]:.4f}  CVs (90/95/99%): {result_johansen.cvt[1]}")

print("\n  Max Eigenvalue Statistic:")
print(f"  r=0  : stat={result_johansen.lr2[0]:.4f}  CVs (90/95/99%): {result_johansen.cvm[0]}")
print(f"  r<=1 : stat={result_johansen.lr2[1]:.4f}  CVs (90/95/99%): {result_johansen.cvm[1]}")

if result_johansen.lr1[0] > result_johansen.cvt[0, 1]:
    print("\n  Decision: Reject H0(r=0) — At least one cointegrating vector exists.")
else:
    print("\n  Decision: Cannot reject H0(r=0) — No cointegration.")

# ===================================================
# 📌 Step 7: Error Correction Model (ECM)
#   Asteriou Ch.10 core result:
#   If cointegrated, model the short-run dynamics
#   via ECM — the ECT coefficient captures speed of
#   adjustment back to long-run equilibrium
# ===================================================
print("\n--- Error Correction Model (ECM) ---")

d_log_er = np.diff(df['Log_ER'].values)
d_log_m2 = np.diff(df['Log_M2'].values)
ect_lag  = df['ECT'].values[:-1]

ecm_df   = pd.DataFrame({
    'd_log_er': d_log_er,
    'd_log_m2': d_log_m2,
    'ECT_lag' : ect_lag
})

X_ecm     = sm.add_constant(ecm_df[['d_log_m2', 'ECT_lag']])
y_ecm     = ecm_df['d_log_er']
model_ecm = sm.OLS(y_ecm, X_ecm).fit()

print(model_ecm.summary())

ect_coef = model_ecm.params['ECT_lag']
print(f"\n  ECT Coefficient: {ect_coef:.4f}")
print(f"  Interpretation: {abs(ect_coef)*100:.1f}% of last period's deviation")
print(f"  from the long-run equilibrium is corrected each month.")
if ect_coef < 0 and model_ecm.pvalues['ECT_lag'] < 0.05:
    print("  Sign is negative and significant — valid error correction.")

# ===================================================
# 📌 Step 8: Plot ECT (Long-run Equilibrium Deviations)
# ===================================================
plt.figure(figsize=(10, 4))
df['ECT'].plot(color='darkred', lw=1.5)
plt.axhline(0, color='black', linestyle='--')
plt.title("Error Correction Term (Deviations from Long-Run Equilibrium)", fontsize=13)
plt.ylabel("Residual (ECT)")
plt.tight_layout()
plt.show()

print("\n--- Analysis Complete ---")
