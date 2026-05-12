"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 11 - Vector Autoregressive Models (VAR)
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 11 - Vector Autoregressive Models
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Macroeconomic Trivariate VAR
           Variables: Industrial Production, M2, Interest Rate
           Pakistan quarterly data analog.
           Asteriou uses a bivariate GDP-Inflation VAR for UK;
           extended here to trivariate Pakistan macro system
           which is more realistic for scholars working in
           monetary and financial economics.
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
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Simulate Stationary Macro Data
#   VAR requires stationary inputs — use growth rates
#   IP growth, M2 growth, and interest rate change
#   Mild cross-variable dynamics included
# ===================================================
np.random.seed(99)
T  = 80

# Simulate a simple reduced-form VAR(1) system
# y_t = A * y_{t-1} + e_t
A = np.array([
    [0.40,  0.15, -0.10],   # IP growth equation
    [0.10,  0.55,  0.05],   # M2 growth equation
    [-0.05, 0.08,  0.45]    # Interest rate change equation
])

cov_e = np.array([
    [1.00, 0.30, -0.15],
    [0.30, 0.80,  0.10],
    [-0.15, 0.10, 0.50]
])

y      = np.zeros((T, 3))
errors = np.random.multivariate_normal([0, 0, 0], cov_e, T)
for t in range(1, T):
    y[t] = A @ y[t-1] + errors[t]

dates = pd.date_range(start='2005Q1', periods=T, freq='QE')
df    = pd.DataFrame(y, index=dates, columns=['IP_growth', 'M2_growth', 'dInterest'])

print("=" * 60)
print("  ASTERIOU CH.11 | Vector Autoregressive (VAR) Model")
print("  Trivariate: IP Growth, M2 Growth, Interest Rate Change")
print("=" * 60)

# ===================================================
# 📌 Step 2: Data Overview and Stationarity Check
# ===================================================
print("\n--- Descriptive Statistics ---")
print(df.describe().round(3))

print("\n--- ADF Tests (Verify Stationarity) ---")
for col in df.columns:
    result = adfuller(df[col], regression='c', autolag='AIC')
    print(f"  {col}: ADF = {result[0]:.4f}, p = {result[1]:.4f}", end=' ')
    print("=> Stationary" if result[1] < 0.05 else "=> Possibly Non-stationary")

# ===================================================
# 📌 Step 3: Time Series Plot
# ===================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
colors = ['navy', 'darkgreen', 'crimson']
for i, (col, c) in enumerate(zip(df.columns, colors)):
    df[col].plot(ax=axes[i], color=c, lw=1.5)
    axes[i].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[i].set_title(col, fontsize=12)
plt.suptitle("Pakistan Macro Variables: VAR Inputs", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 4: Lag Order Selection
#   Asteriou Ch.11 uses AIC, SBC (BIC), HQ criteria
#   Select the lag length minimising information criteria
# ===================================================
model_var = VAR(df)
lag_order = model_var.select_order(maxlags=8)
print("\n--- Lag Order Selection ---")
print(lag_order.summary())

optimal_lag = lag_order.aic   # Using AIC (Asteriou default for forecasting)
print(f"\n  Optimal lag (AIC): p = {optimal_lag}")

# ===================================================
# 📌 Step 5: Estimate VAR Model
# ===================================================
p       = max(optimal_lag, 1)
var_fit = model_var.fit(p)

print("\n--- VAR Model Estimation Results ---")
print(var_fit.summary())

# ===================================================
# 📌 Step 6: Granger Causality Tests
#   Asteriou Ch.11: central use of VAR — who causes whom?
#   Tests each variable as a potential cause of others
# ===================================================
print("\n--- Granger Causality Tests ---")
for caused in df.columns:
    for causing in df.columns:
        if caused != causing:
            gc = var_fit.test_causality(caused, causing, kind='f')
            print(f"  H0: {causing} does NOT Granger-cause {caused}")
            print(f"  F-stat: {gc.test_statistic:.4f}  p-value: {gc.pvalue:.4f}", end=' ')
            print("=> Reject H0" if gc.pvalue < 0.05 else "=> Fail to Reject")

# ===================================================
# 📌 Step 7: Impulse Response Functions (IRF)
#   Asteriou Ch.11: how does system respond to shocks?
#   Core analytical output of VAR models
# ===================================================
irf = var_fit.irf(periods=12)
irf.plot(orth=False, figsize=(13, 9))
plt.suptitle("Impulse Response Functions (12-Quarter Horizon)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# Orthogonalised IRFs (Cholesky decomposition)
irf.plot(orth=True, figsize=(13, 9))
plt.suptitle("Orthogonalised IRF (Cholesky: IP > M2 > Interest)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 8: Forecast Error Variance Decomposition (FEVD)
#   Asteriou Ch.11: how much variation in each variable
#   is explained by shocks from each other variable?
# ===================================================
fevd = var_fit.fevd(periods=12)
fevd.plot(figsize=(11, 7))
plt.suptitle("Forecast Error Variance Decomposition (12 Quarters)", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

print("\n--- FEVD at 12-quarter horizon ---")
print(fevd.decomp[-1].round(3))
print("\n  Rows = Variable being decomposed")
print("  Cols = Proportion of variance explained by each shock")

# ===================================================
# 📌 Step 9: VAR Forecasting
#   Asteriou Ch.11 ends with forecasting application
# ===================================================
forecast_steps = 8
forecast = var_fit.forecast(df.values[-p:], steps=forecast_steps)

forecast_index = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=3),
    periods=forecast_steps, freq='QE'
)
df_forecast = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
for i, (col, c) in enumerate(zip(df.columns, colors)):
    axes[i].plot(df.index[-20:], df[col].values[-20:], color=c, lw=1.5, label='Historical')
    axes[i].plot(df_forecast.index, df_forecast[col], color='orange',
                 lw=2, linestyle='--', label='Forecast')
    axes[i].set_title(col, fontsize=12)
    axes[i].legend(fontsize=9)
plt.suptitle("VAR Forecasts: 8-Quarter Ahead", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 10: Residual Diagnostics
# ===================================================
print("\n--- Portmanteau Test for Residual Autocorrelation ---")
pt = var_fit.test_whiteness(nlags=12)
print(pt.summary())

print("\n--- Normality Test (Joint) ---")
norm = var_fit.test_normality()
print(norm.summary())

print("\n--- Analysis Complete ---")
