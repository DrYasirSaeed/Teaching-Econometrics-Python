"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 09 - Nonstationarity and Unit Root Tests
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 9 - Unit Root Tests
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : KSE-100 Index and Exchange Rate (PKR/USD)
           Two Pakistan time series tested for unit roots.
           Asteriou uses exchange rate and interest rate
           series for the UK; replaced here with PSX and
           SBP-relevant series directly familiar to scholars.
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
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Simulate Pakistan Time Series
#   KSE-100 index: random walk (non-stationary, I(1))
#   Exchange rate: random walk with drift
#   KSE returns   : stationary (differenced KSE)
# ===================================================
np.random.seed(55)
T = 120   # 10 years of monthly data

# Random walk: KSE-100 (I(1))
shocks_kse = np.random.normal(0, 250, T)
kse100     = np.cumsum(shocks_kse) + 30000

# Random walk with drift: PKR/USD
shocks_fx  = np.random.normal(0, 0.5, T)
exchange   = np.cumsum(shocks_fx + 0.1) + 105   # Drift ~0.1/month

# First differences (should be stationary)
d_kse100   = np.diff(kse100)
d_exchange = np.diff(exchange)

dates        = pd.date_range(start='2014-01', periods=T,   freq='ME')
dates_diff   = pd.date_range(start='2014-02', periods=T-1, freq='ME')

df_levels = pd.DataFrame({'KSE100': kse100, 'Exchange': exchange}, index=dates)
df_diff   = pd.DataFrame({'dKSE100': d_kse100, 'dExchange': d_exchange}, index=dates_diff)

print("=" * 60)
print("  ASTERIOU CH.9 | Unit Root Tests")
print("  KSE-100 Index and PKR/USD Exchange Rate")
print("=" * 60)

# ===================================================
# 📌 Step 2: Time Series Plots (Level vs. Differenced)
# ===================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

df_levels['KSE100'].plot(ax=axes[0, 0], color='navy')
axes[0, 0].set_title("KSE-100 Index (Levels)", fontsize=12)
axes[0, 0].set_ylabel("Index Points")

df_diff['dKSE100'].plot(ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title("KSE-100 Returns (First Difference)", fontsize=12)
axes[0, 1].axhline(0, color='red', lw=1, linestyle='--')

df_levels['Exchange'].plot(ax=axes[1, 0], color='darkgreen')
axes[1, 0].set_title("PKR/USD Exchange Rate (Levels)", fontsize=12)
axes[1, 0].set_ylabel("PKR per USD")

df_diff['dExchange'].plot(ax=axes[1, 1], color='olive')
axes[1, 1].set_title("Exchange Rate Change (First Difference)", fontsize=12)
axes[1, 1].axhline(0, color='red', lw=1, linestyle='--')

plt.suptitle("Levels vs. First Differences: Stationarity Inspection", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 3: ACF/PACF Plots
#   Slow decay of ACF = non-stationarity signal
#   Asteriou emphasises this visual before formal tests
# ===================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
plot_acf( df_levels['KSE100'],   ax=axes[0, 0], lags=20, title="ACF — KSE-100 Levels")
plot_acf( df_diff['dKSE100'],    ax=axes[0, 1], lags=20, title="ACF — KSE-100 Differenced")
plot_acf( df_levels['Exchange'], ax=axes[1, 0], lags=20, title="ACF — Exchange Rate Levels")
plot_acf( df_diff['dExchange'],  ax=axes[1, 1], lags=20, title="ACF — Exchange Rate Differenced")
plt.suptitle("ACF: Non-Stationarity Detection via Correlogram", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 4: ADF Test Function (Asteriou's Standard)
#   Runs ADF and prints results in Asteriou table format
# ===================================================
def run_adf(series, name, maxlag=12):
    print(f"\n--- ADF Test: {name} ---")
    for reg in ['n', 'c', 'ct']:
        result = adfuller(series, maxlag=maxlag, regression=reg, autolag='AIC')
        t_stat, p_val, n_lags, n_obs = result[0], result[1], result[2], result[3]
        crit    = result[4]
        label   = {'n': 'No Constant, No Trend',
                   'c': 'Constant Only',
                   'ct': 'Constant + Trend'}[reg]
        print(f"\n  Specification: {label}")
        print(f"  ADF t-stat : {t_stat:.4f}")
        print(f"  p-value    : {p_val:.4f}")
        print(f"  Lags used  : {n_lags}")
        print(f"  Critical Values: 1%={crit['1%']:.3f}  5%={crit['5%']:.3f}  10%={crit['10%']:.3f}")
        if p_val < 0.05:
            print(f"  Decision   : Reject H0 — Series is STATIONARY")
        else:
            print(f"  Decision   : Fail to Reject H0 — Series is NON-STATIONARY (Unit Root)")

run_adf(df_levels['KSE100'],   "KSE-100 Levels")
run_adf(df_diff['dKSE100'],    "KSE-100 First Difference")
run_adf(df_levels['Exchange'], "Exchange Rate Levels")
run_adf(df_diff['dExchange'],  "Exchange Rate First Difference")

# ===================================================
# 📌 Step 5: KPSS Test (Kwiatkowski et al.)
#   Asteriou Ch.9 notes KPSS has H0: stationarity
#   Compare with ADF to check for contradictions
# ===================================================
def run_kpss(series, name):
    print(f"\n--- KPSS Test: {name} ---")
    for reg in ['c', 'ct']:
        result = kpss(series, regression=reg, nlags='auto')
        t_stat, p_val, n_lags, crit = result
        label  = {'c': 'Level Stationarity', 'ct': 'Trend Stationarity'}[reg]
        print(f"\n  H0: Series is Stationary ({label})")
        print(f"  KPSS stat  : {t_stat:.4f}")
        print(f"  p-value    : {p_val:.4f}")
        print(f"  Critical Values: 1%={crit['1%']:.3f}  5%={crit['5%']:.3f}  10%={crit['10%']:.3f}")
        if p_val < 0.05:
            print(f"  Decision   : Reject H0 — NON-STATIONARY")
        else:
            print(f"  Decision   : Fail to Reject H0 — STATIONARY")

run_kpss(df_levels['KSE100'],   "KSE-100 Levels")
run_kpss(df_diff['dKSE100'],    "KSE-100 First Difference")
run_kpss(df_levels['Exchange'], "Exchange Rate Levels")
run_kpss(df_diff['dExchange'],  "Exchange Rate First Difference")

# ===================================================
# 📌 Step 6: Summary Table (Asteriou-style reporting)
# ===================================================
print("\n--- Summary: Order of Integration ---")
summary = pd.DataFrame({
    'Series'         : ['KSE-100 Levels', 'KSE-100 Diff',
                        'Exchange Rate Levels', 'Exchange Rate Diff'],
    'ADF Decision'   : ['Non-stationary I(1)', 'Stationary I(0)',
                        'Non-stationary I(1)', 'Stationary I(0)'],
    'KPSS Decision'  : ['Non-stationary I(1)', 'Stationary I(0)',
                        'Non-stationary I(1)', 'Stationary I(0)'],
    'Integration Order': ['I(1)', 'I(0)', 'I(1)', 'I(0)']
})
print(summary.to_string(index=False))
print("\n  Both KSE-100 and Exchange Rate are I(1) processes.")
print("  Their first differences are I(0) — suitable for cointegration analysis.")
print("  Proceed to Chapter 10 (Cointegration) for further analysis.")

print("\n--- Analysis Complete ---")
