"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 05 - Heteroscedasticity
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 5 - Heteroscedasticity
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Firm Size and Investment Expenditure
           INV = B0 + B1*SIZE + u
           Larger firms tend to have larger variance in
           investment, a classic heteroscedasticity setup
           matching Asteriou Ch.5 cross-sectional example.
           Pakistan-listed firms (PSX) analog data used.
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
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    het_goldfeldquandt
)

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Simulate Cross-Section Data (PSX firms)
#   Deliberately introduce heteroscedasticity:
#   variance in investment grows with firm size
#   This mirrors Asteriou's Ch.5 opening illustration
# ===================================================
np.random.seed(7)
n         = 60
firm_size = np.sort(np.random.uniform(50, 2000, n))   # Total assets, PKR millions

# Error term with variance proportional to firm size
sigma    = 0.05 * firm_size
error    = np.array([np.random.normal(0, s) for s in sigma])

investment = 5 + 0.08 * firm_size + error

df = pd.DataFrame({'FirmSize': firm_size, 'Investment': investment})

print("=" * 58)
print("  ASTERIOU CH.5 | Heteroscedasticity")
print("  INV = B0 + B1*FirmSize + u")
print("  (Variance of u grows with FirmSize)")
print("=" * 58)

print("\n--- Descriptive Statistics ---")
print(df.describe().round(2))

# ===================================================
# 📌 Step 2: Scatter Plot (Visualise the Funnel Shape)
# ===================================================
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='FirmSize', y='Investment', color='steelblue', s=55)
plt.title("Firm Size vs. Investment\n(Note the Funnel Pattern — Heteroscedasticity)", fontsize=13)
plt.xlabel("Firm Size (PKR Millions)")
plt.ylabel("Investment Expenditure (PKR Millions)")
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 3: OLS Estimation (Naive, Ignoring Problem)
# ===================================================
X     = sm.add_constant(df['FirmSize'])
y     = df['Investment']
model = sm.OLS(y, X).fit()

print("\n--- OLS Estimates (Naive: Ignoring Heteroscedasticity) ---")
print(model.summary())

# ===================================================
# 📌 Step 4: Residual Plots for Detection
#   Asteriou recommends plotting residuals vs fitted
#   and residuals vs each regressor as a first-pass
# ===================================================
df['Fitted']   = model.fittedvalues
df['Residuals'] = model.resid
df['AbsResid']  = np.abs(model.resid)
df['SqResid']   = model.resid ** 2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(df['FirmSize'], df['Residuals'], color='steelblue', s=40, alpha=0.7)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title("Residuals vs. Firm Size", fontsize=12)
axes[0].set_xlabel("Firm Size")
axes[0].set_ylabel("Residuals")

axes[1].scatter(df['Fitted'], df['Residuals'], color='darkorange', s=40, alpha=0.7)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title("Residuals vs. Fitted Values", fontsize=12)
axes[1].set_xlabel("Fitted Values")
axes[1].set_ylabel("Residuals")

plt.suptitle("Residual Diagnostics: Heteroscedasticity Detection", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 5: Formal Tests for Heteroscedasticity
# ===================================================

# 5A: Breusch-Pagan Test (Asteriou Ch.5 primary test)
bp      = het_breuschpagan(model.resid, X)
bp_dict = dict(zip(['LM Stat', 'LM p-value', 'F Stat', 'F p-value'], bp))
print("\n--- Breusch-Pagan Test ---")
for k, v in bp_dict.items():
    print(f"  {k}: {v:.4f}")
if bp_dict['LM p-value'] < 0.05:
    print("  Decision: Reject H0 — Heteroscedasticity is present.")
else:
    print("  Decision: Fail to Reject H0 — Homoscedasticity assumed.")

# 5B: White Test (Asteriou Ch.5 alternative)
wt      = het_white(model.resid, X)
wt_dict = dict(zip(['LM Stat', 'LM p-value', 'F Stat', 'F p-value'], wt))
print("\n--- White Test ---")
for k, v in wt_dict.items():
    print(f"  {k}: {v:.4f}")
if wt_dict['LM p-value'] < 0.05:
    print("  Decision: Reject H0 — Heteroscedasticity confirmed.")
else:
    print("  Decision: Fail to Reject H0.")

# 5C: Goldfeld-Quandt Test (split-sample, Asteriou Ch.5)
gq      = het_goldfeldquandt(y, X)
gq_dict = dict(zip(['F Stat', 'p-value', 'Alternative'], gq))
print("\n--- Goldfeld-Quandt Test ---")
print(f"  F Stat : {gq_dict['F Stat']:.4f}")
print(f"  p-value: {gq_dict['p-value']:.4f}")
if gq_dict['p-value'] < 0.05:
    print("  Decision: Reject H0 — Variance differs across subsamples.")

# ===================================================
# 📌 Step 6: Correction Methods
# ===================================================

# 6A: Heteroscedasticity-Consistent (HC) Standard Errors
#     White (1980) robust SEs — Asteriou Ch.5 preferred fix
model_robust = sm.OLS(y, X).fit(cov_type='HC3')
print("\n--- OLS with Robust (HC3) Standard Errors ---")
print(model_robust.summary())

# 6B: Weighted Least Squares (WLS)
#     Asteriou Ch.5: use 1/variance as weights
#     Here we use 1/FirmSize as a proxy for 1/sigma^2
weights  = 1.0 / df['FirmSize']
model_wls = sm.WLS(y, X, weights=weights).fit()
print("\n--- WLS Estimates (Weight = 1/FirmSize) ---")
print(model_wls.summary())

# ===================================================
# 📌 Step 7: Side-by-Side Comparison
# ===================================================
print("\n--- Comparison: OLS vs. Robust SE vs. WLS ---")
comparison = pd.DataFrame({
    'OLS Coef'         : model.params.round(4),
    'OLS SE'           : model.bse.round(4),
    'Robust (HC3) SE'  : model_robust.bse.round(4),
    'WLS Coef'         : model_wls.params.round(4),
    'WLS SE'           : model_wls.bse.round(4)
})
print(comparison)
print("\n  Note: Robust SEs correct inference without changing coefficients.")
print("  WLS changes both coefficients and SEs by reweighting observations.")

print("\n--- Analysis Complete ---")
