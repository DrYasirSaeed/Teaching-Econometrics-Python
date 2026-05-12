"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 02 - Simple Linear Regression (SLR)
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 2 - The Simple Linear Regression Model
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Keynesian Consumption Function
           C = Beta0 + Beta1 * Y + u
           Pakistani household data analog (PKR, millions)
           Book uses UK data; replaced with Pakistan-relevant
           consumption and income figures for local applicability.
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
from scipy import stats

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Data - Consumption and Disposable Income
#   Pakistan household survey analog (PKR thousands)
#   Inspired by Asteriou Ch.2 Table 2.1 (UK 1970-1994)
#   Adapted to Pakistan context for pedagogical relevance
# ===================================================

# Annual household disposable income (X) and
# private consumption expenditure (Y), 20 observations

income = [
    120, 145, 162, 175, 190, 210, 228, 245, 263, 280,
    298, 315, 333, 352, 370, 389, 408, 425, 443, 460
]

consumption = [
    98, 118, 130, 141, 156, 171, 183, 196, 211, 224,
    237, 250, 265, 281, 295, 309, 325, 340, 353, 368
]

years = list(range(2001, 2021))

df = pd.DataFrame({
    'Year'       : years,
    'Income'     : income,
    'Consumption': consumption
})

print("=" * 55)
print("  ASTERIOU CH.2 | Simple Linear Regression")
print("  Consumption Function: C = B0 + B1*Y + u")
print("=" * 55)

# ===================================================
# 📌 Step 2: Descriptive Statistics
# ===================================================
print("\n--- Descriptive Statistics ---")
print(df[['Income', 'Consumption']].describe().round(2))

# ===================================================
# 📌 Step 3: Scatter Plot
# ===================================================
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Income', y='Consumption', color='navy', s=70, label='Observations')
plt.title("Consumption vs. Disposable Income\n(Pakistan Household Analog)", fontsize=13)
plt.xlabel("Disposable Income (PKR Thousands)")
plt.ylabel("Consumption Expenditure (PKR Thousands)")
plt.legend()
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 4: Manual OLS Calculation (Asteriou Method)
#   Demonstrates the closed-form estimators:
#   Beta1 = Sum[(xi - x_bar)(yi - y_bar)] / Sum[(xi - x_bar)^2]
#   Beta0 = y_bar - Beta1 * x_bar
#   This mirrors the step-by-step derivation in Ch.2
# ===================================================
x     = np.array(income)
y     = np.array(consumption)
x_bar = np.mean(x)
y_bar = np.mean(y)

beta1_manual = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
beta0_manual = y_bar - beta1_manual * x_bar

print(f"\n--- Manual OLS Estimates (Asteriou Closed-Form) ---")
print(f"  Beta0 (Intercept) = {beta0_manual:.4f}")
print(f"  Beta1 (MPC)       = {beta1_manual:.4f}")
print(f"  Interpretation: A 1 PKR thousand rise in income is")
print(f"  associated with a {beta1_manual:.4f} PKR thousand rise in consumption.")
print(f"  This is the Marginal Propensity to Consume (MPC).")

# ===================================================
# 📌 Step 5: OLS via Statsmodels
# ===================================================
X_ols = sm.add_constant(df['Income'])
model = sm.OLS(df['Consumption'], X_ols).fit()

print("\n--- OLS Regression Summary (Statsmodels) ---")
print(model.summary())

# ===================================================
# 📌 Step 6: Fitted Values and Residuals
# ===================================================
df['Fitted']   = model.fittedvalues
df['Residuals'] = model.resid

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Income', y='Consumption', color='navy', label='Observed', s=70)
plt.plot(df['Income'], df['Fitted'], color='crimson', lw=2, label='Fitted Line')
plt.title("OLS Fitted Line: Consumption Function", fontsize=13)
plt.xlabel("Disposable Income (PKR Thousands)")
plt.ylabel("Consumption (PKR Thousands)")
plt.legend()
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 7: Residual Plot (Asteriou Diagnostic Check)
# ===================================================
plt.figure(figsize=(8, 4))
sns.residplot(x=df['Income'], y=df['Consumption'], lowess=True,
              scatter_kws={'color': 'steelblue'}, line_kws={'color': 'red', 'lw': 1.5})
plt.axhline(0, color='black', linestyle='--')
plt.title("Residual Plot: Fitted vs. Residuals", fontsize=13)
plt.xlabel("Disposable Income (PKR Thousands)")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 8: Goodness of Fit
# ===================================================
ess = np.sum((df['Fitted'] - y_bar) ** 2)   # Explained SS
rss = np.sum(df['Residuals'] ** 2)           # Residual SS
tss = np.sum((y - y_bar) ** 2)              # Total SS
r2  = ess / tss

print(f"\n--- Goodness of Fit (Asteriou Ch.2 Table) ---")
print(f"  TSS = {tss:.4f}")
print(f"  ESS = {ess:.4f}")
print(f"  RSS = {rss:.4f}")
print(f"  R^2 = {r2:.4f}  (Model explains {r2*100:.1f}% of variation)")

print("\n--- Analysis Complete ---")
