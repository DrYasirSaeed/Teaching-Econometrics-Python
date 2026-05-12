"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 03 - Multiple Linear Regression (MLR)
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 3 - The Multiple Regression Model
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Extended Consumption Function
           C = B0 + B1*Income + B2*Wealth + B3*Interest + u
           Pakistan-context macro data (annual, 2000-2022)
           Book uses UK example; extended here with three
           regressors to demonstrate MLR mechanics and
           coefficient interpretation in Pakistan context.
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
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Data Construction
#   Three regressors mirroring Asteriou Ch.3 extension:
#   Income (Y_d), Household Wealth proxy, Interest Rate
#   Pakistan macro analogs (illustrative scaled values)
# ===================================================

np.random.seed(42)
n = 23

income   = np.array([
    120, 135, 148, 162, 178, 195, 212, 229, 248, 268,
    285, 302, 318, 338, 359, 378, 398, 415, 432, 450, 469, 485, 500
], dtype=float)

# Wealth proxy: cumulative savings index
wealth   = income * 2.1 + np.random.normal(0, 15, n)

# SBP policy rate (approximate, %)
interest = np.array([
    12.0, 10.0, 9.5, 8.5, 9.0, 10.5, 11.5, 13.0, 14.0, 13.5,
    12.5, 11.0, 10.0, 9.5, 9.0, 8.0, 7.5, 10.0, 13.75, 15.0,
    19.0, 21.0, 22.0
])

# Consumption: positively driven by income and wealth,
# negatively by interest rate (standard Keynesian model)
consumption = (
    20
    + 0.72 * income
    + 0.08 * wealth
    - 1.20 * interest
    + np.random.normal(0, 5, n)
)

years = list(range(2000, 2023))

df = pd.DataFrame({
    'Year'       : years,
    'Consumption': consumption,
    'Income'     : income,
    'Wealth'     : wealth,
    'Interest'   : interest
})

print("=" * 60)
print("  ASTERIOU CH.3 | Multiple Linear Regression")
print("  C = B0 + B1*Income + B2*Wealth + B3*Interest + u")
print("=" * 60)

# ===================================================
# 📌 Step 2: Descriptive Statistics
# ===================================================
print("\n--- Descriptive Statistics ---")
print(df.drop('Year', axis=1).describe().round(2))

# ===================================================
# 📌 Step 3: Correlation Matrix (Asteriou Ch.3 Prerequisite)
# ===================================================
print("\n--- Correlation Matrix ---")
corr = df[['Consumption', 'Income', 'Wealth', 'Interest']].corr()
print(corr.round(3))

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix: Consumption Model Variables", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 4: OLS Multiple Regression
# ===================================================
X = sm.add_constant(df[['Income', 'Wealth', 'Interest']])
y = df['Consumption']

model = sm.OLS(y, X).fit()

print("\n--- OLS Multiple Regression Summary ---")
print(model.summary())

# ===================================================
# 📌 Step 5: Coefficient Interpretation (Asteriou Style)
# ===================================================
coefs = model.params
print("\n--- Coefficient Interpretation ---")
print(f"  Intercept  : {coefs['const']:.3f}")
print(f"  Income (B1): {coefs['Income']:.3f}")
print(f"    Holding Wealth and Interest constant,")
print(f"    a 1 PKR thousand increase in income raises")
print(f"    consumption by {coefs['Income']:.3f} PKR thousand (ceteris paribus).")
print(f"  Wealth (B2): {coefs['Wealth']:.3f}")
print(f"    Wealth effect on consumption, holding others fixed.")
print(f"  Interest (B3): {coefs['Interest']:.3f}")
print(f"    A 1 percentage point rise in SBP rate reduces")
print(f"    consumption by {abs(coefs['Interest']):.3f} PKR thousand.")

# ===================================================
# 📌 Step 6: Partial Regression Plots
#   Asteriou recommends these for visualising individual
#   regressor effects after partialling out other Xs
# ===================================================
fig = plt.figure(figsize=(14, 4))
sm.graphics.plot_partregress_grid(model, fig=fig)
plt.suptitle("Partial Regression Plots: Consumption Model", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 7: Diagnostic Tests
# ===================================================

# 7A: Breusch-Pagan Heteroscedasticity Test
bp_test    = het_breuschpagan(model.resid, X)
bp_labels  = ['LM Stat', 'LM p-value', 'F Stat', 'F p-value']
bp_results = dict(zip(bp_labels, bp_test))

print("\n--- Breusch-Pagan Test (Heteroscedasticity) ---")
for k, v in bp_results.items():
    print(f"  {k}: {v:.4f}")
if bp_results['LM p-value'] < 0.05:
    print("  Decision: Reject H0 — Heteroscedasticity detected")
else:
    print("  Decision: Fail to Reject H0 — Homoscedasticity assumed")

# 7B: VIF Check (Multicollinearity)
vif_df = pd.DataFrame({
    'Variable': X.columns,
    'VIF'     : [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("\n--- Variance Inflation Factor (VIF) ---")
print(vif_df.round(3))
print("  Rule: VIF > 10 signals severe multicollinearity")
print("  Note: Income and Wealth are likely correlated by construction.")

# 7C: QQ Plot of Residuals
sm.qqplot(model.resid, line='45', fit=True)
plt.title("Normal Q-Q Plot of Residuals (MLR)", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 8: F-Test for Overall Significance
#   Asteriou Ch.3 Section on Joint Hypothesis Testing
#   H0: B1 = B2 = B3 = 0
# ===================================================
print(f"\n--- F-Test for Overall Model Significance ---")
print(f"  F-statistic : {model.fvalue:.4f}")
print(f"  Prob(F-stat): {model.f_pvalue:.6f}")
if model.f_pvalue < 0.05:
    print("  Decision: Reject H0 — At least one regressor is significant.")
else:
    print("  Decision: Fail to reject H0 — Model has no joint explanatory power.")

print("\n--- Analysis Complete ---")
