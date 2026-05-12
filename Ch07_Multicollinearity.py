"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 07 - Multicollinearity
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 7 - Multicollinearity
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Investment Demand Model
           I = B0 + B1*GDP + B2*Credit + B3*Rate + B4*CPI + u
           GDP and Credit are by construction highly correlated
           (banks lend more as the economy grows), demonstrating
           perfect vs. near multicollinearity as Asteriou does
           in Ch.7 with the near-collinear regressor scenario.
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Simulate Data with Near-Multicollinearity
#   GDP and Credit are correlated at ~0.95
#   This replicates Asteriou's near-collinear design
# ===================================================
np.random.seed(33)
n   = 40

gdp    = np.linspace(1000, 5000, n) + np.random.normal(0, 50, n)  # PKR Bn
credit = 0.75 * gdp + np.random.normal(0, 80, n)                   # Bank credit, PKR Bn
rate   = 8 + np.random.normal(0, 1.2, n)                           # Lending rate %
cpi    = 100 + np.cumsum(np.random.normal(0.5, 0.3, n))            # Price index

investment = (
    -200
    + 0.15 * gdp
    + 0.06 * credit
    - 12.0 * rate
    + 0.80 * cpi
    + np.random.normal(0, 30, n)
)

df = pd.DataFrame({
    'Investment': investment,
    'GDP'       : gdp,
    'Credit'    : credit,
    'Rate'      : rate,
    'CPI'       : cpi
})

print("=" * 60)
print("  ASTERIOU CH.7 | Multicollinearity")
print("  I = B0 + B1*GDP + B2*Credit + B3*Rate + B4*CPI + u")
print("  (GDP and Credit are near-collinear by construction)")
print("=" * 60)

# ===================================================
# 📌 Step 2: Correlation Matrix
#   Asteriou's first diagnostic: check pairwise correlations
#   Rule: |r| > 0.80 raises multicollinearity concern
# ===================================================
corr = df.corr()
print("\n--- Pairwise Correlation Matrix ---")
print(corr.round(3))
print("\n  Note: GDP-Credit correlation =", round(corr.loc['GDP', 'Credit'], 3))
print("  Asteriou threshold: |r| > 0.80 is concerning.")

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", square=True)
plt.title("Correlation Matrix: Investment Demand Model", fontsize=13)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 3: OLS with All Regressors (Full Model)
# ===================================================
X_full = sm.add_constant(df[['GDP', 'Credit', 'Rate', 'CPI']])
y      = df['Investment']
model_full = sm.OLS(y, X_full).fit()

print("\n--- OLS Full Model (All 4 Regressors) ---")
print(model_full.summary())
print("\n  Notice: GDP and Credit may both be insignificant")
print("  despite the model having a high R-squared.")
print("  This is the hallmark of multicollinearity.")

# ===================================================
# 📌 Step 4: VIF Analysis (Asteriou's Primary Diagnostic)
#   Rule: VIF > 5 is moderate concern
#         VIF > 10 is severe multicollinearity
# ===================================================
vif_df = pd.DataFrame({
    'Variable': X_full.columns,
    'VIF'     : [variance_inflation_factor(X_full.values, i)
                 for i in range(X_full.shape[1])]
})
print("\n--- Variance Inflation Factor (VIF) ---")
print(vif_df.round(3))
print("\n  Rule of thumb from Asteriou:")
print("  VIF > 5  : moderate multicollinearity")
print("  VIF > 10 : severe multicollinearity (problematic)")

# ===================================================
# 📌 Step 5: Condition Index (Asteriou Ch.7 Alternative)
#   Eigenvalue analysis of the X'X matrix
#   Condition number > 30 signals severe collinearity
# ===================================================
X_vals   = X_full.values
XtX      = X_vals.T @ X_vals
eigvals  = np.linalg.eigvalsh(XtX)
cond_idx = np.sqrt(eigvals.max() / eigvals)

print(f"\n--- Condition Index (Eigenvalue-based) ---")
print(f"  Eigenvalues: {np.round(eigvals, 2)}")
print(f"  Condition Indices: {np.round(cond_idx, 2)}")
print(f"  Condition Number: {np.round(cond_idx.max(), 2)}")
print(f"  Asteriou rule: Condition number > 30 = severe multicollinearity")
if cond_idx.max() > 30:
    print("  Decision: Severe multicollinearity detected.")

# ===================================================
# 📌 Step 6: Remedies
# ===================================================

# 6A: Drop one collinear variable (Credit)
X_reduced    = sm.add_constant(df[['GDP', 'Rate', 'CPI']])
model_reduced = sm.OLS(y, X_reduced).fit()

print("\n--- Reduced Model (Credit dropped) ---")
print(model_reduced.summary())

# 6B: Auxiliary regression to expose collinearity
#     Regress GDP on Credit and report R^2
X_aux   = sm.add_constant(df[['Credit']])
aux_mdl = sm.OLS(df['GDP'], X_aux).fit()
print(f"\n--- Auxiliary Regression: GDP on Credit ---")
print(f"  R-squared = {aux_mdl.rsquared:.4f}")
print(f"  VIF implied by auxiliary R^2 = {1/(1 - aux_mdl.rsquared):.3f}")
print("  Asteriou shows this is exactly VIF = 1/(1 - Rj^2)")

# 6C: Principal Component Regression (note only)
print("\n--- Ridge / PCA Regression: Conceptual Note ---")
print("  Asteriou Ch.7 mentions ridge regression and PCA as")
print("  advanced remedies. These are available via sklearn:")
print("  sklearn.linear_model.Ridge for ridge regression.")
print("  sklearn.decomposition.PCA for dimension reduction.")

# ===================================================
# 📌 Step 7: Coefficient Instability Demonstration
#   Add noise to show how coefficients flip sign
#   when multicollinearity is present
# ===================================================
print("\n--- Coefficient Stability Comparison ---")
comp = pd.DataFrame({
    'Full Model Coef': model_full.params[['GDP', 'Credit']].round(4),
    'Full Model SE'  : model_full.bse[['GDP', 'Credit']].round(4),
    'Full Model t'   : model_full.tvalues[['GDP', 'Credit']].round(3),
})
print(comp)
print("\n  Large SEs relative to coefficients = t-stats near zero.")
print("  Both variables appear insignificant yet together")
print("  explain most of the variation in investment.")
print("  This is the classic multicollinearity symptom.")

print("\n--- Analysis Complete ---")
