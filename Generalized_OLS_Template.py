"""
Project: Generalized OLS Regression Template
Author: Dr. Yasir Saeed
Affiliation: Kohat University of Science & Technology (KUST)
Description: A reusable framework for Simple Linear Regression, 
             Diagnostic Testing, and Visualization using generic variables x and y.
"""

# ===============================================
# 📌 Step 0: Import libraries
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Professional theme for academic publication
sns.set_theme(style="whitegrid", palette="magma")

# ===============================================
# 📌 Step 1: Data Input (Replace with your own data)
# ===============================================
# Independent Variable (Exogenous)
x_data = [133, 88, 74, 69, 43, 145, 130, 160, 215, 375, 309, 213, 158]

# Dependent Variable (Endogenous)
y_data = [5269, 3040, 2466, 2252, 1413, 5635, 1606, 1583, 8260, 17700, 14614, 7972, 5716]

# Create DataFrame
df = pd.DataFrame({'X': x_data, 'Y': y_data})

print("\n--- Descriptive Statistics ---")
print(df.describe())

# ===============================================
# 📌 Step 2: Visualization Functions
# ===============================================
def analyze_distribution(data, var_name, color):
    plt.figure(figsize=(8, 4))
    sns.histplot(data[var_name], kde=True, color=color, edgecolor='black')
    plt.title(f"Distribution of {var_name}", fontsize=14)
    plt.show()

analyze_distribution(df, 'X', 'teal')
analyze_distribution(df, 'Y', 'darkred')

# Scatter with Regression Line
plt.figure(figsize=(8, 5))
sns.regplot(x='X', y='Y', data=df, line_kws={"color": "black", "lw": 2})
plt.title("Scatter Plot with Fitted Regression Line", fontsize=14)
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (Y)")
plt.show()

# ===============================================
# 📌 Step 3: OLS Regression Analysis
# ===============================================
# Adding constant for the Intercept (Beta 0)
X = sm.add_constant(df['X'])
y = df['Y']

model = sm.OLS(y, X).fit()

print("\n--- OLS Regression Summary ---")
print(model.summary())

# ===============================================
# 📌 Step 4: Diagnostic Testing
# ===============================================

# 4A: Breusch-Pagan Test for Heteroscedasticity
bp_test = het_breuschpagan(model.resid, X)
bp_results = dict(zip(['LM Stat', 'LM p-value', 'F Stat', 'F p-value'], bp_test))

print("\n--- Heteroscedasticity Test (Breusch-Pagan) ---")
print(bp_results)

# Quick Logic for Scholars
if bp_results['LM p-value'] < 0.05:
    print("Decision: Reject H0 (Heteroscedasticity is present)")
else:
    print("Decision: Fail to Reject H0 (Homoscedasticity assumed)")

# 4B: Residual Diagnostics (QQ Plot)
sm.qqplot(model.resid, line='45', fit=True)
plt.title("Normal Q-Q Plot of Residuals")
plt.show()

# 4C: Variance Inflation Factor (VIF) 
# Useful if adding more variables later
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("\n--- Multicollinearity Check (VIF) ---")
print(vif_data)

print("\n--- Analysis Complete ---")
