"""
============================================================
Project  : Applied Econometrics in Python
Chapter  : 08 - Dummy Variables and Structural Breaks
Book Ref : Asteriou & Hall, Applied Econometrics (3rd Ed.)
           Chapter 8 - Dummy Variables
Author   : Dr. Yasir Saeed
           Department of Economics, KUST
Example  : Export Performance Model with Policy Dummies
           EXPORT = B0 + B1*GDP + B2*ER + B3*D_WTO
                       + B4*D_Crisis + B5*D_Covid + u
           Pakistan exports data with three structural dummies:
           WTO accession impact, 2008 global crisis, COVID-19.
           Asteriou uses seasonal dummies and a policy dummy
           in Ch.8; extended here to multiple economic events
           more relevant to Pakistan trade economics.
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
from scipy.stats import f as f_dist

sns.set_theme(style="whitegrid", palette="magma")

# ===================================================
# 📌 Step 1: Annual Pakistan Export Data (2000-2022)
#   GDP (PKR Bn), Exchange Rate (PKR/USD), Exports (USD Bn)
#   Structural dummies for key economic events
# ===================================================
years = list(range(2000, 2023))
n     = len(years)

gdp_growth = np.array([
    3.9, 2.0, 3.1, 4.7, 7.5, 9.0, 5.8, 6.8, 3.7, 0.4,
    2.6, 3.6, 4.4, 3.7, 4.1, 4.7, 5.5, 5.7, 3.3, 0.5,
    5.5, 6.0, 3.2
])

exchange_rate = np.array([
    52, 58, 60, 57, 58, 60, 63, 62, 70, 81,
    86, 88, 95, 99, 102, 104, 105, 110, 139, 163,
    177, 202, 240
], dtype=float)

# WTO-related benefits dummy (2005 onwards, MFA phase-out)
d_wto = np.array([0]*5 + [1]*18, dtype=float)

# Global financial crisis 2008-2009
d_crisis = np.array([0]*8 + [1, 1] + [0]*13, dtype=float)

# COVID-19 shock 2020-2021
d_covid = np.array([0]*20 + [1, 1] + [0]*1, dtype=float)

np.random.seed(22)
exports = (
    5.0
    + 0.8  * gdp_growth
    - 0.02 * exchange_rate
    + 1.8  * d_wto
    - 2.5  * d_crisis
    - 3.2  * d_covid
    + np.random.normal(0, 0.8, n)
)

df = pd.DataFrame({
    'Year'        : years,
    'Exports'     : exports,
    'GDP_growth'  : gdp_growth,
    'Exchange'    : exchange_rate,
    'D_WTO'       : d_wto,
    'D_Crisis'    : d_crisis,
    'D_Covid'     : d_covid
})
df.set_index('Year', inplace=True)

print("=" * 62)
print("  ASTERIOU CH.8 | Dummy Variables and Structural Breaks")
print("  Pakistan Export Model (2000-2022)")
print("=" * 62)

# ===================================================
# 📌 Step 2: Plot Exports with Dummy Events Marked
# ===================================================
plt.figure(figsize=(11, 5))
plt.plot(df.index, df['Exports'], color='navy', lw=2, marker='o', markersize=4, label='Exports (USD Bn)')
plt.axvspan(2005, 2006, alpha=0.15, color='green', label='WTO (from 2005)')
plt.axvspan(2008, 2010, alpha=0.20, color='orange', label='Global Crisis 2008-09')
plt.axvspan(2020, 2022, alpha=0.20, color='red', label='COVID-19 2020-21')
plt.title("Pakistan Exports with Key Structural Events", fontsize=13)
plt.xlabel("Year")
plt.ylabel("Exports (USD Bn)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

# ===================================================
# 📌 Step 3: OLS without Dummies (Baseline)
# ===================================================
X_base  = sm.add_constant(df[['GDP_growth', 'Exchange']])
model_base = sm.OLS(df['Exports'], X_base).fit()

print("\n--- Baseline OLS (No Dummy Variables) ---")
print(model_base.summary())

# ===================================================
# 📌 Step 4: OLS with Additive Dummies (Level Shifts)
#   Asteriou Ch.8: intercept dummies shift the constant
#   for the period when the dummy equals 1
# ===================================================
X_dummy = sm.add_constant(df[['GDP_growth', 'Exchange', 'D_WTO', 'D_Crisis', 'D_Covid']])
model_dummy = sm.OLS(df['Exports'], X_dummy).fit()

print("\n--- OLS with Additive Dummies (Asteriou Level Shift) ---")
print(model_dummy.summary())

print("\n--- Dummy Coefficient Interpretation ---")
coefs = model_dummy.params
print(f"  D_WTO   : {coefs['D_WTO']:.4f}")
print(f"    WTO-related trade liberalisation raised exports by")
print(f"    {coefs['D_WTO']:.2f} USD Bn on average (ceteris paribus).")
print(f"  D_Crisis: {coefs['D_Crisis']:.4f}")
print(f"    The 2008-09 global financial crisis reduced exports by")
print(f"    {abs(coefs['D_Crisis']):.2f} USD Bn.")
print(f"  D_Covid : {coefs['D_Covid']:.4f}")
print(f"    COVID-19 reduced exports by {abs(coefs['D_Covid']):.2f} USD Bn.")

# ===================================================
# 📌 Step 5: Slope Dummy (Interactive Term)
#   Asteriou Ch.8: test whether the GDP-export relationship
#   changed after the WTO accession (slope change)
# ===================================================
df['GDP_x_WTO']  = df['GDP_growth'] * df['D_WTO']
df['ER_x_Crisis'] = df['Exchange']  * df['D_Crisis']

X_interact = sm.add_constant(
    df[['GDP_growth', 'Exchange', 'D_WTO', 'GDP_x_WTO', 'D_Crisis', 'D_Covid']]
)
model_interact = sm.OLS(df['Exports'], X_interact).fit()

print("\n--- OLS with Interactive Dummy (Slope + Intercept Shift) ---")
print(model_interact.summary())

coef_inter = model_interact.params
print(f"\n  GDP_x_WTO (slope change): {coef_inter['GDP_x_WTO']:.4f}")
print(f"  Post-WTO, each 1% GDP growth yields an additional")
print(f"  {coef_inter['GDP_x_WTO']:.4f} USD Bn in exports beyond the pre-WTO effect.")

# ===================================================
# 📌 Step 6: Chow Test for Structural Break
#   Asteriou Ch.8: formal test for parameter stability
#   Split sample at 2008 crisis — does structure change?
# ===================================================
print("\n--- Chow Test for Structural Break at 2008 ---")

df1 = df.loc[2000:2007]   # Pre-crisis
df2 = df.loc[2008:2022]   # Post-crisis
df_all = df.loc[2000:2022]

X_simple = ['GDP_growth', 'Exchange']

def ols_rss(data):
    X = sm.add_constant(data[X_simple])
    y = data['Exports']
    return sm.OLS(y, X).fit().ssr

rss_full   = ols_rss(df_all)
rss_split  = ols_rss(df1) + ols_rss(df2)

k   = 3   # Number of parameters (const + 2 regressors)
n1  = len(df1)
n2  = len(df2)

F_chow = ((rss_full - rss_split) / k) / (rss_split / (n1 + n2 - 2*k))
p_chow = 1 - f_dist.cdf(F_chow, k, n1 + n2 - 2*k)

print(f"  RSS (Full sample)    : {rss_full:.4f}")
print(f"  RSS (Split, pre+post): {rss_split:.4f}")
print(f"  Chow F-statistic     : {F_chow:.4f}")
print(f"  p-value              : {p_chow:.4f}")
if p_chow < 0.05:
    print("  Decision: Reject H0 — Structural break at 2008 detected.")
else:
    print("  Decision: Fail to Reject H0 — No structural break detected.")

# ===================================================
# 📌 Step 7: Seasonal Dummies (Asteriou's Core Ch.8 Use Case)
#   Quarterly data with Q1-Q3 dummies (Q4 is base)
# ===================================================
print("\n--- Seasonal Dummy Illustration (Quarterly) ---")

np.random.seed(11)
T_q  = 60
gdp_q = np.cumsum(np.random.normal(2, 1, T_q)) + 100
season = np.tile([1, 0, 0, 0], 15)   # Q1 dummy (seasonal peak in Pakistan exports)
q2     = np.tile([0, 1, 0, 0], 15)
q3     = np.tile([0, 0, 1, 0], 15)
noise_q = np.random.normal(0, 1, T_q)

y_q = 5 + 0.8*gdp_q + 2.5*season - 0.5*q2 + 0.3*q3 + noise_q

X_seasonal = sm.add_constant(
    pd.DataFrame({'GDP': gdp_q, 'Q1': season, 'Q2': q2, 'Q3': q3})
)
model_seasonal = sm.OLS(y_q, X_seasonal).fit()
print(model_seasonal.summary())
print("  Q4 is the omitted (base) quarter.")
print("  Q1 coefficient shows seasonal premium over Q4.")

print("\n--- Analysis Complete ---")
