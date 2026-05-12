# 🏛️ Teaching Econometrics in Python

**Practice Codes for Applied Econometrics**
Developed by **Dr. Yasir Saeed**
Department of Economics, Kohat University of Science and Technology (KUST)

---

This repository provides MPhil and PhD scholars with reproducible Python practice codes built around **Asteriou and Hall, *Applied Econometrics* (3rd Edition)**. Each script follows the book's chapter structure, uses the book's core examples where applicable, and replaces UK-specific datasets with Pakistan-relevant economic data (KSE-100, SBP policy rates, PSX firms, PKR-denominated series) for local applicability.

---

## 📂 Available Practice Codes

### 1. Generalized OLS Regression Template

A reusable framework for bivariate OLS. Covers descriptive statistics, scatter plots with regression lines, and diagnostic tests including Breusch-Pagan and VIF.

**File:** `Generalized_OLS_Template.py`
**Objective:** Master the foundational OLS workflow before moving to book chapters.

---

### 2. Chapter 2: Simple Linear Regression

Keynesian consumption function: `C = B0 + B1*Y + u`. Demonstrates the closed-form OLS estimators (Beta1 derivation step by step), goodness of fit decomposition (TSS, ESS, RSS, R²), and visual diagnostics. Pakistan household income and consumption analog data (PKR thousands).

**File:** `Ch02_Simple_Linear_Regression.py`
**Book Ref:** Asteriou Ch.2, Table 2.1 analog

---

### 3. Chapter 3: Multiple Linear Regression

Extended consumption function with three regressors: Income, Wealth, and SBP Interest Rate. Covers coefficient interpretation (ceteris paribus), partial regression plots, F-test for joint significance, VIF, and Breusch-Pagan test. Pakistan macro annual data 2000-2022.

**File:** `Ch03_Multiple_Linear_Regression.py`
**Book Ref:** Asteriou Ch.3 extended model

---

### 4. Chapter 5: Heteroscedasticity

Investment expenditure across PSX-listed firms of varying size. Variance deliberately made proportional to firm size (funnel pattern). Covers Breusch-Pagan, White, and Goldfeld-Quandt tests, HC3 robust standard errors, and Weighted Least Squares correction.

**File:** `Ch05_Heteroscedasticity.py`
**Book Ref:** Asteriou Ch.5 cross-section example

---

### 5. Chapter 6: Autocorrelation (Serial Correlation)

Pakistan money demand model (M2, GDP, interest rate) with quarterly data and AR(1) error structure. Covers Durbin-Watson test, Breusch-Godfrey LM test, Ljung-Box test, correlogram (ACF/PACF), Cochrane-Orcutt GLS correction, and Newey-West HAC standard errors.

**File:** `Ch06_Autocorrelation.py`
**Book Ref:** Asteriou Ch.6 money demand example

---

### 6. Chapter 7: Multicollinearity

Investment demand model with GDP and Credit as near-collinear regressors (correlation ~0.95). Covers pairwise correlation matrix, VIF analysis, condition index, auxiliary regression method, coefficient instability demonstration, and variable deletion remedy.

**File:** `Ch07_Multicollinearity.py`
**Book Ref:** Asteriou Ch.7 near-collinearity scenario

---

### 7. Chapter 8: Dummy Variables and Structural Breaks

Pakistan export model (2000-2022) with additive dummies for WTO liberalisation, 2008 global crisis, and COVID-19 shock. Covers slope dummies (interactive terms), seasonal dummies (quarterly), and the Chow test for structural break at 2008.

**File:** `Ch08_Dummy_Variables.py`
**Book Ref:** Asteriou Ch.8 policy dummy and seasonal sections

---

### 8. Chapter 9: Unit Root Tests (Stationarity)

KSE-100 Index and PKR/USD exchange rate tested for unit roots. Covers ADF test (three specifications: no constant, constant only, constant plus trend), KPSS test, ACF/PACF correlograms, and integration order summary table.

**File:** `Ch09_Unit_Root_Tests.py`
**Book Ref:** Asteriou Ch.9 ADF and KPSS sections

---

### 9. Chapter 10: Cointegration and Error Correction Model

Monetary approach to exchange rate using Log(M2) and Log(Exchange Rate). Covers Engle-Granger two-step method (residual-based test), Johansen trace and max-eigenvalue statistics, Error Correction Model (ECM) with speed-of-adjustment interpretation, and ECT time series plot.

**File:** `Ch10_Cointegration_ECM.py`
**Book Ref:** Asteriou Ch.10 Engle-Granger and Johansen sections

---

### 10. Chapter 11: Vector Autoregressive Models (VAR)

Trivariate VAR for Pakistan macro variables: industrial production growth, M2 growth, and interest rate change. Covers lag order selection (AIC/BIC/HQ), Granger causality tests, Impulse Response Functions (IRF), orthogonalised IRF via Cholesky decomposition, Forecast Error Variance Decomposition (FEVD), and 8-quarter ahead forecasting.

**File:** `Ch11_VAR_Models.py`
**Book Ref:** Asteriou Ch.11 VAR and Granger causality sections

---

## 🛠️ Requirements

```
Python 3.8+
pandas
numpy
matplotlib
seaborn
statsmodels
scipy
```

Install all dependencies in one go:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy
```

---

## 📌 Coding Conventions

All scripts follow a consistent format:

- Project header block with chapter reference and data description
- Numbered steps with `# ===` section dividers
- 📌 emoji markers for major steps
- Inline academic reasoning comments explaining the econometric logic
- Pakistan-context variable labels and PKR-denominated values wherever applicable

---

## 📬 Contact

Dr. Yasir Saeed
Department of Economics, KUST
[github.com/DrYasirSaeed](https://github.com/DrYasirSaeed)
