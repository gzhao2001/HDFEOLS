# HDFE & HDFE-IV: High-Dimensional Fixed Effects Estimators

## Overview

This notebook implements two powerful econometric estimators for high-dimensional fixed effects models:

- **HDFE**: High-Dimensional Fixed Effects estimator using alternating projection algorithm
- **HDFE-IV**: High-Dimensional Fixed Effects Instrumental Variables estimator using 2SLS with fixed effects

Both estimators are designed to handle:
-  **Large datasets** (tested up to 20M+ observations)
-  **Multiple fixed effects** with thousands of categories each
-  **GPU acceleration** for faster computation
-  **Robust standard errors** (homoscedastic, heteroscedastic, cluster-robust)
-  **Memory efficient** sparse operations

## Key Features

### HDFE Estimator
- **Alternating Projection**: Efficient demeaning algorithm with Gearhart-Koshy acceleration
- **Sparse Fixed Effects Recovery**: Recovers individual fixed effect coefficients using sparse solvers
- **GPU Support**: Automatic GPU acceleration when available (CuPy)
- **Robust Standard Errors**: Support for various covariance matrix estimators

### HDFE-IV Estimator  
- **Two-Stage Least Squares**: Full 2SLS implementation with fixed effects demeaning
- **Multiple Endogenous Variables**: Unlike pyfixest, supports multiple endogenous variables
- **IV Diagnostics**: First-stage F-statistics, Sargan overidentification test, weak instrument detection
- **Correct IV Standard Errors**: Proper variance calculation for IV estimates

## Installation & Requirements

```python
# Required packages
import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import time
import psutil

# Optional for GPU acceleration
import cupy as cp  # pip install cupy-cuda11x or cupy-cuda12x
```

## API Reference

### HDFE Class

#### Constructor
```python
HDFE(max_iter=5000, tolerance=1e-8, acceleration='gk', use_gpu=None, verbose=False)
```

**Parameters:**
- `max_iter` (int): Maximum iterations for alternating projection algorithm
- `tolerance` (float): Convergence tolerance for demeaning algorithm  
- `acceleration` (str): Acceleration method ('gk' for Gearhart-Koshy)
- `use_gpu` (bool): Force GPU usage (None for auto-detect)
- `verbose` (bool): Print detailed progress information

#### Methods

##### `.fit(data, y_col, X_cols, fe_vars, se_type='homoscedastic', cluster_vars=None, sample_weight=None)`

Fit the HDFE model.

**Parameters:**
- `data` (DataFrame): Input dataset
- `y_col` (str): Dependent variable column name
- `X_cols` (list): List of continuous variable column names
- `fe_vars` (list): List of fixed effect variable column names  
- `se_type` (str): Standard error type ('homoscedastic', 'hc1', 'hc2', 'hc3', 'cluster')
- `cluster_vars` (list): Variables to cluster on (required for 'cluster' SE)
- `sample_weight` (array): Sample weights (optional)

**Returns:** Self (fitted estimator)

##### `.summary()`
Print comprehensive model summary including coefficients, standard errors, and fixed effects statistics.

**Attributes after fitting:**
- `coefficients_`: Estimated coefficients for continuous variables
- `std_errors_`: Standard errors for coefficients
- `t_stats_`: T-statistics  
- `p_values_`: P-values
- `r_squared_`: R-squared statistic
- `fe_coefficients_`: Dictionary of recovered fixed effect coefficients
- `residuals_`: Model residuals
- `fitted_values_`: Fitted values

### HDFE-IV Class

#### Constructor  
```python
HDFEIV(max_iter=5000, tolerance=1e-8, acceleration='gk', use_gpu=None, verbose=False)
```

Inherits all HDFE constructor parameters.

#### Methods

##### `.fit(data, y_col, X_cols, fe_vars, instruments, endogenous_vars, se_type='homoscedastic', cluster_vars=None, sample_weight=None)`

Fit the HDFE-IV model using Two-Stage Least Squares.

**Parameters:** 
- All HDFE parameters plus:
- `_instruments` (list): List of instrument variable column names
- `_endogenous_vars` (list): List of endogenous variable column names (subset of X_cols)

**Returns:** Self (fitted estimator)

##### `.first_stage_results()`
Returns detailed first-stage regression results for each endogenous variable.

##### `.iv_diagnostics()`  
Returns IV diagnostic tests including weak instrument tests and Sargan overidentification test.

**Additional attributes:**
- `_first_stage_r2`: First-stage R-squared for each endogenous variable
- `_first_stage_f_stats`: First-stage F-statistics  
- `_weak_instruments`: Boolean indicating weak instruments
- `_sargan_stat`: Sargan test statistic
- `_sargan_pvalue`: Sargan test p-value


## Quick Start Examples

### Example 1: Basic HDFE Usage

```python
# Example 1: Basic HDFE Usage
# Generate sample data
np.random.seed(42)
n_obs = 100_000
n_firms = 1000
n_workers = 500

# Generate firm and worker IDs  
firm_ids = np.random.randint(0, n_firms, n_obs)
worker_ids = np.random.randint(0, n_workers, n_obs)

# Generate firm and worker fixed effects
firm_effects = np.random.normal(0, 1, n_firms)
worker_effects = np.random.normal(0, 1, n_workers)

# Generate continuous variables
X1 = np.random.normal(2, 1, n_obs)  # Experience
X2 = np.random.normal(0, 1, n_obs)  # Education

# True coefficients
beta_X1 = 0.05  # Return to experience
beta_X2 = 0.10  # Return to education

# Generate dependent variable (log wages)
log_wage = (beta_X1 * X1 + beta_X2 * X2 + 
           firm_effects[firm_ids] + worker_effects[worker_ids] + 
           np.random.normal(0, 0.1, n_obs))

# Create DataFrame
data_example = pd.DataFrame({
    'log_wage': log_wage,
    'experience': X1,
    'education': X2, 
    'firm_id': firm_ids,
    'worker_id': worker_ids
})

print("📊 Sample Data Generated:")
print(f"Observations: {n_obs:,}")
print(f"Firms: {n_firms:,}")  
print(f"Workers: {n_workers:,}")
print(f"True β_experience: {beta_X1:.3f}")
print(f"True β_education: {beta_X2:.3f}")

# Fit HDFE model
hdfe_example = HDFE(verbose=True, use_gpu=False)  # Use CPU for small example
hdfe_example.fit(
    data=data_example,
    y_col='log_wage',
    X_cols=['experience', 'education'],
    fe_vars=['firm_id', 'worker_id'],
    se_type='homoscedastic'
)

# Display results
print("\n" + "="*60)
print("HDFE ESTIMATION RESULTS")
print("="*60)
hdfe_example.summary()
```

### Example 2: HDFE-IV with Endogeneity

```python
# Example 2: HDFE-IV with Endogeneity
# Generate data with endogenous variable (e.g., training participation)

np.random.seed(123)
n_obs = 500_000
n_firms = 800  
n_workers = 400

# Generate IDs and fixed effects
firm_ids = np.random.randint(0, n_firms, n_obs)
worker_ids = np.random.randint(0, n_workers, n_obs)
firm_effects = np.random.normal(0, 0.8, n_firms)
worker_effects = np.random.normal(0, 0.8, n_workers)

# Generate error term
error_term = np.random.normal(0, 0.2, n_obs)

# Generate instruments (policy variables)
Z1 = np.random.normal(0, 1, n_obs)  # Policy instrument 1
Z2 = np.random.normal(1, 1, n_obs)  # Policy instrument 2

# Generate exogenous variables
X1_exog = np.random.normal(2, 1, n_obs)  # Experience (exogenous)

# Generate endogenous variable (training participation)
# Correlated with error term (unobserved ability affects both training and wages)
X2_endog = (0.7 * Z1 + 0.5 * Z2 +           # Instrument relevance
            0.8 * error_term +               # Endogeneity correlation
            0.2 * firm_effects[firm_ids] +   # Firm effect on training
            np.random.normal(0, 0.5, n_obs)) # Random component

# True coefficients
beta_experience = 0.08
beta_training = 0.15    # True training effect (will be biased in OLS)

# Generate dependent variable (log wages)
log_wage = (beta_experience * X1_exog + 
            beta_training * X2_endog +
            firm_effects[firm_ids] + worker_effects[worker_ids] + 
            error_term)

# Create DataFrame
data_iv_example = pd.DataFrame({
    'log_wage': log_wage,
    'experience': X1_exog,
    'training': X2_endog,
    'policy_Z1': Z1,
    'policy_Z2': Z2, 
    'firm_id': firm_ids,
    'worker_id': worker_ids
})

print("📊 IV Example Data Generated:")
print(f"Observations: {n_obs:,}")
print(f"Endogeneity correlation: {np.corrcoef(X2_endog, error_term)[0,1]:.3f}")
print(f"True β_experience: {beta_experience:.3f}")
print(f"True β_training: {beta_training:.3f}")

print("\n🔧 Comparing HDFE (biased) vs HDFE-IV (corrected)...")

# 1. Fit HDFE (will show endogeneity bias)
hdfe_biased = HDFE(verbose=False, use_gpu=False)
hdfe_biased.fit(
    data=data_iv_example,
    y_col='log_wage', 
    X_cols=['experience', 'training'],
    fe_vars=['firm_id', 'worker_id'],
    se_type='homoscedastic'
)

# 2. Fit HDFE-IV (should correct the bias)
hdfeiv_corrected = HDFEIV(verbose=False, use_gpu=False)
hdfeiv_corrected.fit(
    data=data_iv_example,
    y_col='log_wage',
    X_cols=['experience', 'training'], 
    fe_vars=['firm_id', 'worker_id'],
    instruments=['policy_Z1', 'policy_Z2'],
    endogenous_vars=['training'],
    se_type='homoscedastic'
)

# Compare results
print("\n" + "="*80)
print("COMPARISON: HDFE (Biased) vs HDFE-IV (Corrected)")
print("="*80)
print(f"{'Variable':<12} {'True':<8} {'HDFE':<12} {'HDFE-IV':<12} {'Bias_HDFE':<12} {'Bias_IV':<12}")
print("-"*80)

for i, var in enumerate(['experience', 'training']):
    true_val = beta_experience if var == 'experience' else beta_training
    hdfe_coef = hdfe_biased.coefficients_[i]
    hdfeiv_coef = hdfeiv_corrected.coefficients_[i]
    
    bias_hdfe = ((hdfe_coef - true_val) / true_val) * 100
    bias_iv = ((hdfeiv_coef - true_val) / true_val) * 100
    
    print(f"{var:<12} {true_val:<8.3f} {hdfe_coef:<12.3f} {hdfeiv_coef:<12.3f} {bias_hdfe:<12.1f}% {bias_iv:<12.1f}%")

print("\n💡 Key Insight: HDFE-IV corrects the endogeneity bias in the training coefficient!")
print("="*80)
```