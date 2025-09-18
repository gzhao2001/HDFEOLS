# 📖 HDFE Estimator - User Guide

## Overview

The **HDFE** estimator is a high-performance implementation for estimating high-dimensional fixed effects models. It combines the precision of sparse matrix solvers with modern GPU acceleration to efficiently handle datasets with millions of observations and thousands of fixed effect categories.

## Key Features

### 🚀 **Performance**
- **GPU Acceleration**: Automatic GPU detection with CuPy acceleration for alternating projection and cluster-robust standard errors
- **Memory Efficient**: Sparse matrix operations handle large datasets without memory overflow
- **Fast Convergence**: Gearhart-Koshy acceleration method reduces iterations needed

### 🔧 **Robust Standard Errors**
- **Homoscedastic**: Standard OLS assumptions
- **Heteroscedastic-Consistent**: HC1, HC2, HC3 corrections
- **Cluster-Robust**: Single or multi-way clustering using Cameron-Gelbach-Miller method

---

## Installation Requirements

```python
# Required packages
import pandas as pd
import numpy as np
import scipy
import cupy  # Optional, for GPU acceleration
```

## HDFE Class

```python
from HDFE import HDFE
model = HDFE(use_gpu=True, max_iter=10000, tolerance=1e-8, 
    acceleration='gk', verbose=False)
```

**Parameters:**
- `use_gpu` (bool): Enable GPU acceleration using CuPy (default: True)
- `max_iter` (int): Maximum iterations for alternating projection (default: 10,000)
- `tolerance` (float): Convergence tolerance (default: 1e-8)  
- `acceleration` (str): Acceleration method - 'gk' (Gearhart-Koshy) or None (default: 'gk')
- `verbose` (bool): Print detailed progress information (default: False)

### fit() Method

```python
model.fit(data, y_col, X_cols, fe_vars, se_type='homoscedastic', cluster_vars=None)
```

**Parameters:**
- `data` (DataFrame): Input dataset
- `y_col` (str): Dependent variable column name
- `X_cols` (list): List of independent variable column names
- `fe_vars` (list): List of fixed effect variable column names
- `se_type` (str): Standard error type - 'homoscedastic', 'hc1', 'hc2', 'hc3', 'cluster'
- `cluster_vars` (list): Clustering variables for cluster-robust standard errors

### Model Attributes (After Fitting)

- `coefficients_`: Estimated coefficients (numpy array)
- `std_errors_`: Standard errors (numpy array)
- `r_squared_`: R-squared goodness of fit
- `fe_coefficients_`: Dictionary of recovered fixed effects 



## Basic Usage Examples

### 1. Simple Fixed Effects Model

```python
# 📊 DEMO: Creating a Test Dataset
print("🔧 GENERATING DEMONSTRATION DATASET")
print("=" * 50)

# Set random seed for reproducibility
np.random.seed(12345)

# Dataset parameters
n_obs = 50000           # Number of observations
n_firms = 800           # Number of firms (fe1)
n_years = 15            # Number of years (fe2)
n_industries = 25       # Number of industries (for clustering)

print(f"Dataset size: {n_obs:,} observations")
print(f"Fixed Effects: {n_firms:,} firms, {n_years} years")
print(f"Clusters: {n_industries} industries")

# Generate base data
demo_df = pd.DataFrame({
    'obs_id': range(n_obs),
    'firm_id': np.random.randint(1, n_firms + 1, n_obs),
    'year': np.random.randint(2008, 2008 + n_years, n_obs),
    'industry': np.random.randint(1, n_industries + 1, n_obs)
})

# Generate explanatory variables with realistic correlations
demo_df['experience'] = np.maximum(0, np.random.normal(8, 4, n_obs))  # Work experience
demo_df['education'] = np.maximum(8, np.random.normal(14, 3, n_obs))   # Years of education
demo_df['hours'] = np.maximum(20, np.random.normal(40, 8, n_obs))      # Hours worked

# Create true fixed effects with realistic variation
firm_effects = np.random.normal(0, 0.8, n_firms)    # Firm-specific productivity
year_effects = np.random.normal(0, 0.3, n_years)     # Year-specific trends

# Map effects to observations
demo_df['firm_effect'] = firm_effects[demo_df['firm_id'] - 1]
demo_df['year_effect'] = year_effects[demo_df['year'] - 2008]

# Generate log wage with realistic coefficients
true_beta = np.array([0.08, 0.12, 0.02])  # Returns to experience, education, hours

log_wage = (
    10.5 +                                      # Base wage
    true_beta[0] * demo_df['experience'] +      # Experience premium
    true_beta[1] * demo_df['education'] +       # Education premium  
    true_beta[2] * demo_df['hours'] +           # Hours effect
    demo_df['firm_effect'] +                    # Firm fixed effect
    demo_df['year_effect'] +                    # Year fixed effect
    np.random.normal(0, 0.4, n_obs)            # Error term
)

demo_df['log_wage'] = log_wage

# Remove helper columns for clean dataset
demo_clean = demo_df[['obs_id', 'firm_id', 'year', 'industry', 
                      'experience', 'education', 'hours', 'log_wage']].copy()

print(f"\n✅ Demo dataset created successfully!")
print(f"Shape: {demo_clean.shape}")
print(f"\nFirst 5 observations:")
print(demo_clean.head())

print(f"\n📊 Dataset Summary:")
print("-" * 30)
print(f"Mean log wage: {demo_clean['log_wage'].mean():.3f}")
print(f"Firms range: {demo_clean['firm_id'].min()} to {demo_clean['firm_id'].max()}")
print(f"Years range: {demo_clean['year'].min()} to {demo_clean['year'].max()}")
print(f"True coefficients: Experience={true_beta[0]:.3f}, Education={true_beta[1]:.3f}, Hours={true_beta[2]:.3f}")
```

```python
# Example 1: Basic Fixed Effects Model
print("🎯 EXAMPLE 1: BASIC FIXED EFFECTS MODEL")
print("=" * 55)
print("Model: log_wage ~ experience + education | firm_id + year")
print("Standard errors: Homoscedastic (default)")

# Create and fit basic model
model_basic = HDFE(verbose=True)

model_basic.fit(
    data=demo_clean,
    y_col='log_wage',
    X_cols=['experience', 'education'],
    fe_vars=['firm_id', 'year']
)

print("\n📊 RESULTS:")
print("-" * 25)
print(f"Coefficients:")
print(f"  Experience: {model_basic.coefficients_[0]:.6f} (SE: {model_basic.std_errors_[0]:.6f})")
print(f"  Education:  {model_basic.coefficients_[1]:.6f} (SE: {model_basic.std_errors_[1]:.6f})")
print(f"\nModel Statistics:")
print(f"  R-squared: {model_basic.r_squared_:.6f}")
print(f"  Observations: {len(demo_clean):,}")

# Check if we have fixed effects recovered
if hasattr(model_basic, 'recovered_fe_') and model_basic.recovered_fe_:
    fe_count = sum(len(v) for v in model_basic.recovered_fe_.values())
    print(f"  Fixed Effects: {len(model_basic.recovered_fe_)} categories, {fe_count:,} total values")
else:
    print(f"  Fixed Effects: 2 categories (firm_id, year)")

print(f"\n✅ Coefficient Recovery:")
true_coeffs_basic = [0.080, 0.120]  # True experience and education effects
for i, (true, est) in enumerate(zip(true_coeffs_basic, model_basic.coefficients_)):
    var_name = ['Experience', 'Education'][i]
    error = abs(est - true)
    recovery_pct = (1 - error/abs(true)) * 100
    print(f"  {var_name}: True={true:.3f}, Estimated={est:.6f}, Recovery={recovery_pct:.1f}%")

print("\n" + "="*55)
```

### 2. Multi-Way Clustering and Heteroscedastic-Consistent Standard Errors

```python
# Example 3: Comparing Different Standard Error Types
print("🎯 EXAMPLE 3: STANDARD ERROR TYPE COMPARISON")
print("=" * 65)
print("Model: log_wage ~ experience + education | firm_id + year")

# Dictionary to store results for comparison
se_comparison = {}

se_types_demo = [
    ('Homoscedastic', 'homoscedastic', None),
    ('HC1', 'hc1', None),
    ('HC3', 'hc3', None),
    ('Single Cluster', 'cluster', ['industry']),
    ('Multi-way Cluster', 'cluster', ['industry', 'firm_id'])
]

print("\nFitting models with different standard error specifications...")

for se_name, se_type, cluster_vars in se_types_demo:
    print(f"\n🔧 Computing {se_name} standard errors...")
    
    model_se = HDFE(use_gpu=True, verbose=False)
    model_se.fit(
        data=demo_clean,
        y_col='log_wage',
        X_cols=['experience', 'education'],
        fe_vars=['firm_id', 'year'],
        se_type=se_type,
        cluster_vars=cluster_vars
    )
    
    se_comparison[se_name] = {
        'coeffs': model_se.coefficients_.copy(),
        'se': model_se.std_errors_.copy(),
        'r2': model_se.r_squared_
    }

# Display comparison table
print("\n📊 STANDARD ERROR COMPARISON TABLE:")
print("=" * 80)
print(f"{'SE Type':<18} {'Experience':<20} {'Education':<20} {'R²':<10}")
print(f"{'':18} {'Coef (SE)':<20} {'Coef (SE)':<20} {'':10}")
print("-" * 80)

for se_name in se_comparison:
    coef_exp = se_comparison[se_name]['coeffs'][0]
    se_exp = se_comparison[se_name]['se'][0]
    coef_edu = se_comparison[se_name]['coeffs'][1]
    se_edu = se_comparison[se_name]['se'][1]
    r2 = se_comparison[se_name]['r2']
    
    print(f"{se_name:<18} {coef_exp:.4f} ({se_exp:.5f})  {coef_edu:.4f} ({se_edu:.5f})  {r2:.6f}")

print("\n🔍 KEY OBSERVATIONS:")
print("-" * 40)
print("• Coefficients remain identical across SE types (as expected)")
print("• Cluster-robust SEs are typically larger than homoscedastic SEs")
print("• Multi-way clustering captures correlation in multiple dimensions") 
print("• GPU acceleration speeds up cluster SE computation significantly")

print("\n" + "="*65)
```


