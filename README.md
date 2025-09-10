# High-Dimensional Fixed Effects (HDFE) Estimators - User Guide

## Overview

This notebook provides two implementations of high-dimensional fixed effects regression estimators:

1. **`HDFE`** - CPU-based implementation using alternating projection
2. **`HDFEGpu`** - GPU-accelerated version for enhanced performance

Both estimators handle challenging econometric problems with:
- Multiple fixed effects with hundreds/thousands of categories each
- Large datasets (tested up to 2M+ observations) 
- Dozens of continuous variables
- Memory-efficient sparse matrix operations

## Quick Start

### Basic Usage - CPU Implementation

```python
# Import necessary libraries
import pandas as pd
import numpy as np

# Initialize the estimator
model = HDFE(max_iter=50, tol=1e-8)

# Fit the model
model.fit(
    data=your_dataframe,           # pandas DataFrame
    y_col='dependent_variable',     # name of dependent variable
    X_cols=['x1', 'x2', 'x3'],     # list of continuous variable names
    fe_vars=['fe1', 'fe2']    # list of fixed effect variables
)

# Get results
print(f"R-squared: {model.r_squared_:.4f}")
print("Coefficients:", model.coefficients_)
print("Fixed Effects:", model.fe_coefficients_)

# Make predictions
predictions = model.predict(new_data)           # without fixed effects
predictions_full = model.predict_full(new_data) # with fixed effects
```

### GPU-Accelerated Usage

```python
# Initialize GPU-accelerated estimator
gpu_model = HDFEGpu(
    max_iter=50, 
    tol=1e-8, 
    use_gpu=True  # will automatically fall back to CPU if GPU unavailable
)

# Same API as CPU version
gpu_model.fit(
    data=your_dataframe,
    y_col='dependent_variable',
    X_cols=['x1', 'x2', 'x3'],
    fe_vars=['fe1', 'fe2']
)

# Results available through same interface
print(f"GPU R-squared: {gpu_model.r_squared_:.4f}")
```

## Key Features

### 1. Scalable Fixed Effects Handling
- **Multiple Dimensions**: Handle 2, 3, or more fixed effect dimensions simultaneously
- **High Cardinality**: Each fixed effect can have hundreds or thousands of categories
- **Memory Efficient**: Uses sparse matrices and iterative algorithms to minimize memory usage

### 2. Robust Algorithm Implementation
- **Alternating Projection**: Based on Gaure (2013) algorithm for efficient fixed effects absorption
- **Automatic Convergence**: Monitors convergence and stops when tolerance is reached
- **Identification Strategy**: Automatically handles identification by dropping first category from last K-1 fixed effects

### 3. Performance Optimizations
- **CPU**: Numba-accelerated core functions for fast group operations
- **GPU**: CuPy-based GPU acceleration for alternating projection and sparse solving
- **Memory Management**: Efficient data structures to handle large datasets

## Method Details

### Alternating Projection Algorithm

The estimators implement the alternating projection method which:

1. **Initialization**: Start with original data (y, X)
2. **Projection Loop**: For each iteration:
   - Project out fixed effect 1 from (y, X)
   - Project out fixed effect 2 from projected (y, X)  
   - Continue for all fixed effects
   - Check convergence
3. **OLS**: Run OLS on final projected data
4. **Recovery**: Recover fixed effect coefficients using sparse solver

### GPU Acceleration Details

The GPU implementation accelerates:
- **Group Operations**: Fast computation of group means using `cp.bincount`
- **Matrix Operations**: GPU-accelerated matrix algebra
- **Sparse Solving**: GPU sparse linear system solving with CPU fallback
- **Memory Transfer**: Optimized CPU↔GPU data transfer

## API Reference

### Initialization Parameters

```python
HDFE(max_iter=100, tol=1e-8)
HDFEGpu(max_iter=100, tol=1e-8, use_gpu=True)
```

- `max_iter`: Maximum iterations for alternating projection (default: 100)
- `tol`: Convergence tolerance (default: 1e-8)
- `use_gpu`: Enable GPU acceleration (GPU class only, default: True)

### Main Methods

#### `fit(data, y_col, X_cols, fe_vars)`
Fits the model to data.

**Parameters:**
- `data`: pandas DataFrame containing all variables
- `y_col`: string, name of dependent variable column
- `X_cols`: list of strings, names of continuous variable columns  
- `fe_vars`: list of strings, names of fixed effect variable columns

**Returns:** self (fitted model)

#### `predict(data)`
Predicts using fitted model (continuous variables only).

**Parameters:**
- `data`: pandas DataFrame with same X_cols as training data

**Returns:** numpy array of predictions

#### `predict_full(data)`  
Predicts using fitted model including fixed effects.

**Parameters:**
- `data`: pandas DataFrame with same X_cols and fe_vars as training data

**Returns:** numpy array of predictions including fixed effects

#### `summary()`
Prints comprehensive model summary including:
- R-squared and model diagnostics
- Coefficient estimates with standard errors, t-stats, p-values
- Fixed effects summaries (mean, std, min, max by dimension)

### Model Attributes (Available After Fitting)

- `coefficients_`: Estimated coefficients for continuous variables
- `fe_coefficients_`: Dictionary of fixed effect coefficients by dimension
- `std_errors_`: Standard errors for coefficients
- `t_stats_`: t-statistics for coefficients  
- `p_values_`: p-values for coefficients
- `r_squared_`: R-squared value
- `residuals_`: Model residuals
- `fitted_values_`: Fitted values


## Some Notes

1. **Data Preparation**: 
   - Ensure fixed effect variables are categorical/integer
   - Remove missing values beforehand when possible
   - Consider data types (float64 for precision vs float32 for memory)

2. **GPU Usage**:
   - GPU acceleration most beneficial for large datasets (500K+ observations)
   - Ensure sufficient GPU memory
   - GPU speedup increases with number of fixed effect categories

3. **Memory Optimization**:
   - For very large datasets, consider processing in chunks
   - Monitor memory usage during model fitting
   - Use `del` and `gc.collect()` to free memory between operations

