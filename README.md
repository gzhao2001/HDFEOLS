# HDFE & HDFE-IV: High-Dimensional Fixed Effects Estimators

## Overview

GPU-accelerated econometric estimators for high-dimensional fixed effects models:

- **HDFE** — OLS with multiple high-dimensional fixed effects, demeaned via alternating projections
- **HDFE-IV** — 2SLS instrumental variables extension of HDFE

Both classes live in [`HDFE.py`](HDFE.py). `HDFEIV` inherits from `HDFE` and falls back to OLS when no instruments are supplied.

### Algorithm

1. **Demeaning** — Alternating projection over FE groups with Gearhart-Koshy (GK) acceleration.  Per-FE group constants (valid masks, inverse counts) are **precomputed once** before the iteration loop.  Each iteration batch-demeans $y$ and all $X$ columns together per FE group (avoiding per-column loops).  Convergence is checked every `convergence_check_interval` iterations (default 5), reducing snapshot overhead.  GPU path uses CuPy `bincount`/`scatter_add`; CPU path uses NumPy `bincount`.
2. **Coefficient estimation** — OLS on demeaned data (`HDFE`) or direct 2SLS formula $\beta = (X'P_Z X)^{-1} X'P_Z y$ with original $X$ (`HDFEIV`).  If $X'X$ is singular after projection, falls back to `lstsq` with a warning.
3. **FE recovery** — Sparse `D'D` system solved via `spsolve` (GPU: `cupyx`).  **Empty FE categories** (zero-diagonal entries in $D'D$) are pruned before solving.  Since `spsolve` silently returns NaN on singular matrices, the solver detects NaN output and falls back to `scipy.sparse.linalg.lsqr` (minimum-norm least-squares solution).  Set `singular_fallback='raise'` to get an exception instead.  When sample weights are active, the RHS is un-weighted before solving so FE coefficients are on the original scale.
4. **Standard errors** — Sandwich estimator with the appropriate bread/meat for each SE type.  IV models use the IV-specific variance formula $A^{-1} B A^{-1}$ where $A = X'Z(Z'Z)^{-1}Z'X$.
5. **IV singularity guards** — Three guards protect the HDFE-IV pipeline:
   - *Guard 1 (first-stage restricted F-stat):* `matrix_rank` pre-check on $X_{exog}$ + `isfinite` post-check.
   - *Guard 2 (Z'Z inversion):* try/except + `isfinite` post-check on $(Z'Z)^{-1}$.
   - *Guard 3 (A inversion for SE):* try/except + `isfinite` post-check on $A^{-1}$.
   All guards log to `rank_diagnostics()` and either fall back gracefully or raise with a clear message.


## Requirements

```
numpy
pandas
scipy
cupy              # optional — GPU acceleration
```

Install CuPy for your CUDA version:
```bash
pip install cupy-cuda11x   # or cupy-cuda12x
```

## API Reference

### `HDFE`

```python
from HDFE import HDFE

model = HDFE(
    max_iter=5000,       # max alternating-projection iterations
    tolerance=1e-8,      # convergence criterion (mean squared change)
    acceleration='gk',   # 'gk' (Gearhart-Koshy) or 'basic'
    use_gpu=False,       # set True to use CuPy GPU acceleration
    verbose=False,
    convergence_check_interval=5,  # check convergence every N iterations
)
```

#### `.fit(data, y_col, X_cols, fe_vars, se_type='homoscedastic', cluster_vars=None, sample_weight=None, singular_fallback='lsqr')`

| Parameter | Type | Description |
|---|---|---|
| `data` | DataFrame | Input dataset |
| `y_col` | str | Dependent variable column |
| `X_cols` | list[str] | Continuous regressor columns |
| `fe_vars` | list[str] | Fixed-effect columns (must be non-empty) |
| `se_type` | str | `'homoscedastic'`, `'hc1'`, `'hc2'`, `'hc3'`, or `'cluster'` |
| `cluster_vars` | list[str] | Cluster variable(s); required when `se_type='cluster'` |
| `sample_weight` | array | Per-observation weights (WLS); applied as √w scaling |
| `singular_fallback` | str | `'lsqr'` (default) — use least-squares fallback on singular systems; `'raise'` — raise `LinAlgError` instead |

Returns `self`.

#### `.summary()`

Prints coefficients, SEs, t-stats, p-values, R², FE summary statistics, and any rank diagnostics warnings.

#### `.rank_diagnostics()`

Returns a `dict` of rank/singularity diagnostic information from the last `.fit()` call.  Empty dict when no issues were detected.  Possible keys:

| Key | Meaning |
|---|---|
| `XtX_singular` | $X'X$ was singular after demeaning; `lstsq` fallback used |
| `empty_fe_categories` | Dict of FE vars with empty categories that were pruned |
| `spsolve_nan` | `spsolve` returned NaN; `lsqr` fallback used |
| `solve_exception` | `spsolve` raised an exception; `lsqr` fallback used |
| `first_stage_restricted_singular` | Restricted first-stage regression singular (IV only) |
| `ZtZ_singular` | $Z'Z$ is singular — collinear instruments (IV only) |
| `2sls_A_singular` | 2SLS projection matrix singular (IV only) |
| `iv_se_A_singular` | IV SE matrix singular (IV only) |

#### Attributes (after `.fit()`)

| Attribute | Type | Description |
|---|---|---|
| `coefficients_` | ndarray | Estimated β for `X_cols` |
| `std_errors_` | ndarray | Standard errors |
| `t_stats_` | ndarray | t-statistics |
| `p_values_` | ndarray | Two-sided p-values |
| `r_squared_` | float | R² |
| `residuals_` | ndarray | Residuals (in weighted space if WLS) |
| `fitted_values_` | ndarray | Fitted values |
| `fe_coefficients_` | dict[str, ndarray] | Recovered FE coefficients per FE variable |
| `n_categories` | dict[str, int] | Number of levels per FE variable |

---

### `HDFEIV`

```python
from HDFE import HDFEIV

model = HDFEIV(
    max_iter=5000,
    tolerance=1e-8,
    acceleration='gk',
    use_gpu=False,
    verbose=False,
)
```

Inherits all `HDFE` constructor parameters.

#### `.fit(data, y_col, X_cols, fe_vars, se_type='homoscedastic', cluster_vars=None, sample_weight=None, instruments=None, endogenous_vars=None, singular_fallback='lsqr')`

All `HDFE.fit()` parameters plus:

| Parameter | Type | Description |
|---|---|---|
| `instruments` | list[str] | Excluded instrument columns |
| `endogenous_vars` | list[str] | Endogenous variables (must be a subset of `X_cols`) |
| `singular_fallback` | str | `'lsqr'` (default) — use least-squares fallback; `'raise'` — raise `LinAlgError` |

When `instruments` or `endogenous_vars` is `None`/empty, falls back to standard HDFE-OLS.

**Raises:**
- `NotImplementedError` if `se_type` is `'hc2'` or `'hc3'` (not supported for IV)
- `ValueError` if `endogenous_vars` is not a subset of `X_cols`
- `ValueError` if `len(instruments) < len(endogenous_vars)` (under-identification)

Returns `self`.

#### `.first_stage_results()`

Returns a dict keyed by endogenous variable name:

```python
{
    'x_endog': {
        'coefficients': ndarray,       # first-stage π̂
        'instrument_names': list,      # [exog vars] + [excluded instruments]
        'r_squared': float,
        'f_statistic': float,          # partial F for excluded instruments
        'fitted_values': ndarray,
        'residuals': ndarray,
    },
    ...
}
```

#### `.iv_diagnostics()`

```python
{
    'weak_instruments': bool,          # True if min first-stage F < 10
    'first_stage_f_stats': dict,       # {endog_var: F} for each endogenous var
    'sargan_statistic': float | None,  # Sargan χ² (None if exactly identified)
    'sargan_pvalue': float | None,
}
```

#### `.summary()`

Prints second-stage results, first-stage F-stats (displays "N/A (singular)" when unavailable), Sargan test, FE summary, and any rank diagnostics warnings.

#### `.rank_diagnostics()`

Inherited from `HDFE` — includes IV-specific diagnostic keys (see HDFE section above).

---

## Quick Start

### HDFE — Basic usage

```python
import numpy as np, pandas as pd
from HDFE import HDFE

np.random.seed(42)
N = 100_000

firm = np.random.randint(0, 1000, N)
worker = np.random.randint(0, 500, N)
firm_fe = np.random.normal(0, 1, 1000)
worker_fe = np.random.normal(0, 1, 500)

X1 = np.random.normal(2, 1, N)
X2 = np.random.normal(0, 1, N)
y = 0.05 * X1 + 0.10 * X2 + firm_fe[firm] + worker_fe[worker] + np.random.normal(0, 0.1, N)

df = pd.DataFrame({
    'y': y, 'x1': X1, 'x2': X2,
    'firm': firm, 'worker': worker,
})

model = HDFE(verbose=True)
model.fit(df, 'y', ['x1', 'x2'], ['firm', 'worker'], se_type='hc1')
model.summary()
```

### HDFE-IV — Correcting endogeneity

```python
from HDFE import HDFEIV

np.random.seed(123)
N = 500_000

firm = np.random.randint(0, 800, N)
worker = np.random.randint(0, 400, N)
firm_fe = np.random.normal(0, 0.8, 800)
worker_fe = np.random.normal(0, 0.8, 400)

eps = np.random.normal(0, 0.2, N)
Z1 = np.random.normal(0, 1, N)
Z2 = np.random.normal(1, 1, N)

x_exog = np.random.normal(2, 1, N)
# Endogenous: correlated with eps (endogeneity)
x_endog = 0.7 * Z1 + 0.5 * Z2 + 0.8 * eps + np.random.normal(0, 0.5, N)

y = 0.08 * x_exog + 0.15 * x_endog + firm_fe[firm] + worker_fe[worker] + eps

df = pd.DataFrame({
    'y': y, 'x_exog': x_exog, 'x_endog': x_endog,
    'z1': Z1, 'z2': Z2,
    'firm': firm, 'worker': worker,
})

# OLS (biased)
ols = HDFE(verbose=False)
ols.fit(df, 'y', ['x_exog', 'x_endog'], ['firm', 'worker'])

# IV (corrected)
iv = HDFEIV(verbose=False)
iv.fit(df, 'y', ['x_exog', 'x_endog'], ['firm', 'worker'],
       instruments=['z1', 'z2'], endogenous_vars=['x_endog'],
       se_type='hc1')

print(f"True β_endog = 0.15")
print(f"OLS  β_endog = {ols.coefficients_[1]:.4f}  (biased)")
print(f"IV   β_endog = {iv.coefficients_[1]:.4f}  (corrected)")

iv.summary()
diag = iv.iv_diagnostics()
print(f"First-stage F: {diag['first_stage_f_stats']['x_endog']:.1f}")
print(f"Sargan p-value: {diag['sargan_pvalue']:.3f}")
```

### Weighted estimation

```python
w = np.random.exponential(1, N)
model = HDFE()
model.fit(df, 'y', ['x_exog', 'x_endog'], ['firm', 'worker'],
          se_type='cluster', cluster_vars=['firm'], sample_weight=w)
```

### Cluster-robust SEs

```python
model = HDFEIV()
model.fit(df, 'y', ['x_exog', 'x_endog'], ['firm', 'worker'],
          instruments=['z1', 'z2'], endogenous_vars=['x_endog'],
          se_type='cluster', cluster_vars=['firm'])
```