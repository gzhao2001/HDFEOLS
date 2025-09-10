# HDFEOLS.py
# High-Dimensional Fixed Effects Estimators (CPU and GPU versions)
import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import spsolve
import numba
from numba import jit, prange
import warnings

# GPU imports (optional)
try:
    import cupy as cp
    import cupyx.scipy.sparse
    import cupyx.scipy.sparse.linalg
    GPU_AVAILABLE = cp.is_available()
    print(f"CuPy available: {GPU_AVAILABLE}")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - GPU acceleration disabled")

warnings.filterwarnings('ignore')


class HDFE:
    """
    High-dimensional fixed effects regression using alternating projection method (CPU).
    
    Renamed from HighDimFixedEffects for easier importing.
    """
    
    def __init__(self, max_iter=100, tol=1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self.fe_vars = []
        self.fitted = False
        self.fe_coefficients_ = {}
        self.category_orders_ = {}
        self.category_to_index_ = {}
        self.n_categories = {}
        
    def _establish_category_ordering(self, data, fe_vars):
        """Establish consistent category ordering"""
        for fe_var in fe_vars:
            if fe_var not in self.category_orders_:
                unique_cats_str = data[fe_var].astype(str).unique()
                try:
                    unique_cats_numeric = sorted([int(cat) for cat in unique_cats_str])
                    unique_cats = [str(cat) for cat in unique_cats_numeric]
                except ValueError:
                    unique_cats = sorted(unique_cats_str)
                
                self.category_orders_[fe_var] = unique_cats
                self.category_to_index_[fe_var] = {cat: idx for idx, cat in enumerate(unique_cats)}
                self.n_categories[fe_var] = len(unique_cats)
                
        return self._encode_with_consistent_ordering(data, fe_vars)
        
    def _encode_with_consistent_ordering(self, data, fe_vars):
        """Encode categorical variables using consistent ordering"""
        encoded_data = data.copy()
        for fe_var in fe_vars:
            category_map = self.category_to_index_[fe_var]
            encoded_values = []
            for val in data[fe_var].astype(str):
                if val in category_map:
                    encoded_values.append(category_map[val])
                else:
                    encoded_values.append(-1)
            encoded_data[fe_var] = np.array(encoded_values)
        return encoded_data
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _demean_by_group(y, group_ids, n_groups):
        """Remove group means using Numba"""
        result = y.copy()
        group_sums = np.zeros(n_groups, dtype=np.float64)
        group_counts = np.zeros(n_groups, dtype=np.float64)
        
        for i in range(len(y)):
            if group_ids[i] >= 0:
                group_sums[group_ids[i]] += y[i]
                group_counts[group_ids[i]] += 1.0
        
        group_means = np.zeros(n_groups, dtype=np.float64)
        for j in range(n_groups):
            if group_counts[j] > 0:
                group_means[j] = group_sums[j] / group_counts[j]
        
        for i in range(len(y)):
            if group_ids[i] >= 0:
                result[i] = y[i] - group_means[group_ids[i]]
        
        return result, group_means

    def _alternating_projection(self, y, X, encoded_data, fe_vars):
        """Alternating projection algorithm"""
        y_proj = y.copy()
        X_proj = X.copy()
        self.projection_sequence_ = []
        
        print("Starting alternating projection algorithm...")
        
        for iteration in range(self.max_iter):
            y_old = y_proj.copy()
            X_old = X_proj.copy()
            
            for fe_var in fe_vars:
                group_ids = encoded_data[fe_var].values
                n_groups = self.n_categories[fe_var]
                
                y_proj, _ = self._demean_by_group(y_proj, group_ids, n_groups)
                
                for j in range(X_proj.shape[1]):
                    X_proj[:, j], _ = self._demean_by_group(X_proj[:, j], group_ids, n_groups)
            
            y_change = np.mean((y_proj - y_old)**2)
            X_change = np.mean((X_proj - X_old)**2)
            
            if iteration % 5 == 0:
                print(f"Iteration {iteration}: y_change = {y_change:.2e}, X_change = {X_change:.2e}")
            
            if y_change < self.tol and X_change < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
                
        return y_proj, X_proj

    def _build_dummy_matrix(self, encoded_data, fe_vars):
        """Build sparse dummy matrix for fixed effects"""
        n_obs = len(encoded_data)
        total_cols = self.n_categories[fe_vars[0]]
        for fe_var in fe_vars[1:]:
            total_cols += self.n_categories[fe_var] - 1
        
        row_indices = []
        col_indices = []
        data_values = []
        fe_col_info = {}
        current_col = 0
        
        for fe_idx, fe_var in enumerate(fe_vars):
            n_cats = self.n_categories[fe_var]
            group_ids = encoded_data[fe_var].values
            
            if fe_idx == 0:
                fe_col_info[fe_var] = {
                    'start_col': current_col, 'end_col': current_col + n_cats,
                    'n_categories': n_cats, 'dropped_category': None
                }
                for i in range(n_obs):
                    if group_ids[i] >= 0:
                        row_indices.append(i)
                        col_indices.append(current_col + group_ids[i])
                        data_values.append(1.0)
                current_col += n_cats
            else:
                fe_col_info[fe_var] = {
                    'start_col': current_col, 'end_col': current_col + n_cats - 1,
                    'n_categories': n_cats, 'dropped_category': 0
                }
                for i in range(n_obs):
                    if group_ids[i] >= 1:
                        row_indices.append(i)
                        col_indices.append(current_col + group_ids[i] - 1)
                        data_values.append(1.0)
                current_col += n_cats - 1
        
        D = csr_matrix((data_values, (row_indices, col_indices)), 
                       shape=(n_obs, total_cols), dtype=np.float64)
        
        return D, fe_col_info

    def _recover_fixed_effects(self, y, X, encoded_data, beta, y_projected, X_projected):
        """Recover fixed effects using sparse solver"""
        print("Recovering fixed effects...")
        
        D, fe_col_info = self._build_dummy_matrix(encoded_data, self.fe_vars)
        
        residuals_original = y - X @ beta
        residuals_projected = y_projected - X_projected @ beta
        rhs = residuals_original - residuals_projected
        
        try:
            DtD = D.T @ D
            Dtr = D.T @ rhs
            alpha = spsolve(DtD, Dtr)
        except Exception as e:
            print(f"Sparse solver failed: {e}")
            alpha = np.linalg.lstsq(D.toarray(), rhs, rcond=None)[0]
        
        fe_coefficients = {}
        for fe_idx, fe_var in enumerate(self.fe_vars):
            info = fe_col_info[fe_var]
            n_cats = info['n_categories']
            
            if fe_idx == 0:
                fe_coeffs = alpha[info['start_col']:info['end_col']]
            else:
                fe_coeffs = np.zeros(n_cats)
                fe_coeffs[1:] = alpha[info['start_col']:info['end_col']]
            
            fe_coefficients[fe_var] = fe_coeffs
        
        return fe_coefficients
    
    def fit(self, data, y_col, X_cols, fe_vars, sample_weight=None):
        """Fit the high-dimensional fixed effects model"""
        self.fe_vars = fe_vars
        self.y_col = y_col
        self.X_cols = X_cols
        
        if hasattr(data, 'to_pandas'):
            data = data.to_pandas()
            
        encoded_data = self._establish_category_ordering(data, fe_vars)
        
        y = data[y_col].values.astype(np.float64)
        X = data[X_cols].values.astype(np.float64)
        
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        y = y[valid_mask]
        X = X[valid_mask]
        encoded_data = encoded_data[valid_mask]
        
        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]
            y = y * np.sqrt(sample_weight)
            X = X * np.sqrt(sample_weight).reshape(-1, 1)
            
        print(f"Fitting model with {len(y):,} observations, {X.shape[1]} variables, and {len(fe_vars)} fixed effects")
        
        y_projected, X_projected = self._alternating_projection(y, X, encoded_data, fe_vars)
        
        XtX = X_projected.T @ X_projected
        Xty = X_projected.T @ y_projected
        
        self.coefficients_ = np.linalg.solve(XtX, Xty)
        
        self.fe_coefficients_ = self._recover_fixed_effects(y, X, encoded_data, self.coefficients_, 
                                                           y_projected, X_projected)
        
        # Calculate statistics
        y_pred_full = X @ self.coefficients_
        for fe_var in self.fe_vars:
            group_ids = encoded_data[fe_var].values
            for i in range(len(y)):
                if group_ids[i] >= 0:
                    y_pred_full[i] += self.fe_coefficients_[fe_var][group_ids[i]]
        
        residuals_full = y - y_pred_full
        self.residuals_ = residuals_full
        self.fitted_values_ = y_pred_full
        
        df_resid = len(y) - X.shape[1] - sum(self.n_categories.values())
        mse = np.sum(residuals_full**2) / df_resid
        var_coef = mse * np.linalg.inv(XtX)
        self.std_errors_ = np.sqrt(np.diag(var_coef))
        
        self.t_stats_ = self.coefficients_ / self.std_errors_
        self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df_resid))
        
        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum(residuals_full**2)
        self.r_squared_ = 1 - rss / tss
        
        self.fitted = True
        return self
    
    def summary(self):
        """Print regression summary"""
        if not self.fitted:
            raise ValueError("Model must be fitted before summary")
            
        print("=" * 80)
        print("HIGH-DIMENSIONAL FIXED EFFECTS REGRESSION RESULTS")
        print("(Alternating Projection Method)")
        print("=" * 80)
        print(f"R-squared: {self.r_squared_:.4f}")
        print(f"Number of fixed effects: {len(self.fe_vars)}")
        print(f"Fixed effect categories: {dict(self.n_categories)}")
        
        print("\nMain Coefficients:")
        print("-" * 80)
        print(f"{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<8} {'P>|t|':<8}")
        print("-" * 80)
        
        for i, var in enumerate(self.X_cols):
            print(f"{var:<20} {self.coefficients_[i]:<12.4f} {self.std_errors_[i]:<12.4f} "
                  f"{self.t_stats_[i]:<8.3f} {self.p_values_[i]:<8.3f}")
        
        print("\nFixed Effects Summary:")
        print("-" * 40)
        for fe_var in self.fe_vars:
            fe_coeffs = self.fe_coefficients_[fe_var]
            print(f"{fe_var}: mean={np.mean(fe_coeffs):.4f}, std={np.std(fe_coeffs):.4f}, "
                  f"min={np.min(fe_coeffs):.4f}, max={np.max(fe_coeffs):.4f}")
        
        print("=" * 80)


class HDFEGpu(HDFE):
    """
    GPU-accelerated high-dimensional fixed effects regression.
    
    Renamed from FinalFixedGPUHighDimFixedEffects for easier importing.
    """
    
    def __init__(self, max_iter=100, tol=1e-8, use_gpu=True):
        super().__init__(max_iter, tol)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.device_info = {}
        
        if self.use_gpu:
            self._initialize_gpu()
        else:
            print("GPU acceleration disabled - using CPU implementation")

    def _initialize_gpu(self):
        """Initialize GPU resources"""
        try:
            device = cp.cuda.Device()
            device_name = device.attributes.get('Name', 'Unknown GPU')
            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            
            self.device_info = {
                'name': device_name,
                'memory': (free_memory, total_memory),
                'device_id': device.id
            }
            
            print(f"GPU Initialized: {device_name}")
            print(f"Available Memory: {free_memory / 1024**3:.1f} GB / {total_memory / 1024**3:.1f} GB")
            
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            self.use_gpu = False

    @staticmethod
    def _gpu_group_demean_final(y_gpu, group_ids_gpu, n_groups):
        """GPU-accelerated group demeaning"""
        group_sums = cp.bincount(group_ids_gpu, weights=y_gpu, minlength=n_groups)
        group_counts = cp.bincount(group_ids_gpu, minlength=n_groups)
        
        group_means = cp.zeros(n_groups, dtype=cp.float64)
        nonzero_mask = group_counts > 0
        group_means[nonzero_mask] = group_sums[nonzero_mask] / group_counts[nonzero_mask]
        
        result = y_gpu - group_means[group_ids_gpu]
        return result, group_means
    
    def _alternating_projection_gpu_final(self, y, X, encoded_data, fe_vars):
        """GPU-accelerated alternating projection"""
        print("Starting GPU-accelerated alternating projection...")
        
        y_gpu = cp.asarray(y, dtype=cp.float64)
        X_gpu = cp.asarray(X, dtype=cp.float64)
        
        group_ids_list = []
        fe_var_names = []
        n_groups_list = []
        
        for fe_var in fe_vars:
            group_ids_gpu = cp.asarray(encoded_data[fe_var].values, dtype=cp.int32)
            group_ids_list.append(group_ids_gpu)
            fe_var_names.append(fe_var)
            n_groups_list.append(self.n_categories[fe_var])
        
        y_proj_gpu = y_gpu.copy()
        X_proj_gpu = X_gpu.copy()
        
        for iteration in range(self.max_iter):
            y_old_gpu = y_proj_gpu.copy()
            X_old_gpu = X_proj_gpu.copy()
            
            for idx, fe_var in enumerate(fe_var_names):
                group_ids = group_ids_list[idx]
                n_groups = n_groups_list[idx]
                
                y_proj_gpu, _ = self._gpu_group_demean_final(y_proj_gpu, group_ids, n_groups)
                
                for j in range(X_proj_gpu.shape[1]):
                    X_proj_gpu[:, j], _ = self._gpu_group_demean_final(X_proj_gpu[:, j], group_ids, n_groups)
            
            y_change = float(cp.mean((y_proj_gpu - y_old_gpu)**2))
            X_change = float(cp.mean((X_proj_gpu - X_old_gpu)**2))
            
            if iteration % 5 == 0:
                print(f"GPU Iteration {iteration}: y_change = {y_change:.2e}, X_change = {X_change:.2e}")
            
            if y_change < self.tol and X_change < self.tol:
                print(f"GPU algorithm converged after {iteration + 1} iterations")
                break
        
        y_projected = cp.asnumpy(y_proj_gpu)
        X_projected = cp.asnumpy(X_proj_gpu)
        
        # Cleanup
        del y_gpu, X_gpu, y_proj_gpu, X_proj_gpu
        for group_ids_gpu in group_ids_list:
            del group_ids_gpu
        
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        
        return y_projected, X_projected

    def _recover_fixed_effects(self, y, X, encoded_data, beta, y_projected, X_projected):
        """Fixed effects recovery with GPU sparse solving"""
        print("Recovering fixed effects with GPU sparse solving...")
        
        residuals_original = y - X @ beta
        residuals_projected = y_projected - X_projected @ beta
        rhs_full = residuals_original - residuals_projected
        
        fe_vars = self.fe_vars
        n_obs = len(y)
        total_cols = sum(self.n_categories[fe_var] for fe_var in fe_vars) - (len(fe_vars) - 1)
        
        D_blocks = []
        for i, fe_var in enumerate(fe_vars):
            group_ids = encoded_data[fe_var].values
            n_groups = self.n_categories[fe_var]
            
            if i == 0:
                n_cols = n_groups
                row_indices = np.arange(n_obs)
                col_indices = group_ids
                data = np.ones(n_obs)
                valid_mask = col_indices < n_cols
                
                D_block = csr_matrix(
                    (data[valid_mask], (row_indices[valid_mask], col_indices[valid_mask])),
                    shape=(n_obs, n_cols)
                )
            else:
                n_cols = n_groups - 1
                mask = group_ids > 0
                row_indices = np.where(mask)[0]
                col_indices = group_ids[mask] - 1
                data = np.ones(len(row_indices))
                
                D_block = csr_matrix(
                    (data, (row_indices, col_indices)),
                    shape=(n_obs, n_cols)
                )
            
            D_blocks.append(D_block)
        
        D = hstack(D_blocks, format='csr')
        DtD = D.T @ D
        DtR = D.T @ rhs_full
        
        # GPU sparse solving
        try:
            print("🚀 Using GPU for sparse solving...")
            DtD_gpu = cupyx.scipy.sparse.csr_matrix(DtD.astype(np.float64))
            DtR_gpu = cp.asarray(DtR.astype(np.float64))
            alpha_gpu = cupyx.scipy.sparse.linalg.spsolve(DtD_gpu, DtR_gpu)
            alpha = cp.asnumpy(alpha_gpu)
            print("✅ GPU sparse solve successful!")
        except Exception as e:
            print(f"GPU solve failed ({e}), falling back to CPU...")
            alpha = spsolve(DtD, DtR)
        
        # Reconstruct FE coefficients
        fe_coefficients = {}
        current_col = 0
        
        for i, fe_var in enumerate(fe_vars):
            n_groups = self.n_categories[fe_var]
            
            if i == 0:
                fe_coeffs = alpha[current_col:current_col+n_groups]
                current_col += n_groups
            else:
                fe_coeffs = np.zeros(n_groups)
                fe_coeffs[1:] = alpha[current_col:current_col+n_groups-1]
                current_col += n_groups - 1
            
            fe_coefficients[fe_var] = fe_coeffs
        
        return fe_coefficients
    
    def _alternating_projection(self, y, X, encoded_data, fe_vars):
        """Override with GPU implementation"""
        if self.use_gpu:
            try:
                return self._alternating_projection_gpu_final(y, X, encoded_data, fe_vars)
            except Exception as e:
                print(f"GPU alternating projection failed: {e}")
                print("Falling back to CPU implementation...")
                self.use_gpu = False
                return super()._alternating_projection(y, X, encoded_data, fe_vars)
        else:
            return super()._alternating_projection(y, X, encoded_data, fe_vars)
    
    def fit(self, data, y_col, X_cols, fe_vars, sample_weight=None):
        """Fit model with GPU acceleration"""
        print(f"Fitting model with {'GPU' if self.use_gpu else 'CPU'} acceleration...")
        return super().fit(data, y_col, X_cols, fe_vars, sample_weight)
    
    def summary(self):
        """Print regression summary"""
        if not self.fitted:
            raise ValueError("Model must be fitted before summary")
            
        print("=" * 80)
        print("HIGH-DIMENSIONAL FIXED EFFECTS REGRESSION RESULTS")
        print("(Alternating Projection Method)")
        print("=" * 80)
        print(f"R-squared: {self.r_squared_:.4f}")
        print(f"Number of fixed effects: {len(self.fe_vars)}")
        print(f"Fixed effect categories: {dict(self.n_categories)}")
        
        print("\nMain Coefficients:")
        print("-" * 80)
        print(f"{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<8} {'P>|t|':<8}")
        print("-" * 80)
        
        for i, var in enumerate(self.X_cols):
            print(f"{var:<20} {self.coefficients_[i]:<12.4f} {self.std_errors_[i]:<12.4f} "
                  f"{self.t_stats_[i]:<8.3f} {self.p_values_[i]:<8.3f}")
        
        print("\nFixed Effects Summary:")
        print("-" * 40)
        for fe_var in self.fe_vars:
            fe_coeffs = self.fe_coefficients_[fe_var]
            print(f"{fe_var}: mean={np.mean(fe_coeffs):.4f}, std={np.std(fe_coeffs):.4f}, "
                  f"min={np.min(fe_coeffs):.4f}, max={np.max(fe_coeffs):.4f}")

        if self.use_gpu and self.device_info:
            print(f"GPU: {self.device_info['name']}")

        print("=" * 80)
    


# Test when run directly
if __name__ == "__main__":
    print("High-Dimensional Fixed Effects Estimators")
    print(f"GPU Available: {GPU_AVAILABLE}")
    print("Classes available: HDFE (CPU), HDFEGpu (GPU)")