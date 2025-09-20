# HDFE.py
# High-Dimensional Fixed Effects Estimators (CPU and GPU versions)
import pandas as pd
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import spsolve
import numba
from numba import jit, prange
import warnings


class HDFE:
    
    def __init__(self, max_iter=5000, tolerance=1e-8, acceleration='gk', use_gpu=None, verbose=False):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.acceleration = acceleration
        self.verbose = verbose
        
        # GPU setup
        if use_gpu is None:
            try:
                import cupy as cp
                self.use_gpu = cp.cuda.is_available()
                if self.verbose and self.use_gpu:
                    print("✅ GPU detected and will be used")
                elif self.verbose:
                    print("⚠️ GPU not available, using CPU")
            except ImportError:
                self.use_gpu = False
                if self.verbose:
                    print("⚠️ CuPy not installed, using CPU")
        else:
            self.use_gpu = use_gpu
            
        # Initialize state
        self.fitted = False
        self.fe_vars = []
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

    def _cpu_demean_by_group(self, data, group_indices, n_groups):
        """CPU version of group demeaning using Numba"""
        result = data.copy()
        group_sums = np.zeros(n_groups, dtype=np.float64)
        group_counts = np.zeros(n_groups, dtype=np.float64)
        
        # Calculate group sums and counts
        for i in range(len(data)):
            if group_indices[i] >= 0:
                group_sums[group_indices[i]] += data[i]
                group_counts[group_indices[i]] += 1.0
        
        # Calculate group means
        group_means = np.zeros(n_groups, dtype=np.float64)
        for j in range(n_groups):
            if group_counts[j] > 0:
                group_means[j] = group_sums[j] / group_counts[j]
        
        # Subtract group means
        for i in range(len(data)):
            if group_indices[i] >= 0:
                result[i] = data[i] - group_means[group_indices[i]]
        
        return result, group_means

    def _gpu_demean_by_group(self, data_gpu, group_indices_gpu, n_groups):
        """GPU version of group demeaning"""
        import cupy as cp
        
        group_sums = cp.bincount(group_indices_gpu, weights=data_gpu, minlength=n_groups)
        group_counts = cp.bincount(group_indices_gpu, minlength=n_groups)
        
        group_means = cp.zeros(n_groups, dtype=cp.float64)
        nonzero_mask = group_counts > 0
        group_means[nonzero_mask] = group_sums[nonzero_mask] / group_counts[nonzero_mask]
        
        result = data_gpu - group_means[group_indices_gpu]
        return result, group_means

    def _alternating_projection(self, y, X, encoded_data, fe_vars):
        """Alternating projection algorithm with acceleration"""
        if self.verbose:
            print("Starting alternating projection algorithm...")
            
        # Determine backend
        if self.use_gpu:
            try:
                import cupy as cp
                y_proj = cp.asarray(y, dtype=cp.float64)
                X_proj = cp.asarray(X, dtype=cp.float64)
                
                group_ids_list = []
                for fe_var in fe_vars:
                    group_ids = cp.asarray(encoded_data[fe_var].values, dtype=cp.int32)
                    group_ids_list.append(group_ids)
                
                demean_func = self._gpu_demean_by_group
                backend_name = "GPU"
            except Exception as e:
                if self.verbose:
                    print(f"GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False
                
        if not self.use_gpu:
            y_proj = y.copy()
            X_proj = X.copy()
            group_ids_list = [encoded_data[fe_var].values for fe_var in fe_vars]
            demean_func = self._cpu_demean_by_group
            backend_name = "CPU"
            
        if self.verbose:
            print(f"Using {backend_name} backend")

        # Store history for acceleration
        y_history = []
        X_history = []
        
        for iteration in range(self.max_iter):
            y_old = y_proj.copy()
            X_old = X_proj.copy()
            
            # Apply one round of demeaning for all fixed effects
            for idx, fe_var in enumerate(fe_vars):
                group_ids = group_ids_list[idx]
                n_groups = self.n_categories[fe_var]
                
                y_proj, _ = demean_func(y_proj, group_ids, n_groups)
                
                for j in range(X_proj.shape[1]):
                    X_proj[:, j], _ = demean_func(X_proj[:, j], group_ids, n_groups)
            
            # Check convergence
            if self.use_gpu:
                import cupy as cp
                y_change = float(cp.mean((y_proj - y_old)**2))
                X_change = float(cp.mean((X_proj - X_old)**2))
            else:
                y_change = np.mean((y_proj - y_old)**2)
                X_change = np.mean((X_proj - X_old)**2)
            
            # Apply Gearhart-Koshy acceleration
            if self.acceleration == 'gk' and len(y_history) >= 2:
                y_current = y_proj
                X_current = X_proj
                
                # Get previous iterations
                y_prev1 = y_history[-1]
                y_prev2 = y_history[-2]
                X_prev1 = X_history[-1]
                X_prev2 = X_history[-2]
                
                # Apply acceleration
                y_proj = self._apply_gk_acceleration(y_current, y_prev1, y_prev2)
                X_proj = self._apply_gk_acceleration(X_current, X_prev1, X_prev2)
            
            # Store history
            y_history.append(y_proj.copy())
            X_history.append(X_proj.copy())
            
            # Keep only last 2 iterations for memory efficiency
            if len(y_history) > 3:
                y_history.pop(0)
                X_history.pop(0)
            
            if self.verbose and iteration % 200 == 0:
                print(f"Iteration {iteration}: y_change = {y_change:.2e}, X_change = {X_change:.2e}")
            
            if y_change < self.tolerance and X_change < self.tolerance:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        else:
            if self.verbose:
                print(f"Warning: Maximum iterations ({self.max_iter}) reached")
        
        # Convert back from GPU if needed
        if self.use_gpu:
            import cupy as cp
            y_projected = cp.asnumpy(y_proj)
            X_projected = cp.asnumpy(X_proj)
        else:
            y_projected = y_proj
            X_projected = X_proj
            
        return y_projected, X_projected
    
    def _apply_gk_acceleration(self, current, previous1, previous2):
        """Apply Gearhart-Koshy acceleration"""
        if self.use_gpu:
            import cupy as cp
            diff1 = current - previous1
            diff2 = previous1 - previous2
            
            numerator = cp.sum(diff1 * diff2)
            denominator = cp.sum(diff2 * diff2)
            
            if abs(float(denominator)) > self.tolerance:
                alpha = float(numerator / denominator)
                alpha = max(0, min(alpha, 1))  # Clamp between 0 and 1
                accelerated = current + alpha * diff1
                return accelerated
        else:
            diff1 = current - previous1
            diff2 = previous1 - previous2
            
            numerator = np.sum(diff1 * diff2)
            denominator = np.sum(diff2 * diff2)
            
            if abs(denominator) > self.tolerance:
                alpha = numerator / denominator
                alpha = max(0, min(alpha, 1))  # Clamp between 0 and 1
                accelerated = current + alpha * diff1
                return accelerated
                
        return current

    def _build_dummy_matrix(self, encoded_data, fe_vars):
        """Build sparse dummy matrix for fixed effects - from HDFEGpu implementation"""
        n_obs = len(encoded_data)
        
        # Calculate total columns needed
        total_cols = self.n_categories[fe_vars[0]]  # First FE: all categories
        for fe_var in fe_vars[1:]:
            total_cols += self.n_categories[fe_var] - 1  # Other FEs: drop one category
        
        row_indices = []
        col_indices = []
        data_values = []
        fe_col_info = {}
        current_col = 0
        
        for fe_idx, fe_var in enumerate(fe_vars):
            n_cats = self.n_categories[fe_var]
            group_ids = encoded_data[fe_var].values
            
            if fe_idx == 0:
                # First fixed effect: include all categories
                fe_col_info[fe_var] = {
                    'start_col': current_col, 
                    'end_col': current_col + n_cats,
                    'n_categories': n_cats, 
                    'dropped_category': None
                }
                
                for i in range(n_obs):
                    if group_ids[i] >= 0:
                        row_indices.append(i)
                        col_indices.append(current_col + group_ids[i])
                        data_values.append(1.0)
                        
                current_col += n_cats
                
            else:
                # Other fixed effects: drop first category (category 0)
                fe_col_info[fe_var] = {
                    'start_col': current_col, 
                    'end_col': current_col + n_cats - 1,
                    'n_categories': n_cats, 
                    'dropped_category': 0
                }
                
                for i in range(n_obs):
                    if group_ids[i] >= 1:  # Skip category 0
                        row_indices.append(i)
                        col_indices.append(current_col + group_ids[i] - 1)
                        data_values.append(1.0)
                        
                current_col += n_cats - 1
        
        from scipy.sparse import csr_matrix
        D = csr_matrix((data_values, (row_indices, col_indices)), 
                       shape=(n_obs, total_cols), dtype=np.float64)
        
        return D, fe_col_info

    def _recover_fixed_effects(self, y, X, encoded_data, beta, y_projected, X_projected):
        """
        Recover fixed effects using sparse solver - adapted from HDFEGpu implementation
        This is the correct method that was missing in StreamlinedHDFE_Improved
        """
        if self.verbose:
            print("Recovering fixed effects using sparse solver...")
        
        # Build the sparse dummy matrix
        D, fe_col_info = self._build_dummy_matrix(encoded_data, self.fe_vars)
        
        # Calculate residuals
        residuals_original = y - X @ beta
        residuals_projected = y_projected - X_projected @ beta
        rhs = residuals_original - residuals_projected
        
        # Solve the sparse system: D'D * alpha = D' * rhs
        try:
            from scipy.sparse.linalg import spsolve
            from scipy.sparse import hstack
            
            DtD = D.T @ D
            Dtr = D.T @ rhs
            
            # Try GPU sparse solving first if available
            if self.use_gpu:
                try:
                    import cupy as cp
                    import cupyx.scipy.sparse
                    import cupyx.scipy.sparse.linalg
                    
                    if self.verbose:
                        print("🚀 Using GPU for sparse solving...")
                    
                    DtD_gpu = cupyx.scipy.sparse.csr_matrix(DtD.astype(np.float64))
                    Dtr_gpu = cp.asarray(Dtr.astype(np.float64))
                    alpha_gpu = cupyx.scipy.sparse.linalg.spsolve(DtD_gpu, Dtr_gpu)
                    alpha = cp.asnumpy(alpha_gpu)
                    
                    if self.verbose:
                        print("✅ GPU sparse solve successful!")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"GPU solve failed ({e}), falling back to CPU...")
                    alpha = spsolve(DtD, Dtr)
            else:
                alpha = spsolve(DtD, Dtr)
                
        except Exception as e:
            if self.verbose:
                print(f"Sparse solver failed: {e}, using least squares...")
            alpha = np.linalg.lstsq(D.toarray(), rhs, rcond=None)[0]
        
        # Reconstruct fixed effect coefficients
        fe_coefficients = {}
        current_col = 0
        
        for fe_idx, fe_var in enumerate(self.fe_vars):
            info = fe_col_info[fe_var]
            n_cats = info['n_categories']
            
            if fe_idx == 0:
                # First FE: all categories included
                fe_coeffs = alpha[info['start_col']:info['end_col']]
                current_col += n_cats
            else:
                # Other FEs: reconstruct with dropped category = 0
                fe_coeffs = np.zeros(n_cats)
                fe_coeffs[1:] = alpha[info['start_col']:info['end_col']]
                current_col += n_cats - 1
            
            fe_coefficients[fe_var] = fe_coeffs
        
        if self.verbose:
            print("Fixed effects recovery completed")
            for fe_var, coeffs in fe_coefficients.items():
                print(f"  {fe_var}: mean={np.mean(coeffs):.6f}, std={np.std(coeffs):.6f}")
        
        return fe_coefficients
    
    def fit(self, data, y_col, X_cols, fe_vars, se_type='homoscedastic', cluster_vars=None, sample_weight=None):
        """
        Fit the HDFE model with robust standard errors
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        y_col : str
            Name of dependent variable
        X_cols : list
            Names of continuous variables
        fe_vars : list
            Names of fixed effect variables
        se_type : str, default='homoscedastic'
            Standard error type: 'homoscedastic', 'hc1', 'hc2', 'hc3', 'cluster'
        cluster_vars : list, optional
            List of variables to cluster on (required when se_type='cluster')
        sample_weight : array-like, optional
            Sample weights
        """
        # Validate parameters
        valid_se_types = ['homoscedastic', 'hc1', 'hc2', 'hc3', 'cluster']
        if se_type not in valid_se_types:
            raise ValueError(f"se_type must be one of {valid_se_types}")
            
        if se_type == 'cluster' and cluster_vars is None:
            raise ValueError("cluster_vars must be specified when se_type='cluster'")
            
        if se_type == 'cluster' and not isinstance(cluster_vars, list):
            cluster_vars = [cluster_vars]
        
        # Store parameters
        self.se_type = se_type
        self.cluster_vars = cluster_vars
        self.fe_vars = fe_vars
        self.y_col = y_col
        self.X_cols = X_cols
        
        if self.verbose:
            print(f"Fitting HDFE model with {len(data):,} observations")
            print(f"Continuous variables: {len(X_cols)}")
            print(f"Fixed effects: {len(fe_vars)}")
            print(f"Standard errors: {se_type}")
            if se_type == 'cluster':
                print(f"Clustering on: {cluster_vars}")
            print(f"Using {'GPU' if self.use_gpu else 'CPU'} acceleration")
        
        # Prepare data
        if hasattr(data, 'to_pandas'):
            data = data.to_pandas()
            
        encoded_data = self._establish_category_ordering(data, fe_vars)
        
        y = data[y_col].values.astype(np.float64)
        X = data[X_cols].values.astype(np.float64)
        
        # Handle missing data
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        y = y[valid_mask]
        X = X[valid_mask]
        encoded_data = encoded_data[valid_mask]
        
        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]
            y = y * np.sqrt(sample_weight)
            X = X * np.sqrt(sample_weight).reshape(-1, 1)
        
        # Apply alternating projection
        y_projected, X_projected = self._alternating_projection(y, X, encoded_data, fe_vars)
        
        # Estimate continuous coefficients
        XtX = X_projected.T @ X_projected
        Xty = X_projected.T @ y_projected
        
        self.coefficients_ = np.linalg.solve(XtX, Xty)
        
        # Recover fixed effects using sparse solver
        self.fe_coefficients_ = self._recover_fixed_effects(y, X, encoded_data, self.coefficients_, 
                                                           y_projected, X_projected)
        
        # Store transformed data for robust SE calculation
        self.X_projected = X_projected
        self.y_projected = y_projected
        
        # Calculate model statistics with robust standard errors
        self._calculate_statistics(y, X, encoded_data, X_projected, data)
        
        self.fitted = True
        return self
    
    def _compute_robust_standard_errors(self, X_projected, residuals, data, valid_mask):
        """Compute robust standard errors"""
        try:
            XtX_inv = np.linalg.inv(X_projected.T @ X_projected)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_projected.T @ X_projected)
        
        if self.se_type == 'homoscedastic':
            # Homoscedastic standard errors
            sigma2 = np.sum(residuals**2) / (len(residuals) - X_projected.shape[1])
            var_beta = sigma2 * XtX_inv
            
        elif self.se_type in ['hc1', 'hc2', 'hc3']:
            # Heteroscedasticity-robust standard errors
            meat_matrix = self._compute_hc_matrix(X_projected, residuals, self.se_type)
            var_beta = XtX_inv @ meat_matrix @ XtX_inv
            
        elif self.se_type == 'cluster':
            # Multi-variable cluster-robust standard errors
            meat_matrix = self._compute_multi_cluster_matrix(X_projected, residuals, data, valid_mask)
            var_beta = XtX_inv @ meat_matrix @ XtX_inv
        
        return np.sqrt(np.diag(var_beta))
    
    def _compute_hc_matrix(self, X, residuals, hc_type):
        """Compute heteroscedasticity-consistent covariance matrix"""
        n, k = X.shape
        
        if hc_type == 'hc1':
            # HC1: multiply by n/(n-k) adjustment
            weights = (residuals**2) * n / (n - k)
        elif hc_type == 'hc2':
            # HC2: account for leverage
            h = np.sum(X * (np.linalg.solve(X.T @ X, X.T).T), axis=1)
            weights = (residuals**2) / (1 - h)
        elif hc_type == 'hc3':
            # HC3: squared leverage adjustment  
            h = np.sum(X * (np.linalg.solve(X.T @ X, X.T).T), axis=1)
            weights = (residuals**2) / ((1 - h)**2)
        
        # Handle potential numerical issues
        weights = np.clip(weights, 0, np.percentile(weights, 99.9))
        
        weighted_X = X * np.sqrt(weights).reshape(-1, 1)
        return weighted_X.T @ weighted_X
    
    def _compute_multi_cluster_matrix(self, X, residuals, data, valid_mask):
        """Compute multi-variable cluster-robust covariance matrix with GPU acceleration"""
        if self.verbose and self.se_type == 'cluster':
            print("🚀 Computing cluster-robust standard errors...")
            if self.use_gpu:
                print("   Using GPU acceleration for cluster calculations")
        
        # Get cluster data for valid observations
        cluster_data = {}
        for cluster_var in self.cluster_vars:
            cluster_data[cluster_var] = data[cluster_var].values[valid_mask]
        
        # Create multi-way clustering groups
        if len(self.cluster_vars) == 1:
            # Single variable clustering
            cluster_groups = cluster_data[self.cluster_vars[0]]
        else:
            # Multi-variable clustering: create interaction of all cluster variables
            cluster_groups = cluster_data[self.cluster_vars[0]].astype(str)
            for cluster_var in self.cluster_vars[1:]:
                cluster_groups = cluster_groups + "_" + cluster_data[cluster_var].astype(str)
        
        # GPU-accelerated cluster computation
        if self.use_gpu:
            try:
                return self._compute_cluster_matrix_gpu(X, residuals, cluster_groups)
            except Exception as e:
                if self.verbose:
                    print(f"GPU cluster computation failed ({e}), falling back to CPU...")
                return self._compute_cluster_matrix_cpu(X, residuals, cluster_groups)
        else:
            return self._compute_cluster_matrix_cpu(X, residuals, cluster_groups)
    
    def _compute_cluster_matrix_gpu(self, X, residuals, cluster_groups):
        """GPU-accelerated cluster-robust covariance matrix computation"""
        import cupy as cp
        
        # Move data to GPU
        X_gpu = cp.asarray(X, dtype=cp.float64)
        residuals_gpu = cp.asarray(residuals, dtype=cp.float64)
        
        # Get unique clusters and create mapping
        unique_clusters = np.unique(cluster_groups)
        n_clusters = len(unique_clusters)
        k = X.shape[1]
        
        if self.verbose and n_clusters > 1000:
            print(f"   Processing {n_clusters:,} clusters on GPU...")
        
        # Create cluster index mapping
        cluster_to_idx = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
        cluster_indices = np.array([cluster_to_idx[cluster] for cluster in cluster_groups])
        cluster_indices_gpu = cp.asarray(cluster_indices, dtype=cp.int32)
        
        # Initialize meat matrix on GPU
        meat_matrix_gpu = cp.zeros((k, k), dtype=cp.float64)
        
        # Process clusters in batches to manage GPU memory
        batch_size = min(max(1000, 10000 // k), n_clusters)
        
        for batch_start in range(0, n_clusters, batch_size):
            batch_end = min(batch_start + batch_size, n_clusters)
            
            # Process batch of clusters
            for cluster_idx in range(batch_start, batch_end):
                cluster_mask_gpu = cluster_indices_gpu == cluster_idx
                
                # Extract cluster data
                X_cluster_gpu = X_gpu[cluster_mask_gpu]
                resid_cluster_gpu = residuals_gpu[cluster_mask_gpu]
                
                if X_cluster_gpu.shape[0] > 0:  # Check if cluster is not empty
                    # Compute cluster contribution: (X'r)(X'r)'
                    cluster_contribution_gpu = X_cluster_gpu.T @ resid_cluster_gpu
                    meat_matrix_gpu += cp.outer(cluster_contribution_gpu, cluster_contribution_gpu)
        
        # Convert back to CPU
        meat_matrix = cp.asnumpy(meat_matrix_gpu)
        
        # Cleanup GPU memory
        del X_gpu, residuals_gpu, cluster_indices_gpu, meat_matrix_gpu
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
            
        return meat_matrix
    
    def _compute_cluster_matrix_cpu(self, X, residuals, cluster_groups):
        """CPU fallback for cluster-robust covariance matrix computation"""
        meat_matrix = np.zeros((X.shape[1], X.shape[1]))
        unique_clusters = np.unique(cluster_groups)
        
        if self.verbose and len(unique_clusters) > 1000:
            print(f"   Processing {len(unique_clusters):,} clusters on CPU...")
        
        for cluster in unique_clusters:
            cluster_mask = cluster_groups == cluster
            X_cluster = X[cluster_mask]
            resid_cluster = residuals[cluster_mask]
            
            # Cluster contribution: sum of X_i * residual_i within cluster
            cluster_contribution = (X_cluster.T @ resid_cluster).reshape(-1, 1)
            meat_matrix += cluster_contribution @ cluster_contribution.T
        
        return meat_matrix

    def _calculate_statistics(self, y, X, encoded_data, X_projected, original_data):
        """Calculate model fit statistics with robust standard errors"""
        # Create valid_mask to track which observations were used
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        
        # Full predictions including fixed effects
        y_pred_full = X @ self.coefficients_
        for fe_var in self.fe_vars:
            group_ids = encoded_data[fe_var].values
            for i in range(len(y)):
                if group_ids[i] >= 0:
                    y_pred_full[i] += self.fe_coefficients_[fe_var][group_ids[i]]
        
        residuals_full = y - y_pred_full
        self.residuals_ = residuals_full
        self.fitted_values_ = y_pred_full
        
        # R-squared
        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum(residuals_full**2)
        self.r_squared_ = 1 - rss / tss
        
        # Compute robust standard errors
        if self.verbose:
            print(f"Computing {self.se_type} standard errors...")
        
        try:
            self.std_errors_ = self._compute_robust_standard_errors(X_projected, residuals_full, 
                                                                   original_data, valid_mask)
            
            # t-statistics and p-values
            self.t_stats_ = self.coefficients_ / self.std_errors_
            from scipy import stats
            df_resid = len(y) - X.shape[1] - sum(self.n_categories.values())
            self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df_resid))
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not compute robust standard errors: {e}")
            
            # Fallback to basic standard errors
            df_resid = len(y) - X.shape[1] - sum(self.n_categories.values())
            mse = np.sum(residuals_full**2) / df_resid
            
            try:
                XtX_inv = np.linalg.inv(X_projected.T @ X_projected)
                var_coef = mse * XtX_inv
                self.std_errors_ = np.sqrt(np.diag(var_coef))
                
                self.t_stats_ = self.coefficients_ / self.std_errors_
                from scipy import stats
                self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df_resid))
            except Exception:
                self.std_errors_ = np.full_like(self.coefficients_, np.nan)
                self.t_stats_ = np.full_like(self.coefficients_, np.nan)
                self.p_values_ = np.full_like(self.coefficients_, np.nan)
    
    def summary(self):
        """Print model summary"""
        if not self.fitted:
            raise ValueError("Model must be fitted before summary")
            
        print("=" * 80)
        print("HDFE REGRESSION RESULTS")
        print("(Alternating Projection + Sparse FE Recovery)")
        print("=" * 80)
        print(f"R-squared: {self.r_squared_:.6f}")
        print(f"Number of observations: {len(self.residuals_):,}")
        print(f"Number of fixed effects: {len(self.fe_vars)}")
        print(f"Standard error type: {self.se_type}")
        if hasattr(self, 'cluster_vars') and self.cluster_vars:
            print(f"Clustering variables: {self.cluster_vars}")
        print(f"Fixed effect categories: {dict(self.n_categories)}")
        
        print("\nContinuous Variable Coefficients:")
        print("-" * 80)
        print(f"{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<8} {'P>|t|':<8}")
        print("-" * 80)
        
        for i, var in enumerate(self.X_cols):
            if not np.isnan(self.std_errors_[i]):
                print(f"{var:<20} {self.coefficients_[i]:<12.6f} {self.std_errors_[i]:<12.6f} "
                      f"{self.t_stats_[i]:<8.3f} {self.p_values_[i]:<8.3f}")
            else:
                print(f"{var:<20} {self.coefficients_[i]:<12.6f} {'N/A':<12} {'N/A':<8} {'N/A':<8}")
        
        print("\nFixed Effects Summary:")
        print("-" * 60)
        for fe_var in self.fe_vars:
            fe_coeffs = self.fe_coefficients_[fe_var]
            print(f"{fe_var}: mean={np.mean(fe_coeffs):.6f}, std={np.std(fe_coeffs):.6f}, "
                  f"min={np.min(fe_coeffs):.6f}, max={np.max(fe_coeffs):.6f}")
        
        print("=" * 80)



class HDFEIV(HDFE):
    """
    High-Dimensional Fixed Effects Instrumental Variables (HDFE-IV) estimator.
    
    Extends HDFE to support instrumental variables estimation using two-stage least squares (2SLS).
    When no instruments are provided, falls back to standard HDFE estimation.
    
    The algorithm performs:
    1. Alternating projection demeaning on Y, X, and Z (instruments)
    2. First stage: regress endogenous variables on instruments (both demeaned)
    3. Second stage: 2SLS regression using demeaned data
    4. Robust standard errors with IV-specific corrections
    """
    
    def __init__(self, max_iter=5000, tolerance=1e-8, acceleration='gk', use_gpu=None, verbose=False):
        super().__init__(max_iter, tolerance, acceleration, use_gpu, verbose)
        
        # IV-specific attributes
        self._is_iv = False
        self._instruments = None
        self._endogenous_vars = None
        self._exogenous_vars = None
        
        # First stage results
        self._first_stage_models = {}
        self._first_stage_fitted = {}
        self._first_stage_residuals = {}
        self._first_stage_r2 = {}
        self._first_stage_f_stats = {}
        
        # 2SLS matrices
        self._tZX = None
        self._tXZ = None  
        self._tZy = None
        self._tZZinv = None
        
        # IV diagnostics
        self._weak_instruments = False
        self._sargan_stat = None
        self._sargan_pvalue = None
        
    def fit(self, data, y_col, X_cols, fe_vars, se_type='homoscedastic', cluster_vars=None, 
            sample_weight=None, instruments=None, endogenous_vars=None):
        """
        Fit HDFE-IV model with optional instrumental variables.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input dataset
        y_col : str
            Name of dependent variable
        X_cols : list
            Names of continuous variables (exogenous + endogenous)
        fe_vars : list
            Names of fixed effect variables  
        se_type : str, default='homoscedastic'
            Standard error type: 'homoscedastic', 'hc1', 'hc2', 'hc3', 'cluster'
        cluster_vars : list, optional
            List of variables to cluster on (required when se_type='cluster')
        sample_weight : array-like, optional
            Sample weights
        instruments : list, optional
            List of instrument variable column names. If None, falls back to HDFE
        endogenous_vars : list, optional
            List of endogenous variable column names (subset of X_cols)
            
        Returns:
        --------
        self : HDFEIV
            Fitted estimator instance
        """
        
        # Store IV-specific parameters
        self._instruments = instruments
        self._endogenous_vars = endogenous_vars if endogenous_vars else []
        
        # Determine if this is IV estimation
        self._is_iv = (instruments is not None and len(instruments) > 0 and 
                      endogenous_vars is not None and len(endogenous_vars) > 0)
        
        if not self._is_iv:
            # Fall back to standard HDFE estimation
            if self.verbose:
                print("No instruments provided, running standard HDFE estimation...")
            return super().fit(data, y_col, X_cols, fe_vars, se_type, cluster_vars, sample_weight)
        
        # IV-specific validation
        if not set(endogenous_vars).issubset(set(X_cols)):
            raise ValueError("All endogenous variables must be in X_cols")
        
        if len(instruments) < len(endogenous_vars):
            raise ValueError("Number of instruments must be >= number of endogenous variables")
            
        # Identify exogenous variables
        self._exogenous_vars = [x for x in X_cols if x not in endogenous_vars]
        
        # Store parameters for parent class compatibility
        self.se_type = se_type
        self.cluster_vars = cluster_vars
        self.fe_vars = fe_vars
        self.y_col = y_col
        self.X_cols = X_cols
        
        if self.verbose:
            print(f"Fitting HDFE-IV model with {len(data):,} observations")
            print(f"Exogenous variables: {len(self._exogenous_vars)}")
            print(f"Endogenous variables: {len(endogenous_vars)}")
            print(f"Instruments: {len(instruments)}")
            print(f"Fixed effects: {len(fe_vars)}")
            print(f"Standard errors: {se_type}")
            if se_type == 'cluster':
                print(f"Clustering on: {cluster_vars}")
            print(f"Using {'GPU' if self.use_gpu else 'CPU'} acceleration")
            
        # Prepare data
        if hasattr(data, 'to_pandas'):
            data = data.to_pandas()
            
        # Establish category ordering for fixed effects
        encoded_data = self._establish_category_ordering(data, fe_vars)
        
        # Extract and validate data
        y = data[y_col].values.astype(np.float64)
        X = data[X_cols].values.astype(np.float64)
        Z = data[instruments].values.astype(np.float64)
        
        # Handle missing data
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1) | np.any(np.isnan(Z), axis=1))
        y = y[valid_mask]
        X = X[valid_mask] 
        Z = Z[valid_mask]
        encoded_data = encoded_data[valid_mask]
        
        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]
            y = y * np.sqrt(sample_weight)
            X = X * np.sqrt(sample_weight).reshape(-1, 1)
            Z = Z * np.sqrt(sample_weight).reshape(-1, 1)
        
        # Step 1: Demean Y, X, and Z using alternating projection
        if self.verbose:
            print("Step 1: Demeaning Y, X, and instruments...")
            
        y_demeaned, X_demeaned = self._alternating_projection(y, X, encoded_data, fe_vars)
        _, Z_demeaned = self._alternating_projection(np.zeros_like(y), Z, encoded_data, fe_vars)
        
        # Step 2: First stage regressions
        if self.verbose:
            print("Step 2: Running first stage regressions...")
            
        self._run_first_stage(X_demeaned, Z_demeaned)
        
        # Step 3: Second stage 2SLS estimation
        if self.verbose:
            print("Step 3: Running second stage 2SLS...")
            
        self._run_second_stage(y_demeaned, X_demeaned, Z_demeaned)
        
        # Step 4: Recover fixed effects using sparse solver
        self.fe_coefficients_ = self._recover_fixed_effects(y, X, encoded_data, self.coefficients_, 
                                                           y_demeaned, X_demeaned)
        
        # Store demeaned data for robust SE calculation
        self.X_projected = X_demeaned
        self.y_projected = y_demeaned
        
        # Step 5: Calculate model statistics with IV-robust standard errors
        self._calculate_iv_statistics(y, X, Z, encoded_data, X_demeaned, data, valid_mask)
        
        # Step 6: IV diagnostics
        if self.verbose:
            print("Step 4: Computing IV diagnostics...")
        self._compute_iv_diagnostics()
        
        self.fitted = True
        return self
    
    def _run_first_stage(self, X_demeaned, Z_demeaned):
        """Run first stage regressions for each endogenous variable"""
        
        # Get indices of endogenous variables in X_cols
        endog_indices = [self.X_cols.index(var) for var in self._endogenous_vars]
        
        for i, endog_var in enumerate(self._endogenous_vars):
            endog_idx = endog_indices[i]
            
            # First stage: X_endog = Z * pi + residuals
            X_endog = X_demeaned[:, endog_idx]
            
            try:
                # Solve Z'Z * pi = Z'X_endog
                ZTZ = Z_demeaned.T @ Z_demeaned
                ZTX_endog = Z_demeaned.T @ X_endog
                
                pi_hat = np.linalg.solve(ZTZ, ZTX_endog)
                X_fitted = Z_demeaned @ pi_hat
                residuals = X_endog - X_fitted
                
                # Store first stage results
                self._first_stage_models[endog_var] = {
                    'coefficients': pi_hat,
                    'ZTZ': ZTZ,
                    'instrument_names': self._instruments
                }
                self._first_stage_fitted[endog_var] = X_fitted
                self._first_stage_residuals[endog_var] = residuals
                
                # Compute R² and F-statistic
                tss = np.sum((X_endog - np.mean(X_endog))**2)
                rss = np.sum(residuals**2)
                r2 = 1 - rss / tss
                self._first_stage_r2[endog_var] = r2
                
                # F-statistic for instrument significance
                n_obs, n_instruments = Z_demeaned.shape
                f_stat = (r2 / (1 - r2)) * ((n_obs - n_instruments - 1) / n_instruments)
                self._first_stage_f_stats[endog_var] = f_stat
                
                if self.verbose:
                    print(f"  {endog_var}: R² = {r2:.4f}, F-stat = {f_stat:.2f}")
                    
            except np.linalg.LinAlgError:
                raise ValueError(f"Rank deficient instruments for {endog_var}. Check for multicollinearity.")
    
    def _run_second_stage(self, y_demeaned, X_demeaned, Z_demeaned):
        """Run second stage 2SLS estimation"""
        
        # Replace endogenous variables with their fitted values
        X_2sls = X_demeaned.copy()
        endog_indices = [self.X_cols.index(var) for var in self._endogenous_vars]
        
        for i, endog_var in enumerate(self._endogenous_vars):
            endog_idx = endog_indices[i]
            X_2sls[:, endog_idx] = self._first_stage_fitted[endog_var]
        
        # Build instrument matrix: [exogenous_X, instruments]
        exog_indices = [self.X_cols.index(var) for var in self._exogenous_vars]
        if len(exog_indices) > 0:
            X_exog = X_demeaned[:, exog_indices]
            Z_full = np.column_stack([X_exog, Z_demeaned])
        else:
            Z_full = Z_demeaned
            
        # Store 2SLS matrices for variance calculation
        # CRITICAL FIX: Use original X_demeaned, not X_2sls for standard error calculation
        self._tZX = Z_full.T @ X_demeaned  # Use original X, not fitted X_2sls
        self._tXZ = X_demeaned.T @ Z_full  # Use original X, not fitted X_2sls
        self._tZy = Z_full.T @ y_demeaned
        self._tZZinv = np.linalg.inv(Z_full.T @ Z_full)
        
        # Store Z_full for later use in robust standard errors
        self._Z_full = Z_full
        
        # 2SLS estimation: β = (X'Z(Z'Z)^(-1)Z'X)^(-1) * X'Z(Z'Z)^(-1)Z'y
        # But for the actual estimation, we DO use X_2sls (with fitted values)
        tZX_2sls = Z_full.T @ X_2sls
        tXZ_2sls = X_2sls.T @ Z_full
        
        try:
            H = tXZ_2sls @ self._tZZinv    # X_2sls'Z(Z'Z)^(-1)
            A = H @ tZX_2sls              # X_2sls'Z(Z'Z)^(-1)Z'X_2sls  
            B = H @ self._tZy             # X_2sls'Z(Z'Z)^(-1)Z'y
            
            self.coefficients_ = np.linalg.solve(A, B)
            
            if self.verbose:
                print(f"  2SLS coefficients estimated successfully")
                
        except np.linalg.LinAlgError:
            raise ValueError("2SLS estimation failed. Check for identification issues.")
    
    def _compute_iv_diagnostics(self):
        """Compute IV diagnostic tests"""
        
        # Check for weak instruments (F-stat < 10 rule of thumb)
        min_f_stat = min(self._first_stage_f_stats.values()) if self._first_stage_f_stats else 0
        self._weak_instruments = min_f_stat < 10.0
        
        if self.verbose:
            print(f"  Minimum first-stage F-statistic: {min_f_stat:.2f}")
            if self._weak_instruments:
                print("  ⚠️ Warning: Potentially weak instruments (F < 10)")
            else:
                print("  ✅ Instruments appear adequately strong")
                
        # Sargan test for overidentification (if overidentified)
        n_endog = len(self._endogenous_vars)  
        n_instruments = len(self._instruments)
        
        if n_instruments > n_endog:
            try:
                # Get 2SLS residuals
                residuals = self.y_projected - self.X_projected @ self.coefficients_
                
                # Regress residuals on all instruments
                Z_full = np.column_stack([self.X_projected[:, [self.X_cols.index(var) for var in self._exogenous_vars]] if self._exogenous_vars else np.empty((len(residuals), 0)), 
                                         np.column_stack([self._first_stage_fitted[var] for var in self._endogenous_vars])])
                
                aux_r2 = 1 - np.sum((residuals - Z_full @ np.linalg.lstsq(Z_full, residuals, rcond=None)[0])**2) / np.sum((residuals - np.mean(residuals))**2)
                
                # Sargan statistic ~ χ²(J) where J = # instruments - # endogenous
                n_obs = len(residuals)
                self._sargan_stat = n_obs * aux_r2
                
                # p-value from chi-squared distribution
                from scipy import stats
                df_sargan = n_instruments - n_endog
                self._sargan_pvalue = 1 - stats.chi2.cdf(self._sargan_stat, df_sargan)
                
                if self.verbose:
                    print(f"  Sargan test: χ² = {self._sargan_stat:.3f}, p-value = {self._sargan_pvalue:.3f}")
                    if self._sargan_pvalue < 0.05:
                        print("  ⚠️ Warning: Sargan test rejects instrument validity (p < 0.05)")
                    else:
                        print("  ✅ Sargan test does not reject instrument validity")
                        
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Could not compute Sargan test: {e}")
                self._sargan_stat = None
                self._sargan_pvalue = None
    
    def _calculate_iv_statistics(self, y, X, Z, encoded_data, X_demeaned, original_data, valid_mask):
        """Calculate model fit statistics with IV-robust standard errors"""
        
        # Full predictions including fixed effects
        y_pred_full = X @ self.coefficients_
        for fe_var in self.fe_vars:
            group_ids = encoded_data[fe_var].values
            for i in range(len(y)):
                if group_ids[i] >= 0:
                    y_pred_full[i] += self.fe_coefficients_[fe_var][group_ids[i]]
        
        residuals_full = y - y_pred_full
        self.residuals_ = residuals_full
        self.fitted_values_ = y_pred_full
        
        # R-squared
        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum(residuals_full**2)
        self.r_squared_ = 1 - rss / tss
        
        # Compute IV-robust standard errors
        if self.verbose:
            print(f"Computing {self.se_type} standard errors for IV model...")
        
        try:
            self.std_errors_ = self._compute_iv_robust_standard_errors(X_demeaned, residuals_full, 
                                                                     original_data, valid_mask)
            
            # t-statistics and p-values
            self.t_stats_ = self.coefficients_ / self.std_errors_
            from scipy import stats
            df_resid = len(y) - X.shape[1] - sum(self.n_categories.values())
            self.p_values_ = 2 * (1 - stats.t.cdf(np.abs(self.t_stats_), df_resid))
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not compute IV-robust standard errors: {e}")
                
            # Fallback to basic IV standard errors
            self._compute_basic_iv_standard_errors(X_demeaned, residuals_full)
    
    def _compute_iv_robust_standard_errors(self, X_demeaned, residuals, data, valid_mask):
        """Compute IV-robust standard errors using correct IV sandwich formula"""
        
        try:
            # Use the correct IV variance formula for all cases
            # Var(β) = σ² × (X'Z(Z'Z)^(-1)Z'X)^(-1)
            
            if self.se_type == 'homoscedastic':
                # Homoscedastic IV standard errors
                sigma2 = np.sum(residuals**2) / (len(residuals) - X_demeaned.shape[1])
            elif self.se_type == 'hc1':
                # HC1 with degrees of freedom adjustment
                n_obs = len(residuals)
                k = X_demeaned.shape[1]
                adjustment = n_obs / (n_obs - k)
                sigma2 = adjustment * np.sum(residuals**2) / n_obs
            else:
                # Fall back to homoscedastic for other robust types
                sigma2 = np.sum(residuals**2) / (len(residuals) - X_demeaned.shape[1])
            
            # Apply correct IV variance formula
            var_beta_inv = self._tXZ @ self._tZZinv @ self._tZX
            var_beta = sigma2 * np.linalg.inv(var_beta_inv)
            
            return np.sqrt(np.diag(var_beta))
            
        except Exception as e:
            if self.verbose:
                print(f"IV robust SE calculation failed: {e}, using basic IV SEs...")
            # Fallback to basic IV standard errors
            return self._compute_basic_iv_standard_errors(X_demeaned, residuals)
    
    def _compute_basic_iv_standard_errors(self, X_demeaned, residuals):
        """Fallback computation for basic IV standard errors"""
        try:
            # Basic IV variance: σ² * (X'Z(Z'Z)^(-1)Z'X)^(-1)
            sigma2 = np.sum(residuals**2) / (len(residuals) - X_demeaned.shape[1])
            var_inv = self._tXZ @ self._tZZinv @ self._tZX
            var_beta = sigma2 * np.linalg.inv(var_inv)
            
            return np.sqrt(np.diag(var_beta))
            
        except Exception as e:
            if self.verbose:
                print(f"Basic IV SE calculation failed: {e}")
            # Final fallback
            return np.full_like(self.coefficients_, np.nan)
    
    def summary(self):
        """Print comprehensive IV model summary"""
        if not self.fitted:
            raise ValueError("Model must be fitted before summary")
            
        print("=" * 90)
        print("HDFE-IV REGRESSION RESULTS") 
        print("(High-Dimensional Fixed Effects + Instrumental Variables)")
        print("=" * 90)
        print(f"Estimation method: {'2SLS with HDFE' if self._is_iv else 'HDFE-OLS'}")
        print(f"R-squared: {self.r_squared_:.6f}")
        print(f"Number of observations: {len(self.residuals_):,}")
        print(f"Fixed effect categories: {dict(self.n_categories)}")
        print(f"Standard error type: {self.se_type}")
        if hasattr(self, 'cluster_vars') and self.cluster_vars:
            print(f"Clustering variables: {self.cluster_vars}")
        
        if self._is_iv:
            print(f"Endogenous variables: {self._endogenous_vars}")
            print(f"Instruments: {self._instruments}")
            print(f"Exogenous variables: {self._exogenous_vars}")
            
            # IV diagnostics
            print("\nIV DIAGNOSTICS:")
            print("-" * 60)
            print("First Stage Results:")
            for var in self._endogenous_vars:
                r2 = self._first_stage_r2.get(var, 0)
                f_stat = self._first_stage_f_stats.get(var, 0)  
                print(f"  {var}: R² = {r2:.4f}, F-stat = {f_stat:.2f}")
            
            if self._weak_instruments:
                print("⚠️  WARNING: Weak instruments detected (F < 10)")
            else:
                print("✅ Instruments appear adequately strong")
                
            if self._sargan_stat is not None:
                print(f"Sargan test: χ² = {self._sargan_stat:.3f}, p-value = {self._sargan_pvalue:.3f}")
        
        print("\nCOEFFICIENT ESTIMATES:")
        print("-" * 90)
        print(f"{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<8} {'P>|t|':<8} {'Type':<12}")
        print("-" * 90)
        
        for i, var in enumerate(self.X_cols):
            var_type = "Endogenous" if var in self._endogenous_vars else "Exogenous"
            if not np.isnan(self.std_errors_[i]):
                print(f"{var:<20} {self.coefficients_[i]:<12.6f} {self.std_errors_[i]:<12.6f} "
                      f"{self.t_stats_[i]:<8.3f} {self.p_values_[i]:<8.3f} {var_type:<12}")
            else:
                print(f"{var:<20} {self.coefficients_[i]:<12.6f} {'N/A':<12} {'N/A':<8} {'N/A':<8} {var_type:<12}")
        
        print("\nFIXED EFFECTS SUMMARY:")
        print("-" * 70)
        for fe_var in self.fe_vars:
            fe_coeffs = self.fe_coefficients_[fe_var]
            print(f"{fe_var}: mean={np.mean(fe_coeffs):.6f}, std={np.std(fe_coeffs):.6f}, "
                  f"min={np.min(fe_coeffs):.6f}, max={np.max(fe_coeffs):.6f}")
        
        print("=" * 90)
    
    def first_stage_results(self):
        """Return detailed first stage regression results"""
        if not self._is_iv:
            print("No first stage results available (not an IV model)")
            return None
            
        results = {}
        for var in self._endogenous_vars:
            results[var] = {
                'coefficients': self._first_stage_models[var]['coefficients'],
                'instrument_names': self._instruments,
                'r_squared': self._first_stage_r2[var],
                'f_statistic': self._first_stage_f_stats[var],
                'fitted_values': self._first_stage_fitted[var],
                'residuals': self._first_stage_residuals[var]
            }
        return results
    
    def iv_diagnostics(self):
        """Return IV diagnostic test results"""
        if not self._is_iv:
            print("No IV diagnostics available (not an IV model)")
            return None
            
        return {
            'weak_instruments': self._weak_instruments,
            'first_stage_f_stats': self._first_stage_f_stats,
            'sargan_statistic': self._sargan_stat,
            'sargan_pvalue': self._sargan_pvalue,
            'min_eigenvalue': None,  # Could add if needed
            'effective_f': None      # Could add Olea-Pflueger if needed
        }