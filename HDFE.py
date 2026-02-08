# HDFE.py
# High-Dimensional Fixed Effects Estimators (CPU and GPU versions)
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings


class HDFE:
    """
    High-Dimensional Fixed Effects estimator using alternating projection
    demeaning with sparse-system fixed effects recovery.
    """

    def __init__(self, max_iter=5000, tolerance=1e-8, acceleration='gk',
                 use_gpu=False, verbose=False, convergence_check_interval=5):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.acceleration = acceleration
        self.verbose = verbose
        self.convergence_check_interval = convergence_check_interval

        # GPU setup — only attempt import when explicitly requested
        if use_gpu:
            try:
                import cupy as cp
                self.use_gpu = cp.cuda.is_available()
                if self.verbose:
                    print("GPU detected and will be used" if self.use_gpu
                          else "GPU not available, falling back to CPU")
            except ImportError:
                self.use_gpu = False
                if self.verbose:
                    print("CuPy not installed, falling back to CPU")
        else:
            self.use_gpu = False

        # State
        self.fitted = False
        self.fe_vars = []
        self.category_orders_ = {}
        self.category_to_index_ = {}
        self.n_categories = {}
        self._rank_diagnostics = {}

    # ── Category encoding (vectorised) ──────────────────────────────────────

    def _establish_category_ordering(self, data, fe_vars):
        for fe_var in fe_vars:
            if fe_var not in self.category_orders_:
                unique_cats_str = data[fe_var].dropna().astype(str).unique()
                try:
                    unique_cats_numeric = sorted([int(c) for c in unique_cats_str])
                    unique_cats = [str(c) for c in unique_cats_numeric]
                except ValueError:
                    unique_cats = sorted(unique_cats_str)
                self.category_orders_[fe_var] = unique_cats
                self.category_to_index_[fe_var] = {
                    cat: idx for idx, cat in enumerate(unique_cats)}
                self.n_categories[fe_var] = len(unique_cats)
        return self._encode_with_consistent_ordering(data, fe_vars)

    def _encode_with_consistent_ordering(self, data, fe_vars):
        encoded_data = data.copy()
        for fe_var in fe_vars:
            cat_map = self.category_to_index_[fe_var]
            encoded_data[fe_var] = (data[fe_var].astype(str)
                                    .map(cat_map).fillna(-1)
                                    .astype(int).values)
        return encoded_data

    # ── Precomputed group info ──────────────────────────────────────────────

    def _precompute_group_info_cpu(self, group_indices, n_groups):
        """Precompute per-FE constants once before the iteration loop."""
        valid = group_indices >= 0
        safe_idx = np.where(valid, group_indices, 0)
        group_counts = np.bincount(
            safe_idx, weights=valid.astype(np.float64),
            minlength=n_groups)
        inv_counts = np.zeros(n_groups, dtype=np.float64)
        nz = group_counts > 0
        inv_counts[nz] = 1.0 / group_counts[nz]
        return {
            'valid': valid,
            'safe_idx': safe_idx,
            'valid_idx': np.where(valid)[0],
            'safe_valid': safe_idx[valid],
            'inv_counts': inv_counts,
            'n_groups': n_groups,
        }

    def _precompute_group_info_gpu(self, group_indices_gpu, n_groups):
        """Precompute per-FE constants once before the iteration loop (GPU)."""
        import cupy as cp
        valid = group_indices_gpu >= 0
        safe_idx = cp.where(valid, group_indices_gpu, cp.int32(0))
        group_counts = cp.bincount(
            safe_idx, weights=valid.astype(cp.float64),
            minlength=n_groups)
        inv_counts = cp.zeros(n_groups, dtype=cp.float64)
        nz = group_counts > 0
        inv_counts[nz] = 1.0 / group_counts[nz]
        return {
            'valid': valid,
            'safe_idx': safe_idx,
            'valid_idx': cp.where(valid)[0],
            'safe_valid': safe_idx[valid],
            'inv_counts': inv_counts,
            'n_groups': n_groups,
        }

    # ── Batch group demeaning ───────────────────────────────────────────────

    def _cpu_demean_batch(self, y, X, ginfo):
        """Demean y (1-D) and X (2-D) by a single FE group. Returns new arrays."""
        valid = ginfo['valid']
        safe_idx = ginfo['safe_idx']
        safe_valid = ginfo['safe_valid']
        valid_idx = ginfo['valid_idx']
        inv_counts = ginfo['inv_counts']
        ng = ginfo['n_groups']

        # y
        y_sums = np.bincount(safe_idx,
                             weights=np.where(valid, y, 0.0),
                             minlength=ng).astype(np.float64)
        y_means = y_sums * inv_counts
        y_new = y.copy()
        y_new[valid_idx] -= y_means[safe_valid]

        # X — all columns at once via loop over bincount (avoids 2-D scatter)
        n_cols = X.shape[1]
        X_new = X.copy()
        X_valid = X[valid_idx]
        for j in range(n_cols):
            col_sums = np.bincount(safe_valid,
                                   weights=X_valid[:, j],
                                   minlength=ng).astype(np.float64)
            X_new[valid_idx, j] -= (col_sums * inv_counts)[safe_valid]

        return y_new, X_new

    def _gpu_demean_batch(self, y, X, ginfo):
        """Demean y (1-D) and X (2-D) by a single FE group (GPU).
        Uses scatter_add for all columns at once. Returns new arrays."""
        import cupy as cp
        valid = ginfo['valid']
        safe_idx = ginfo['safe_idx']
        safe_valid = ginfo['safe_valid']
        valid_idx = ginfo['valid_idx']
        inv_counts = ginfo['inv_counts']
        ng = ginfo['n_groups']

        # y
        y_sums = cp.bincount(safe_idx,
                             weights=cp.where(valid, y, 0.0),
                             minlength=ng).astype(cp.float64)
        y_means = y_sums * inv_counts
        y_new = y.copy()
        y_new[valid_idx] -= y_means[safe_valid]

        # X — all columns via scatter_add
        n_cols = X.shape[1]
        group_sums = cp.zeros((ng, n_cols), dtype=cp.float64)
        cp.add.at(group_sums, safe_valid, X[valid_idx])
        group_means = group_sums * inv_counts.reshape(-1, 1)
        X_new = X.copy()
        X_new[valid_idx] -= group_means[safe_valid]

        return y_new, X_new

    # ── Alternating projection ──────────────────────────────────────────────

    def _alternating_projection(self, y, X, encoded_data, fe_vars):
        if self.verbose:
            print("Starting alternating projection algorithm...")
        if self.use_gpu:
            try:
                import cupy as cp
                free_mem = cp.cuda.Device().mem_info[0]
                required_mem = int((y.nbytes + X.nbytes) * 3)
                if required_mem > free_mem * 0.85:
                    raise MemoryError(
                        f"Insufficient GPU memory ({free_mem/1e9:.1f}GB free, "
                        f"~{required_mem/1e9:.1f}GB needed)")
                y_proj = cp.asarray(y, dtype=cp.float64)
                X_proj = cp.asarray(X, dtype=cp.float64)
                group_ids_list = [
                    cp.asarray(encoded_data[fv].values, dtype=cp.int32)
                    for fv in fe_vars]
                # Precompute group info once
                group_infos = [
                    self._precompute_group_info_gpu(gids, self.n_categories[fv])
                    for gids, fv in zip(group_ids_list, fe_vars)]
                demean_batch = self._gpu_demean_batch
                backend_name = "GPU"
            except Exception as e:
                if self.verbose:
                    print(f"GPU init failed: {e}, falling back to CPU")
                self.use_gpu = False

        if not self.use_gpu:
            y_proj = y.copy()
            X_proj = X.copy()
            group_ids_list = [encoded_data[fv].values for fv in fe_vars]
            # Precompute group info once
            group_infos = [
                self._precompute_group_info_cpu(gids, self.n_categories[fv])
                for gids, fv in zip(group_ids_list, fe_vars)]
            demean_batch = self._cpu_demean_batch
            backend_name = "CPU"

        if self.verbose:
            print(f"Using {backend_name}, acceleration={self.acceleration}")

        if self.acceleration == 'gk':
            return self._alternating_projection_gk(
                y_proj, X_proj, group_infos, demean_batch)
        else:
            return self._alternating_projection_basic(
                y_proj, X_proj, group_infos, demean_batch)

    def _alternating_projection_basic(self, y_proj, X_proj, group_infos,
                                       demean_batch):
        check_iv = self.convergence_check_interval
        for iteration in range(self.max_iter):
            # Only snapshot + check periodically
            do_check = (iteration % check_iv == 0) or (iteration == self.max_iter - 1)
            if do_check:
                y_old = y_proj.copy()
                X_old = X_proj.copy()

            # Batch demean y + all X columns per FE
            for ginfo in group_infos:
                y_proj, X_proj = demean_batch(y_proj, X_proj, ginfo)

            if do_check:
                if self.use_gpu:
                    import cupy as cp
                    y_chg = float(cp.mean((y_proj - y_old)**2))
                    X_chg = float(cp.mean((X_proj - X_old)**2))
                else:
                    y_chg = np.mean((y_proj - y_old)**2)
                    X_chg = np.mean((X_proj - X_old)**2)
                if self.verbose and iteration % 200 == 0:
                    print(f"  iter {iteration}: y_chg={y_chg:.2e}, X_chg={X_chg:.2e}")
                if y_chg < self.tolerance and X_chg < self.tolerance:
                    if self.verbose:
                        print(f"  Converged after {iteration+1} iterations")
                    break
        else:
            if self.verbose:
                print(f"  Warning: max iterations ({self.max_iter}) reached")
        if self.use_gpu:
            import cupy as cp
            return cp.asnumpy(y_proj), cp.asnumpy(X_proj)
        return y_proj, X_proj

    def _alternating_projection_gk(self, y_proj, X_proj, group_infos,
                                    demean_batch):
        check_iv = self.convergence_check_interval
        y_hist, X_hist = [], []
        for iteration in range(self.max_iter):
            do_check = (iteration % check_iv == 0) or (iteration == self.max_iter - 1)
            if do_check:
                y_old = y_proj.copy()
                X_old = X_proj.copy()

            # Batch demean y + all X columns per FE
            for ginfo in group_infos:
                y_proj, X_proj = demean_batch(y_proj, X_proj, ginfo)

            # Convergence check only periodically
            if do_check:
                if self.use_gpu:
                    import cupy as cp
                    y_chg = float(cp.mean((y_proj - y_old)**2))
                    X_chg = float(cp.mean((X_proj - X_old)**2))
                else:
                    y_chg = np.mean((y_proj - y_old)**2)
                    X_chg = np.mean((X_proj - X_old)**2)

            # GK acceleration
            if len(y_hist) >= 2:
                y_proj = self._apply_gk_acceleration(
                    y_proj, y_hist[-1], y_hist[-2])
                X_proj = self._apply_gk_acceleration(
                    X_proj, X_hist[-1], X_hist[-2])
            y_hist.append(y_proj.copy())
            X_hist.append(X_proj.copy())
            if len(y_hist) > 3:
                y_hist.pop(0); X_hist.pop(0)

            if do_check:
                if self.verbose and iteration % 200 == 0:
                    print(f"  iter {iteration}: y_chg={y_chg:.2e}, X_chg={X_chg:.2e}")
                if y_chg < self.tolerance and X_chg < self.tolerance:
                    if self.verbose:
                        print(f"  Converged after {iteration+1} iterations")
                    break
        else:
            if self.verbose:
                print(f"  Warning: max iterations ({self.max_iter}) reached")
        if self.use_gpu:
            import cupy as cp
            return cp.asnumpy(y_proj), cp.asnumpy(X_proj)
        return y_proj, X_proj

    def _apply_gk_acceleration(self, current, prev1, prev2):
        if self.use_gpu:
            import cupy as cp
            d1 = current - prev1; d2 = prev1 - prev2
            denom = cp.sum(d2 * d2)
            if abs(float(denom)) > self.tolerance:
                a = float(cp.sum(d1 * d2) / denom)
                a = max(0, min(a, 1))
                return current + a * d1
        else:
            d1 = current - prev1; d2 = prev1 - prev2
            denom = np.sum(d2 * d2)
            if abs(denom) > self.tolerance:
                a = np.sum(d1 * d2) / denom
                a = max(0, min(a, 1))
                return current + a * d1
        return current

    # ── Sparse dummy matrix & FE recovery ───────────────────────────────────

    def _build_dummy_matrix(self, encoded_data, fe_vars):
        """Build sparse dummy matrix for FE recovery (vectorised)."""
        if len(fe_vars) == 0:
            raise ValueError("fe_vars must be non-empty for dummy matrix construction")
        n_obs = len(encoded_data)
        total_cols = self.n_categories[fe_vars[0]]
        for fv in fe_vars[1:]:
            total_cols += self.n_categories[fv] - 1

        all_rows, all_cols = [], []
        fe_col_info = {}
        cur = 0
        for fe_idx, fv in enumerate(fe_vars):
            nc = self.n_categories[fv]
            gids = encoded_data[fv].values
            if fe_idx == 0:
                fe_col_info[fv] = {
                    'start_col': cur, 'end_col': cur + nc,
                    'n_categories': nc, 'dropped_category': None}
                v = gids >= 0
                all_rows.append(np.where(v)[0])
                all_cols.append(cur + gids[v])
                cur += nc
            else:
                fe_col_info[fv] = {
                    'start_col': cur, 'end_col': cur + nc - 1,
                    'n_categories': nc, 'dropped_category': 0}
                v = gids >= 1
                all_rows.append(np.where(v)[0])
                all_cols.append(cur + gids[v] - 1)
                cur += nc - 1

        ri = np.concatenate(all_rows)
        ci = np.concatenate(all_cols)
        D = csr_matrix((np.ones(len(ri), dtype=np.float64),
                        (ri, ci)), shape=(n_obs, total_cols))
        return D, fe_col_info

    def _recover_fixed_effects(self, y, X, encoded_data, beta,
                                y_projected, X_projected,
                                sample_weight=None,
                                singular_fallback='lsqr'):
        """Recover FE coefficients using sparse solver (weight-aware)."""
        if self.verbose:
            print("Recovering fixed effects using sparse solver...")
        if len(self.fe_vars) == 0:
            return {}

        D, fe_col_info = self._build_dummy_matrix(encoded_data, self.fe_vars)
        res_orig = y - X @ beta
        res_proj = y_projected - X_projected @ beta
        rhs = res_orig - res_proj

        # Undo sqrt-weighting so FE recovery is in the original scale
        if sample_weight is not None:
            sw = np.sqrt(sample_weight)
            sw_safe = np.where(sw > 0, sw, 1.0)
            rhs = rhs / sw_safe

        from scipy.sparse.linalg import lsqr

        # Prune columns with zero observations (empty categories)
        DtD = D.T @ D
        diag_DtD = np.array(DtD.diagonal()).ravel()
        empty_cols = np.where(diag_DtD == 0)[0]

        if len(empty_cols) > 0:
            empty_fe_info = {}
            for fv in self.fe_vars:
                info = fe_col_info[fv]
                fe_empty = [c for c in empty_cols
                            if info['start_col'] <= c < info['end_col']]
                if fe_empty:
                    empty_fe_info[fv] = len(fe_empty)

            self._rank_diagnostics['empty_fe_categories'] = empty_fe_info
            msg = (f"D'D has {len(empty_cols)} zero-diagonal entries "
                   f"(empty FE categories): {empty_fe_info}")
            if singular_fallback == 'raise':
                raise np.linalg.LinAlgError(
                    msg + ". Set singular_fallback='lsqr' to use "
                    "least-squares fallback.")

            if self.verbose:
                print(f"  ⚠️ {msg}, pruning before solve...")
            keep_cols = np.where(diag_DtD > 0)[0]
            D_pruned = D[:, keep_cols]
            DtD = D_pruned.T @ D_pruned
            Dtr = D_pruned.T @ rhs
        else:
            Dtr = D.T @ rhs
            keep_cols = None

        try:
            if self.use_gpu:
                try:
                    import cupy as cp
                    import cupyx.scipy.sparse as csp
                    import cupyx.scipy.sparse.linalg as cspl
                    alpha_solve = cp.asnumpy(cspl.spsolve(
                        csp.csr_matrix(DtD.astype(np.float64)),
                        cp.asarray(Dtr.astype(np.float64))))
                except Exception:
                    alpha_solve = spsolve(DtD, Dtr)
            else:
                alpha_solve = spsolve(DtD, Dtr)

            # spsolve silently returns NaN on singular matrices
            if not np.all(np.isfinite(alpha_solve)):
                msg = ("spsolve returned NaN — D'D is singular "
                       "(collinear fixed effects)")
                self._rank_diagnostics['spsolve_nan'] = True
                if singular_fallback == 'raise':
                    raise np.linalg.LinAlgError(
                        msg + ". Set singular_fallback='lsqr' to use "
                        "least-squares fallback.")
                if self.verbose:
                    print(f"  ⚠️ {msg}, falling back to lsqr...")
                if keep_cols is not None:
                    alpha_solve = lsqr(D_pruned, rhs)[0]
                else:
                    alpha_solve = lsqr(D, rhs)[0]

        except np.linalg.LinAlgError:
            raise  # re-raise our own LinAlgError
        except Exception:
            msg = "D'D solve failed"
            self._rank_diagnostics['solve_exception'] = True
            if singular_fallback == 'raise':
                raise np.linalg.LinAlgError(
                    msg + ". Set singular_fallback='lsqr' to use "
                    "least-squares fallback.")
            if self.verbose:
                print(f"  ⚠️ {msg}, falling back to lsqr...")
            if keep_cols is not None:
                alpha_solve = lsqr(D_pruned, rhs)[0]
            else:
                alpha_solve = lsqr(D, rhs)[0]

        # Re-insert zeros for pruned columns
        if keep_cols is not None:
            alpha = np.zeros(D.shape[1])
            alpha[keep_cols] = alpha_solve
        else:
            alpha = alpha_solve

        fe_coefficients = {}
        for fe_idx, fv in enumerate(self.fe_vars):
            info = fe_col_info[fv]
            nc = info['n_categories']
            if fe_idx == 0:
                fe_coefficients[fv] = alpha[info['start_col']:info['end_col']]
            else:
                c = np.zeros(nc)
                c[1:] = alpha[info['start_col']:info['end_col']]
                fe_coefficients[fv] = c
        if self.verbose:
            for fv, c in fe_coefficients.items():
                print(f"  {fv}: mean={np.mean(c):.6f}, std={np.std(c):.6f}")
        return fe_coefficients

    # ── Main fit ────────────────────────────────────────────────────────────

    def fit(self, data, y_col, X_cols, fe_vars,
            se_type='homoscedastic', cluster_vars=None, sample_weight=None,
            singular_fallback='lsqr'):
        if singular_fallback not in ('lsqr', 'raise'):
            raise ValueError("singular_fallback must be 'lsqr' or 'raise'")
        self._rank_diagnostics = {}
        valid_se_types = ['homoscedastic', 'hc1', 'hc2', 'hc3', 'cluster']
        if se_type not in valid_se_types:
            raise ValueError(f"se_type must be one of {valid_se_types}")
        if se_type == 'cluster' and cluster_vars is None:
            raise ValueError("cluster_vars required when se_type='cluster'")
        if se_type == 'cluster' and not isinstance(cluster_vars, list):
            cluster_vars = [cluster_vars]
        if not fe_vars:
            raise ValueError("fe_vars must be non-empty for HDFE estimation")

        self.se_type = se_type
        self.cluster_vars = cluster_vars
        self.fe_vars = fe_vars
        self.y_col = y_col
        self.X_cols = X_cols

        if self.verbose:
            print(f"Fitting HDFE: {len(data):,} obs, {len(X_cols)} vars, "
                  f"{len(fe_vars)} FEs, SE={se_type}, "
                  f"{'GPU' if self.use_gpu else 'CPU'}")

        if hasattr(data, 'to_pandas'):
            data = data.to_pandas()
        encoded_data = self._establish_category_ordering(data, fe_vars)

        y = data[y_col].values.astype(np.float64)
        X = data[X_cols].values.astype(np.float64)
        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
        for fv in fe_vars:
            valid_mask &= data[fv].notna().values
        y = y[valid_mask]; X = X[valid_mask]
        encoded_data = encoded_data[valid_mask]

        self._sample_weight = None
        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]
            self._sample_weight = sample_weight
            y = y * np.sqrt(sample_weight)
            X = X * np.sqrt(sample_weight).reshape(-1, 1)

        y_proj, X_proj = self._alternating_projection(y, X, encoded_data, fe_vars)
        try:
            self.coefficients_ = np.linalg.solve(
                X_proj.T @ X_proj, X_proj.T @ y_proj)
        except np.linalg.LinAlgError:
            self._rank_diagnostics['XtX_singular'] = True
            warnings.warn(
                "X'X is singular after projection — using least-squares "
                "solution. Check for collinear regressors.", stacklevel=2)
            self.coefficients_, _, _, _ = np.linalg.lstsq(
                X_proj, y_proj, rcond=None)
        self.fe_coefficients_ = self._recover_fixed_effects(
            y, X, encoded_data, self.coefficients_, y_proj, X_proj,
            sample_weight=self._sample_weight,
            singular_fallback=singular_fallback)
        self.X_projected = X_proj
        self.y_projected = y_proj
        self._calculate_statistics(
            y, X, encoded_data, X_proj, data, valid_mask)
        self.fitted = True
        return self

    # ── Standard errors ─────────────────────────────────────────────────────

    def _compute_robust_standard_errors(self, X_proj, residuals, data, valid_mask):
        try:
            XtX_inv = np.linalg.inv(X_proj.T @ X_proj)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_proj.T @ X_proj)
        if self.se_type == 'homoscedastic':
            sigma2 = np.sum(residuals**2) / self._df_resid
            var_beta = sigma2 * XtX_inv
        elif self.se_type in ('hc1', 'hc2', 'hc3'):
            meat = self._compute_hc_matrix(X_proj, residuals, self.se_type)
            var_beta = XtX_inv @ meat @ XtX_inv
        elif self.se_type == 'cluster':
            meat = self._compute_multi_cluster_matrix(
                X_proj, residuals, data, valid_mask)
            var_beta = XtX_inv @ meat @ XtX_inv
        return np.sqrt(np.diag(var_beta))

    def _compute_hc_matrix(self, X, residuals, hc_type):
        n, k = X.shape
        if hc_type == 'hc1':
            w = (residuals**2) * n / (n - k)
        elif hc_type == 'hc2':
            h = np.sum(X * np.linalg.solve(X.T @ X, X.T).T, axis=1)
            w = (residuals**2) / (1 - h)
        elif hc_type == 'hc3':
            h = np.sum(X * np.linalg.solve(X.T @ X, X.T).T, axis=1)
            w = (residuals**2) / ((1 - h)**2)
        w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
        wX = X * np.sqrt(w).reshape(-1, 1)
        return wX.T @ wX

    def _compute_multi_cluster_matrix(self, X, residuals, data, valid_mask):
        if self.verbose:
            print("Computing cluster-robust SEs...")
        cdat = {}
        for cv in self.cluster_vars:
            cdat[cv] = data[cv].values[valid_mask]
        if len(self.cluster_vars) == 1:
            cg = cdat[self.cluster_vars[0]]
        else:
            cg = cdat[self.cluster_vars[0]].astype(str)
            for cv in self.cluster_vars[1:]:
                cg = cg + "_" + cdat[cv].astype(str)
        if self.use_gpu:
            try:
                return self._cluster_gpu(X, residuals, cg)
            except Exception:
                pass
        return self._cluster_cpu(X, residuals, cg)

    def _cluster_gpu(self, X, residuals, cg):
        import cupy as cp
        Xg = cp.asarray(X, dtype=cp.float64)
        rg = cp.asarray(residuals, dtype=cp.float64)
        uq = np.unique(cg); nc = len(uq); k = X.shape[1]
        c2i = {c: i for i, c in enumerate(uq)}
        ci = cp.asarray(np.array([c2i[c] for c in cg]), dtype=cp.int32)
        meat = cp.zeros((k, k), dtype=cp.float64)
        for j in range(nc):
            m = ci == j
            if cp.any(m):
                s = Xg[m].T @ rg[m]
                meat += cp.outer(s, s)
        out = cp.asnumpy(meat)
        del Xg, rg, ci, meat
        try: cp.get_default_memory_pool().free_all_blocks()
        except: pass
        return out

    def _cluster_cpu(self, X, residuals, cg):
        meat = np.zeros((X.shape[1], X.shape[1]))
        for c in np.unique(cg):
            m = cg == c
            s = (X[m].T @ residuals[m]).reshape(-1, 1)
            meat += s @ s.T
        return meat

    # ── Statistics ──────────────────────────────────────────────────────────

    def _calculate_statistics(self, y, X, encoded_data, X_proj,
                               original_data, valid_mask):
        y_pred = X @ self.coefficients_
        for fv in self.fe_vars:
            gids = encoded_data[fv].values
            v = gids >= 0
            fe_contrib = self.fe_coefficients_[fv][gids[v]]
            if self._sample_weight is not None:
                fe_contrib = fe_contrib * np.sqrt(self._sample_weight[v])
            y_pred[v] += fe_contrib
        resid = y - y_pred
        self.residuals_ = resid
        self.fitted_values_ = y_pred
        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum(resid**2)
        self.r_squared_ = 1 - rss / tss

        # dof
        if len(self.fe_vars) > 0:
            df_abs = self.n_categories[self.fe_vars[0]]
            for fv in self.fe_vars[1:]:
                df_abs += self.n_categories[fv] - 1
        else:
            df_abs = 0
        self._df_resid = max(len(y) - X.shape[1] - df_abs, 1)
        if self._df_resid <= X.shape[1]:
            warnings.warn(
                f"Very low residual dof ({self._df_resid}). "
                f"N={len(y)}, k={X.shape[1]}, absorbed={df_abs}. "
                "SEs may be unreliable.", stacklevel=2)

        try:
            self.std_errors_ = self._compute_robust_standard_errors(
                X_proj, resid, original_data, valid_mask)
            self.t_stats_ = self.coefficients_ / self.std_errors_
            self.p_values_ = 2 * (1 - stats.t.cdf(
                np.abs(self.t_stats_), self._df_resid))
        except Exception:
            mse = rss / self._df_resid
            try:
                inv = np.linalg.inv(X_proj.T @ X_proj)
                self.std_errors_ = np.sqrt(np.diag(mse * inv))
                self.t_stats_ = self.coefficients_ / self.std_errors_
                self.p_values_ = 2 * (1 - stats.t.cdf(
                    np.abs(self.t_stats_), self._df_resid))
            except Exception:
                self.std_errors_ = np.full_like(self.coefficients_, np.nan)
                self.t_stats_ = np.full_like(self.coefficients_, np.nan)
                self.p_values_ = np.full_like(self.coefficients_, np.nan)

    # ── Rank diagnostics ────────────────────────────────────────────────────

    def rank_diagnostics(self):
        """Return rank diagnostic information from the last fit."""
        if not self.fitted:
            raise ValueError("Model must be fitted before rank diagnostics")
        return dict(self._rank_diagnostics)

    # ── Summary ─────────────────────────────────────────────────────────────

    def summary(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before summary")
        print("=" * 80)
        print("HDFE REGRESSION RESULTS")
        print("=" * 80)
        print(f"R²: {self.r_squared_:.6f}")
        print(f"Observations: {len(self.residuals_):,}  |  df_resid: {self._df_resid:,}")
        print(f"Fixed effects: {len(self.fe_vars)}  |  SE type: {self.se_type}")
        if hasattr(self, 'cluster_vars') and self.cluster_vars:
            print(f"Clustering: {self.cluster_vars}")
        print(f"FE categories: {dict(self.n_categories)}")
        hdr = f"\n{'Variable':<20} {'Coef':<12} {'Std Err':<12} {'t':<8} {'P>|t|':<8}"
        print(hdr); print("-" * 60)
        for i, v in enumerate(self.X_cols):
            if not np.isnan(self.std_errors_[i]):
                print(f"{v:<20} {self.coefficients_[i]:<12.6f} "
                      f"{self.std_errors_[i]:<12.6f} "
                      f"{self.t_stats_[i]:<8.3f} {self.p_values_[i]:<8.3f}")
            else:
                print(f"{v:<20} {self.coefficients_[i]:<12.6f} "
                      f"{'N/A':<12} {'N/A':<8} {'N/A':<8}")
        print("\nFixed Effects Summary:"); print("-" * 60)
        for fv in self.fe_vars:
            c = self.fe_coefficients_[fv]
            print(f"  {fv}: mean={np.mean(c):.4f}, std={np.std(c):.4f}, "
                  f"min={np.min(c):.4f}, max={np.max(c):.4f}")

        if self._rank_diagnostics:
            print("\n⚠️  Rank Diagnostics:")
            for key, val in self._rank_diagnostics.items():
                print(f"  {key}: {val}")
        print("=" * 80)


class HDFEIV(HDFE):
    """
    High-Dimensional Fixed Effects Instrumental Variables (HDFE-IV) estimator.

    Extends HDFE with two-stage least squares (2SLS) estimation.  When no
    instruments are supplied, falls back to standard HDFE-OLS.
    """

    def __init__(self, max_iter=5000, tolerance=1e-8, acceleration='gk',
                 use_gpu=False, verbose=False):
        super().__init__(max_iter, tolerance, acceleration, use_gpu, verbose)
        self._reset_iv_state()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _reset_iv_state(self):
        """Clear all IV-specific state (called at the start of every fit)."""
        self._is_iv = False
        self._instruments = None
        self._endogenous_vars = None
        self._exogenous_vars = None

        self._first_stage_models = {}
        self._first_stage_fitted = {}
        self._first_stage_residuals = {}
        self._first_stage_r2 = {}
        self._first_stage_f_stats = {}

        self._tZX = None
        self._tXZ = None
        self._tZy = None
        self._tZZinv = None
        self._Z_full = None

        self._weak_instruments = False
        self._sargan_stat = None
        self._sargan_pvalue = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, data, y_col, X_cols, fe_vars, se_type='homoscedastic',
            cluster_vars=None, sample_weight=None, instruments=None,
            endogenous_vars=None, singular_fallback='lsqr'):
        """
        Fit HDFE-IV model with optional instrumental variables.

        Parameters
        ----------
        data : DataFrame
        y_col : str
        X_cols : list -- continuous variables (exogenous + endogenous)
        fe_vars : list -- fixed-effect variables
        se_type : str -- 'homoscedastic', 'hc1', 'cluster'
                        (hc2/hc3 not supported for IV)
        cluster_vars : list, optional
        sample_weight : array-like, optional
        instruments : list, optional -- excluded instrument column names
        endogenous_vars : list, optional -- subset of X_cols
        singular_fallback : str -- 'lsqr' (default) or 'raise'
        """
        self._reset_iv_state()
        if singular_fallback not in ('lsqr', 'raise'):
            raise ValueError("singular_fallback must be 'lsqr' or 'raise'")
        self._rank_diagnostics = {}

        self._instruments = instruments
        self._endogenous_vars = endogenous_vars if endogenous_vars else []
        self._is_iv = (instruments is not None and len(instruments) > 0 and
                       endogenous_vars is not None and len(endogenous_vars) > 0)

        if not self._is_iv:
            if self.verbose:
                print("No instruments provided, running standard HDFE estimation...")
            return super().fit(data, y_col, X_cols, fe_vars, se_type,
                               cluster_vars, sample_weight,
                               singular_fallback=singular_fallback)

        if se_type in ('hc2', 'hc3'):
            raise NotImplementedError(
                f"se_type='{se_type}' is not supported for IV models. "
                "Use 'homoscedastic', 'hc1', or 'cluster'.")

        if not set(endogenous_vars).issubset(set(X_cols)):
            raise ValueError("All endogenous variables must be in X_cols")
        if len(instruments) < len(endogenous_vars):
            raise ValueError(
                "Number of instruments must be >= number of endogenous variables")

        valid_se_types = ['homoscedastic', 'hc1', 'cluster']
        if se_type not in valid_se_types:
            raise ValueError(f"se_type must be one of {valid_se_types} for IV")
        if se_type == 'cluster' and cluster_vars is None:
            raise ValueError("cluster_vars must be specified when se_type='cluster'")
        if se_type == 'cluster' and not isinstance(cluster_vars, list):
            cluster_vars = [cluster_vars]
        if not fe_vars:
            raise ValueError("fe_vars must be non-empty for HDFE estimation")

        self._exogenous_vars = [x for x in X_cols if x not in endogenous_vars]
        self.se_type = se_type
        self.cluster_vars = cluster_vars
        self.fe_vars = fe_vars
        self.y_col = y_col
        self.X_cols = X_cols

        if self.verbose:
            print(f"Fitting HDFE-IV: {len(data):,} obs, "
                  f"{len(self._exogenous_vars)} exog, "
                  f"{len(endogenous_vars)} endog, "
                  f"{len(instruments)} instruments, "
                  f"{len(fe_vars)} FEs, SE={se_type}")

        # Prepare data
        if hasattr(data, 'to_pandas'):
            data = data.to_pandas()
        encoded_data = self._establish_category_ordering(data, fe_vars)

        y = data[y_col].values.astype(np.float64)
        X = data[X_cols].values.astype(np.float64)
        Z = data[instruments].values.astype(np.float64)

        valid_mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1)
                       | np.any(np.isnan(Z), axis=1))
        for fv in fe_vars:
            valid_mask &= data[fv].notna().values
        y = y[valid_mask]
        X = X[valid_mask]
        Z = Z[valid_mask]
        encoded_data = encoded_data[valid_mask]

        self._sample_weight = None
        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]
            self._sample_weight = sample_weight
            sw_sqrt = np.sqrt(sample_weight)
            y = y * sw_sqrt
            X = X * sw_sqrt.reshape(-1, 1)
            Z = Z * sw_sqrt.reshape(-1, 1)

        # Demean all columns in one pass
        if self.verbose:
            print("  Demeaning y, X, Z (single pass)...")
        combined = np.column_stack([X, Z])
        y_demeaned, combined_demeaned = self._alternating_projection(
            y, combined, encoded_data, fe_vars)
        X_demeaned = combined_demeaned[:, :X.shape[1]]
        Z_demeaned = combined_demeaned[:, X.shape[1]:]

        # First stage
        if self.verbose:
            print("  First stage regressions...")
        self._run_first_stage(X_demeaned, Z_demeaned)

        # Second stage (direct 2SLS formula)
        if self.verbose:
            print("  Second stage 2SLS...")
        self._run_second_stage(y_demeaned, X_demeaned)

        # Recover fixed effects
        self.fe_coefficients_ = self._recover_fixed_effects(
            y, X, encoded_data, self.coefficients_, y_demeaned, X_demeaned,
            sample_weight=self._sample_weight,
            singular_fallback=singular_fallback)

        self.X_projected = X_demeaned
        self.y_projected = y_demeaned

        # dof
        if len(self.fe_vars) > 0:
            df_absorbed = self.n_categories[self.fe_vars[0]]
            for fe in self.fe_vars[1:]:
                df_absorbed += self.n_categories[fe] - 1
        else:
            df_absorbed = 0
        self._df_resid = max(len(y) - X.shape[1] - df_absorbed, 1)

        if self._df_resid <= X.shape[1]:
            warnings.warn(
                f"Very low residual degrees of freedom ({self._df_resid}). "
                f"N={len(y)}, k={X.shape[1]}, absorbed={df_absorbed}. "
                "Standard errors may be unreliable.",
                stacklevel=2)

        # Statistics
        self._calculate_iv_statistics(
            y, X, Z, encoded_data, X_demeaned, data, valid_mask)

        # Diagnostics
        if self.verbose:
            print("  IV diagnostics...")
        self._compute_iv_diagnostics()

        self.fitted = True
        return self

    # ── first stage ─────────────────────────────────────────────────────────

    def _run_first_stage(self, X_demeaned, Z_demeaned):
        """First-stage regressions on Z_full = [X_exog, Z_excluded]."""
        exog_indices = [self.X_cols.index(v) for v in self._exogenous_vars]

        if len(exog_indices) > 0:
            X_exog = X_demeaned[:, exog_indices]
            Z_full = np.column_stack([X_exog, Z_demeaned])
        else:
            Z_full = Z_demeaned.copy()

        self._Z_full = Z_full
        ZTZ_full = Z_full.T @ Z_full

        endog_indices = [self.X_cols.index(v) for v in self._endogenous_vars]

        for i, endog_var in enumerate(self._endogenous_vars):
            X_endog = X_demeaned[:, endog_indices[i]]

            try:
                ZTX = Z_full.T @ X_endog
                pi_hat = np.linalg.solve(ZTZ_full, ZTX)
                X_fitted = Z_full @ pi_hat
                residuals = X_endog - X_fitted

                self._first_stage_models[endog_var] = {
                    'coefficients': pi_hat,
                    'instrument_names': self._exogenous_vars + self._instruments
                }
                self._first_stage_fitted[endog_var] = X_fitted
                self._first_stage_residuals[endog_var] = residuals

                tss = np.sum((X_endog - np.mean(X_endog))**2)
                rss = np.sum(residuals**2)
                r2 = 1 - rss / tss
                self._first_stage_r2[endog_var] = r2

                # Partial F-stat for *excluded* instruments only
                if len(exog_indices) > 0:
                    # Guard 1: rank check before solve (solve on near-singular
                    # matrices returns finite garbage instead of raising)
                    if np.linalg.matrix_rank(X_exog) < X_exog.shape[1]:
                        f_stat = np.nan
                        self._rank_diagnostics[
                            'first_stage_restricted_singular'] = True
                        if self.verbose:
                            print(f"    ⚠️ X_exog rank-deficient for "
                                  f"{endog_var}, partial F-stat unavailable")
                    else:
                        try:
                            pi_restricted = np.linalg.solve(
                                X_exog.T @ X_exog, X_exog.T @ X_endog)
                            if not np.all(np.isfinite(pi_restricted)):
                                raise np.linalg.LinAlgError(
                                    "Non-finite restricted coefficients")
                            rss_restricted = np.sum(
                                (X_endog - X_exog @ pi_restricted)**2)
                            q = Z_demeaned.shape[1]
                            n_obs = len(X_endog)
                            k_full = Z_full.shape[1]
                            f_stat = (((rss_restricted - rss) / q)
                                      / (rss / (n_obs - k_full)))
                            if not np.isfinite(f_stat):
                                raise np.linalg.LinAlgError(
                                    "Non-finite F-stat")
                        except np.linalg.LinAlgError:
                            f_stat = np.nan
                            self._rank_diagnostics[
                                'first_stage_restricted_singular'] = True
                            if self.verbose:
                                print(f"    ⚠️ X_exog singular for "
                                      f"{endog_var}, partial F unavailable")
                else:
                    n_obs = len(X_endog)
                    k = Z_demeaned.shape[1]
                    f_stat = (r2 / max(1 - r2, 1e-30)) * ((n_obs - k) / k)

                self._first_stage_f_stats[endog_var] = f_stat

                if self.verbose:
                    print(f"    {endog_var}: R²={r2:.4f}, "
                          f"partial-F={f_stat:.2f}")

            except np.linalg.LinAlgError:
                raise ValueError(
                    f"Rank-deficient instruments for {endog_var}. "
                    "Check for multicollinearity.")

    # ── second stage ────────────────────────────────────────────────────────

    def _run_second_stage(self, y_demeaned, X_demeaned):
        """
        2SLS with original X_demeaned (not X_2sls with fitted values).

        beta = (X'Z (Z'Z)^{-1} Z'X)^{-1}  X'Z (Z'Z)^{-1} Z'y
        """
        Z = self._Z_full

        self._tZX  = Z.T @ X_demeaned
        self._tXZ  = X_demeaned.T @ Z
        self._tZy  = Z.T @ y_demeaned

        # Guard 2: Z'Z inversion with NaN/Inf check
        try:
            self._tZZinv = np.linalg.inv(Z.T @ Z)
            if not np.all(np.isfinite(self._tZZinv)):
                raise np.linalg.LinAlgError(
                    "Non-finite Z'Z inverse")
        except np.linalg.LinAlgError:
            self._rank_diagnostics['ZtZ_singular'] = True
            raise ValueError(
                "Z'Z is singular — instruments (including exogenous "
                "regressors) are collinear. Check for multicollinearity "
                "in the instrument set.")

        try:
            H = self._tXZ @ self._tZZinv
            A = H @ self._tZX
            b = H @ self._tZy
            self.coefficients_ = np.linalg.solve(A, b)
            if self.verbose:
                print("    2SLS coefficients computed.")
        except np.linalg.LinAlgError:
            self._rank_diagnostics['2sls_A_singular'] = True
            raise ValueError(
                "2SLS estimation failed — check for identification issues.")

    # ── IV standard errors ──────────────────────────────────────────────────

    def _compute_iv_robust_standard_errors(self, X_demeaned, residuals,
                                           data, valid_mask):
        """
        IV-robust standard errors using the correct sandwich formula.

        Var(beta) = A^{-1} B A^{-1}
        where A = X'Z(Z'Z)^{-1}Z'X  and  B depends on se_type.
        """
        Z = self._Z_full
        A = self._tXZ @ self._tZZinv @ self._tZX

        # Guard 3: A inversion with NaN/Inf check
        try:
            A_inv = np.linalg.inv(A)
            if not np.all(np.isfinite(A_inv)):
                raise np.linalg.LinAlgError(
                    "Non-finite A inverse")
        except np.linalg.LinAlgError:
            self._rank_diagnostics['iv_se_A_singular'] = True
            raise np.linalg.LinAlgError(
                "X'Z(Z'Z)⁻¹Z'X is singular — cannot compute "
                "IV standard errors")

        if self.se_type == 'homoscedastic':
            sigma2 = np.sum(residuals**2) / self._df_resid
            var_beta = sigma2 * A_inv

        elif self.se_type == 'hc1':
            n = len(residuals)
            k = X_demeaned.shape[1]
            e2 = residuals**2
            meat_ZZ = (Z * e2.reshape(-1, 1)).T @ Z
            meat_ZZ *= n / max(n - k, 1)
            B = self._tXZ @ self._tZZinv @ meat_ZZ @ self._tZZinv @ self._tZX
            var_beta = A_inv @ B @ A_inv

        elif self.se_type == 'cluster':
            cluster_data = {}
            for cv in self.cluster_vars:
                cluster_data[cv] = data[cv].values[valid_mask]

            if len(self.cluster_vars) == 1:
                cluster_groups = cluster_data[self.cluster_vars[0]]
            else:
                cluster_groups = cluster_data[self.cluster_vars[0]].astype(str)
                for cv in self.cluster_vars[1:]:
                    cluster_groups = (cluster_groups + "_"
                                     + cluster_data[cv].astype(str))

            unique_clusters = np.unique(cluster_groups)
            m_z = Z.shape[1]
            meat_ZZ = np.zeros((m_z, m_z))
            for cluster in unique_clusters:
                mask = cluster_groups == cluster
                score = Z[mask].T @ residuals[mask]
                meat_ZZ += np.outer(score, score)

            B = self._tXZ @ self._tZZinv @ meat_ZZ @ self._tZZinv @ self._tZX
            var_beta = A_inv @ B @ A_inv

        else:
            raise NotImplementedError(
                f"se_type='{self.se_type}' not supported for IV.")

        return np.sqrt(np.diag(var_beta))

    # ── IV statistics ───────────────────────────────────────────────────────

    def _calculate_iv_statistics(self, y, X, Z, encoded_data, X_demeaned,
                                 original_data, valid_mask):
        """Calculate model statistics with IV-robust standard errors."""
        y_pred_full = X @ self.coefficients_
        for fe_var in self.fe_vars:
            group_ids = encoded_data[fe_var].values
            valid = group_ids >= 0
            fe_contrib = self.fe_coefficients_[fe_var][group_ids[valid]]
            if self._sample_weight is not None:
                fe_contrib = fe_contrib * np.sqrt(self._sample_weight[valid])
            y_pred_full[valid] += fe_contrib

        residuals_full = y - y_pred_full
        self.residuals_ = residuals_full
        self.fitted_values_ = y_pred_full

        tss = np.sum((y - np.mean(y))**2)
        rss = np.sum(residuals_full**2)
        self.r_squared_ = 1 - rss / tss
        if self.verbose:
            print(f"  Computing {self.se_type} IV standard errors...")

        try:
            self.std_errors_ = self._compute_iv_robust_standard_errors(
                X_demeaned, residuals_full, original_data, valid_mask)
            self.t_stats_ = self.coefficients_ / self.std_errors_
            self.p_values_ = 2 * (1 - stats.t.cdf(
                np.abs(self.t_stats_), self._df_resid))
        except Exception as e:
            if self.verbose:
                print(f"  Warning: IV SE computation failed: {e}")
            self.std_errors_ = np.full_like(self.coefficients_, np.nan)
            self.t_stats_ = np.full_like(self.coefficients_, np.nan)
            self.p_values_ = np.full_like(self.coefficients_, np.nan)

    # ── IV diagnostics ──────────────────────────────────────────────────────

    def _compute_iv_diagnostics(self):
        """Weak-instrument check and Sargan over-identification test."""
        min_f = (min(self._first_stage_f_stats.values())
                 if self._first_stage_f_stats else 0)
        self._weak_instruments = min_f < 10.0

        if self.verbose:
            print(f"    Min first-stage partial-F: {min_f:.2f}"
                  f"{'  ⚠️ weak' if self._weak_instruments else '  ✅ ok'}")
        n_excl = len(self._instruments)
        n_endog = len(self._endogenous_vars)

        if n_excl > n_endog:
            try:
                e = self.y_projected - self.X_projected @ self.coefficients_
                Z = self._Z_full
                ZTZ = Z.T @ Z
                ZTe = Z.T @ e
                gamma = np.linalg.solve(ZTZ, ZTe)
                e_hat = Z @ gamma
                ess = np.sum(e_hat**2)
                tss_e = np.sum((e - np.mean(e))**2)
                aux_r2 = ess / tss_e if tss_e > 0 else 0.0

                self._sargan_stat = len(e) * aux_r2
                df_sargan = n_excl - n_endog
                self._sargan_pvalue = 1 - stats.chi2.cdf(
                    self._sargan_stat, df_sargan)

                if self.verbose:
                    print(f"    Sargan χ²={self._sargan_stat:.3f}  "
                          f"p={self._sargan_pvalue:.3f}  "
                          f"(df={df_sargan})")
            except Exception as e:
                if self.verbose:
                    print(f"    Sargan test failed: {e}")
                self._sargan_stat = None
                self._sargan_pvalue = None

    # ── summary ─────────────────────────────────────────────────────────────

    def summary(self):
        """Print comprehensive IV model summary."""
        if not self.fitted:
            raise ValueError("Model must be fitted before summary")

        if not self._is_iv:
            return super().summary()

        print("=" * 90)
        print("HDFE-IV REGRESSION RESULTS (2SLS with HDFE)")
        print("=" * 90)
        print(f"R²: {self.r_squared_:.6f}  |  N: {len(self.residuals_):,}  |  "
              f"df_resid: {self._df_resid:,}")
        print(f"SE type: {self.se_type}")
        if self.cluster_vars:
            print(f"Cluster: {self.cluster_vars}")
        print(f"Endogenous: {self._endogenous_vars}")
        print(f"Excluded instruments: {self._instruments}")
        print(f"Exogenous: {self._exogenous_vars}")

        print("\nFirst Stage:")
        print("-" * 60)
        for var in self._endogenous_vars:
            f_val = self._first_stage_f_stats[var]
            f_str = f"{f_val:.2f}" if np.isfinite(f_val) else "N/A (singular)"
            print(f"  {var}: R²={self._first_stage_r2[var]:.4f}  "
                  f"partial-F={f_str}")
        if self._weak_instruments:
            print("  ⚠️  Weak instruments (F < 10)")

        if self._sargan_stat is not None:
            print(f"\nSargan test: χ²={self._sargan_stat:.3f}  "
                  f"p={self._sargan_pvalue:.3f}")

        print(f"\n{'Variable':<20} {'Coef':<12} {'Std Err':<12} "
              f"{'t':<8} {'P>|t|':<8} {'Type':<12}")
        print("-" * 72)
        for i, var in enumerate(self.X_cols):
            vtype = "Endogenous" if var in self._endogenous_vars else "Exogenous"
            if not np.isnan(self.std_errors_[i]):
                print(f"{var:<20} {self.coefficients_[i]:<12.6f} "
                      f"{self.std_errors_[i]:<12.6f} "
                      f"{self.t_stats_[i]:<8.3f} {self.p_values_[i]:<8.3f} "
                      f"{vtype:<12}")
            else:
                print(f"{var:<20} {self.coefficients_[i]:<12.6f} "
                      f"{'N/A':<12} {'N/A':<8} {'N/A':<8} {vtype:<12}")

        print("\nFixed Effects:")
        print("-" * 60)
        for fe_var in self.fe_vars:
            c = self.fe_coefficients_[fe_var]
            print(f"  {fe_var}: mean={np.mean(c):.4f}  std={np.std(c):.4f}  "
                  f"min={np.min(c):.4f}  max={np.max(c):.4f}")
        if self._rank_diagnostics:
            print("\n⚠️  Rank Diagnostics:")
            for key, val in self._rank_diagnostics.items():
                print(f"  {key}: {val}")
        print("=" * 90)

    def first_stage_results(self):
        """Return first-stage regression details."""
        if not self._is_iv:
            print("Not an IV model.")
            return None
        return {var: {
            'coefficients': self._first_stage_models[var]['coefficients'],
            'instrument_names': self._first_stage_models[var]['instrument_names'],
            'r_squared': self._first_stage_r2[var],
            'f_statistic': self._first_stage_f_stats[var],
            'fitted_values': self._first_stage_fitted[var],
            'residuals': self._first_stage_residuals[var],
        } for var in self._endogenous_vars}

    def iv_diagnostics(self):
        """Return IV diagnostic test results."""
        if not self._is_iv:
            print("Not an IV model.")
            return None
        return {
            'weak_instruments': self._weak_instruments,
            'first_stage_f_stats': dict(self._first_stage_f_stats),
            'sargan_statistic': self._sargan_stat,
            'sargan_pvalue': self._sargan_pvalue,
        }
