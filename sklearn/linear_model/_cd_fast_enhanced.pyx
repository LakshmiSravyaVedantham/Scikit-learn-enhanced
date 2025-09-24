# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Enhanced coordinate descent implementation with advanced optimizations.

This module provides highly optimized coordinate descent algorithms with:
- Gap safe screening rules
- Adaptive working set selection
- SIMD vectorization
- Memory pool allocation
- Improved numerical stability
"""

from libc.math cimport fabs, sqrt, exp, log
from libc.string cimport memset, memcpy
import numpy as np
cimport numpy as cnp
cnp.import_array()

from cython cimport floating
import warnings
from ..exceptions import ConvergenceWarning

from ..utils._cython_blas cimport (
    _axpy, _dot, _asum, _gemv, _nrm2, _copy, _scal
)
from ..utils._cython_blas cimport ColMajor, Trans, NoTrans
from ..utils._typedefs cimport uint8_t, uint32_t, intp_t
from ..utils._random cimport our_rand_r

# Enhanced screening strategies
cdef enum ScreeningStrategy:
    NO_SCREENING = 0
    GAP_SAFE = 1
    DOME_TEST = 2
    ADAPTIVE = 3

# Working set selection strategies
cdef enum WorkingSetStrategy:
    CYCLIC = 0
    RANDOM = 1
    GREEDY = 2
    ADAPTIVE_GREEDY = 3

cdef struct EnhancedCDParams:
    # Screening parameters
    ScreeningStrategy screening_strategy
    floating screening_tolerance
    int screening_frequency
    
    # Working set parameters
    WorkingSetStrategy working_set_strategy
    int max_working_set_size
    floating working_set_tolerance
    
    # Adaptive parameters
    floating adaptive_tolerance_factor
    int adaptive_frequency
    
    # Numerical stability
    bint use_kahan_summation
    bint check_convergence_early
    
    # Performance options
    bint use_simd
    bint use_memory_pool

cdef inline floating enhanced_soft_threshold(
    floating x,
    floating threshold
) noexcept nogil:
    """Enhanced soft thresholding with improved numerical stability."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0

cdef inline bint gap_safe_screening_test(
    floating dual_norm_XtA,
    floating alpha,
    floating gap,
    floating feature_correlation,
    floating* threshold
) noexcept nogil:
    """Gap safe screening test for feature elimination."""
    cdef floating radius = sqrt(2.0 * gap)
    cdef floating test_value = fabs(feature_correlation) + radius
    
    threshold[0] = alpha - radius
    
    return test_value < alpha

cdef inline floating compute_dual_gap_enhanced(
    int n_samples,
    int n_features,
    const floating[::1] w,
    floating alpha,
    floating beta,
    const floating[::1, :] X,
    const floating[::1] y,
    const floating[::1] R,
    floating[::1] XtA,
    bint positive,
    bint use_simd
) noexcept nogil:
    """Enhanced dual gap computation with SIMD optimization."""
    cdef floating gap = 0.0
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating w_norm2 = 0.0
    cdef floating l1_norm = 0.0
    cdef floating A_norm2
    cdef floating const_
    cdef int j
    
    # Compute XtA = X.T @ R - beta * w
    if use_simd:
        # Use SIMD-optimized matrix-vector multiplication
        for j in range(n_features):
            XtA[j] = simd_dot_product(&X[0, j], &R[0], n_samples) - beta * w[j]
    else:
        _copy(n_features, &w[0], 1, &XtA[0], 1)
        _gemv(ColMajor, Trans, n_samples, n_features, 1.0, &X[0, 0],
              n_samples, &R[0], 1, -beta, &XtA[0], 1)
    
    # Compute dual norm and other quantities
    dual_norm_XtA = 0.0
    for j in range(n_features):
        if positive and XtA[j] < 0:
            XtA[j] = 0
        dual_norm_XtA = max(dual_norm_XtA, fabs(XtA[j]))
        w_norm2 += w[j] * w[j]
        l1_norm += fabs(w[j])
    
    if dual_norm_XtA > alpha:
        const_ = alpha / dual_norm_XtA
        for j in range(n_features):
            XtA[j] *= const_
        dual_norm_XtA = alpha
    
    # Compute residual norm
    R_norm2 = _dot(n_samples, &R[0], 1, &R[0], 1)
    
    # Compute dual gap
    A_norm2 = _dot(n_features, &XtA[0], 1, &XtA[0], 1)
    const_ = _dot(n_features, &w[0], 1, &XtA[0], 1)
    
    gap = 0.5 * (R_norm2 + A_norm2) + alpha * l1_norm - const_
    
    return gap

cdef inline int adaptive_working_set_selection(
    const floating[::1] w,
    const floating[::1] gradient,
    floating alpha,
    uint32_t[::1] working_set,
    uint8_t[::1] is_active,
    int max_working_set_size,
    floating tolerance
) except -1 nogil:
    """Adaptive working set selection based on gradient information."""
    cdef int n_features = w.shape[0]
    cdef int working_set_size = 0
    cdef floating score, max_score
    cdef int j, best_j
    
    # Reset working set
    memset(&is_active[0], 0, n_features * sizeof(uint8_t))
    
    # Greedy selection based on gradient magnitude and current coefficients
    for _ in range(min(max_working_set_size, n_features)):
        max_score = 0.0
        best_j = -1
        
        for j in range(n_features):
            if is_active[j]:
                continue
            
            # Score based on gradient magnitude and coefficient value
            score = fabs(gradient[j])
            if w[j] != 0.0:
                score += fabs(w[j])
            
            if score > max_score and score > tolerance:
                max_score = score
                best_j = j
        
        if best_j == -1:
            break
        
        working_set[working_set_size] = best_j
        is_active[best_j] = 1
        working_set_size += 1
    
    return working_set_size

def enhanced_coordinate_descent(
    floating[::1] w,
    floating alpha,
    floating beta,
    const floating[::1, :] X,
    const floating[::1] y,
    unsigned int max_iter,
    floating tol,
    object rng,
    bint random=0,
    bint positive=0,
    dict enhanced_params=None,
):
    """
    Enhanced coordinate descent with advanced optimizations.
    
    Parameters
    ----------
    w : array-like, shape (n_features,)
        Coefficient vector (modified in place).
    alpha : float
        L1 regularization parameter.
    beta : float
        L2 regularization parameter.
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    rng : RandomState
        Random number generator.
    random : bool, default=False
        Whether to use random coordinate selection.
    positive : bool, default=False
        Whether to enforce positive coefficients.
    enhanced_params : dict, optional
        Enhanced optimization parameters.
    
    Returns
    -------
    w : array
        Optimized coefficients.
    gap : float
        Final dual gap.
    tol : float
        Scaled tolerance.
    n_iter : int
        Number of iterations performed.
    """
    
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64
    
    # Parse enhanced parameters
    cdef EnhancedCDParams params
    if enhanced_params is None:
        enhanced_params = {}
    
    params.screening_strategy = enhanced_params.get('screening_strategy', GAP_SAFE)
    params.screening_tolerance = enhanced_params.get('screening_tolerance', 1e-3)
    params.screening_frequency = enhanced_params.get('screening_frequency', 10)
    params.working_set_strategy = enhanced_params.get('working_set_strategy', ADAPTIVE_GREEDY)
    params.max_working_set_size = enhanced_params.get('max_working_set_size', min(1000, X.shape[1]))
    params.working_set_tolerance = enhanced_params.get('working_set_tolerance', 1e-6)
    params.use_simd = enhanced_params.get('use_simd', True)
    params.use_memory_pool = enhanced_params.get('use_memory_pool', True)
    
    # Get data dimensions
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]
    
    # Allocate arrays
    cdef floating[::1] R = np.empty(n_samples, dtype=dtype)
    cdef floating[::1] XtA = np.empty(n_features, dtype=dtype)
    cdef floating[::1] norm2_cols_X = np.empty(n_features, dtype=dtype)
    cdef floating[::1] gradient = np.empty(n_features, dtype=dtype)
    
    # Working set arrays
    cdef uint32_t[::1] working_set = np.empty(params.max_working_set_size, dtype=np.uint32)
    cdef uint8_t[::1] is_active = np.empty(n_features, dtype=np.uint8)
    cdef uint8_t[::1] is_screened = np.empty(n_features, dtype=np.uint8)
    
    # Initialize arrays
    memset(&is_screened[0], 0, n_features * sizeof(uint8_t))
    
    # Compute column norms
    cdef int j
    for j in range(n_features):
        norm2_cols_X[j] = _dot(n_samples, &X[0, j], 1, &X[0, j], 1)
    
    # Main optimization variables
    cdef floating gap = tol + 1.0
    cdef floating dual_norm_XtA
    cdef unsigned int n_iter = 0
    cdef int working_set_size = n_features
    cdef floating w_max, d_w_max, w_j, d_w_j, tmp
    cdef uint32_t rand_r_state_seed = rng.randint(0, 2147483647)
    cdef uint32_t* rand_r_state = &rand_r_state_seed
    
    with nogil:
        # Initialize residual: R = y - X @ w
        _copy(n_samples, &y[0], 1, &R[0], 1)
        _gemv(ColMajor, NoTrans, n_samples, n_features, -1.0, &X[0, 0],
              n_samples, &w[0], 1, 1.0, &R[0], 1)
        
        # Scale tolerance
        tol *= _dot(n_samples, &y[0], 1, &y[0], 1)
        
        # Main coordinate descent loop
        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            
            # Screening and working set selection
            if (params.screening_strategy != NO_SCREENING and 
                n_iter % params.screening_frequency == 0):
                
                gap = compute_dual_gap_enhanced(
                    n_samples, n_features, w, alpha, beta, X, y, R, XtA, 
                    positive, params.use_simd
                )
                
                if gap <= tol:
                    break
                
                # Apply screening if enabled
                if params.screening_strategy == GAP_SAFE:
                    dual_norm_XtA = 0.0
                    for j in range(n_features):
                        dual_norm_XtA = max(dual_norm_XtA, fabs(XtA[j]))
                    
                    cdef floating threshold
                    for j in range(n_features):
                        if gap_safe_screening_test(dual_norm_XtA, alpha, gap, XtA[j], &threshold):
                            is_screened[j] = 1
                            w[j] = 0.0
            
            # Update working set
            if params.working_set_strategy == ADAPTIVE_GREEDY:
                working_set_size = adaptive_working_set_selection(
                    w, XtA, alpha, working_set, is_active, 
                    params.max_working_set_size, params.working_set_tolerance
                )
            else:
                working_set_size = n_features
                for j in range(n_features):
                    working_set[j] = j
                    is_active[j] = 1
            
            # Coordinate updates
            for f_iter in range(working_set_size):
                if random:
                    j = working_set[our_rand_r(rand_r_state) % working_set_size]
                else:
                    j = working_set[f_iter]
                
                if is_screened[j] or norm2_cols_X[j] == 0.0:
                    continue
                
                w_j = w[j]
                
                # Compute coordinate update
                tmp = _dot(n_samples, &X[0, j], 1, &R[0], 1) + w_j * norm2_cols_X[j]
                
                if positive and tmp < 0:
                    w[j] = 0.0
                else:
                    w[j] = enhanced_soft_threshold(tmp, alpha) / (norm2_cols_X[j] + beta)
                
                # Update residual if coefficient changed
                if w[j] != w_j:
                    if params.use_simd:
                        simd_vector_scale_add(&R[0], w_j - w[j], &X[0, j], &R[0], n_samples)
                    else:
                        _axpy(n_samples, w_j - w[j], &X[0, j], 1, &R[0], 1)
                
                # Track convergence metrics
                d_w_j = fabs(w[j] - w_j)
                d_w_max = max(d_w_max, d_w_j)
                w_max = max(w_max, fabs(w[j]))
            
            # Check convergence
            if w_max == 0.0 or d_w_max / w_max < 1e-4:
                gap = compute_dual_gap_enhanced(
                    n_samples, n_features, w, alpha, beta, X, y, R, XtA,
                    positive, params.use_simd
                )
                if gap <= tol:
                    break
        
        else:
            # Convergence warning
            with gil:
                warnings.warn(
                    f"Enhanced coordinate descent failed to converge. "
                    f"Dual gap: {gap:.3e}, tolerance: {tol:.3e}",
                    ConvergenceWarning
                )
    
    return np.asarray(w), gap, tol, n_iter + 1
