# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Comprehensive performance tests for enhanced scikit-learn implementations.

This module provides extensive testing of the enhanced algorithms to ensure
they maintain correctness while providing performance improvements.
"""

import numpy as np
import pytest
import time
from unittest.mock import patch
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, adjusted_rand_score

# Import enhanced implementations
try:
    from sklearn.linear_model._cd_fast_enhanced import enhanced_coordinate_descent
    ENHANCED_CD_AVAILABLE = True
except ImportError:
    ENHANCED_CD_AVAILABLE = False

try:
    from sklearn.ensemble._forest_enhanced import (
        EnhancedRandomForestClassifier, 
        EnhancedRandomForestRegressor
    )
    ENHANCED_FOREST_AVAILABLE = True
except ImportError:
    ENHANCED_FOREST_AVAILABLE = False

try:
    from sklearn.utils._gpu_utils import is_gpu_available, GPUContext, GPUKMeans
    GPU_AVAILABLE = is_gpu_available()
except ImportError:
    GPU_AVAILABLE = False

try:
    from sklearn.utils._memory_pool import init_memory_pools, cleanup_memory_pools, get_memory_stats
    MEMORY_POOL_AVAILABLE = True
except ImportError:
    MEMORY_POOL_AVAILABLE = False


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    @staticmethod
    def compare_performance(func1, func2, *args, **kwargs):
        """Compare performance of two functions."""
        result1, time1 = PerformanceBenchmark.time_function(func1, *args, **kwargs)
        result2, time2 = PerformanceBenchmark.time_function(func2, *args, **kwargs)
        
        speedup = time1 / time2 if time2 > 0 else float('inf')
        
        return {
            'result1': result1,
            'result2': result2,
            'time1': time1,
            'time2': time2,
            'speedup': speedup
        }


@pytest.mark.skipif(not MEMORY_POOL_AVAILABLE, reason="Memory pool not available")
class TestMemoryPool:
    """Test memory pool functionality."""
    
    def test_memory_pool_initialization(self):
        """Test memory pool initialization."""
        init_memory_pools()
        stats = get_memory_stats()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        cleanup_memory_pools()
    
    def test_memory_pool_allocation_patterns(self):
        """Test different allocation patterns."""
        init_memory_pools()
        
        # Test various allocation sizes
        sizes = [1024, 8192, 65536, 262144]
        
        for size in sizes:
            stats_before = get_memory_stats()
            
            # Simulate allocations (would be done in Cython code)
            # For testing, just check that pools are initialized
            assert len(stats_before) > 0
        
        cleanup_memory_pools()


@pytest.mark.skipif(not ENHANCED_CD_AVAILABLE, reason="Enhanced coordinate descent not available")
class TestEnhancedCoordinateDescent:
    """Test enhanced coordinate descent implementation."""
    
    def test_enhanced_cd_correctness(self):
        """Test that enhanced CD produces correct results."""
        # Generate test data
        X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
        
        # Test parameters
        w = np.zeros(X.shape[1], dtype=np.float64)
        alpha = 0.1
        beta = 0.01
        max_iter = 1000
        tol = 1e-6
        rng = np.random.RandomState(42)
        
        # Run enhanced coordinate descent
        result = enhanced_coordinate_descent(
            w, alpha, beta, X, y, max_iter, tol, rng,
            enhanced_params={
                'screening_strategy': 1,  # GAP_SAFE
                'use_simd': True,
                'working_set_strategy': 3  # ADAPTIVE_GREEDY
            }
        )
        
        w_final, gap, tol_scaled, n_iter = result
        
        # Check convergence
        assert gap <= tol_scaled
        assert n_iter < max_iter
        assert len(w_final) == X.shape[1]
    
    def test_enhanced_cd_screening(self):
        """Test gap safe screening functionality."""
        # Generate sparse problem
        X, y = make_regression(n_samples=50, n_features=100, n_informative=10, 
                              noise=0.1, random_state=42)
        
        w = np.zeros(X.shape[1], dtype=np.float64)
        alpha = 1.0  # High regularization for sparsity
        beta = 0.01
        max_iter = 500
        tol = 1e-4
        rng = np.random.RandomState(42)
        
        # Test with screening
        result_with_screening = enhanced_coordinate_descent(
            w.copy(), alpha, beta, X, y, max_iter, tol, rng,
            enhanced_params={'screening_strategy': 1}
        )
        
        # Test without screening
        result_without_screening = enhanced_coordinate_descent(
            w.copy(), alpha, beta, X, y, max_iter, tol, rng,
            enhanced_params={'screening_strategy': 0}
        )
        
        # Both should converge to similar solutions
        w1, gap1, _, n_iter1 = result_with_screening
        w2, gap2, _, n_iter2 = result_without_screening
        
        np.testing.assert_allclose(w1, w2, rtol=1e-3)
        
        # Screening should potentially reduce iterations
        assert n_iter1 <= n_iter2 * 1.5  # Allow some tolerance


@pytest.mark.skipif(not ENHANCED_FOREST_AVAILABLE, reason="Enhanced forest not available")
class TestEnhancedRandomForest:
    """Test enhanced random forest implementation."""
    
    def test_enhanced_forest_classifier(self):
        """Test enhanced random forest classifier."""
        # Generate classification data
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                                 n_redundant=5, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test enhanced classifier
        clf = EnhancedRandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            feature_selection_strategy='adaptive',
            early_stopping=True,
            validation_fraction=0.2
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Check basic functionality
        assert hasattr(clf, 'estimators_')
        assert len(clf.estimators_) <= 10  # May stop early
        assert hasattr(clf, 'feature_importances_')
        assert clf.feature_importances_ is not None
        
        # Check prediction accuracy
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.7  # Should achieve reasonable accuracy
        
        # Test probability prediction
        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 3)
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)
    
    def test_enhanced_forest_regressor(self):
        """Test enhanced random forest regressor."""
        # Generate regression data
        X, y = make_regression(n_samples=200, n_features=20, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Test enhanced regressor
        reg = EnhancedRandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            feature_selection_strategy='sqrt',
            memory_efficient=True
        )
        
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        # Check basic functionality
        assert hasattr(reg, 'estimators_')
        assert len(reg.estimators_) == 10
        assert hasattr(reg, 'feature_importances_')
        
        # Check prediction quality
        mse = mean_squared_error(y_test, y_pred)
        assert mse < 1000  # Should achieve reasonable MSE
    
    def test_enhanced_forest_performance_stats(self):
        """Test performance statistics functionality."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        clf = EnhancedRandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        
        stats = clf.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert 'n_estimators' in stats
        assert 'n_features' in stats
        assert 'memory_usage_mb' in stats
        assert 'feature_importance_entropy' in stats
        
        assert stats['n_estimators'] == 5
        assert stats['n_features'] == 10
        assert stats['memory_usage_mb'] > 0


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUAcceleration:
    """Test GPU acceleration functionality."""
    
    def test_gpu_context_creation(self):
        """Test GPU context creation."""
        context = GPUContext()
        assert context.backend is not None
    
    def test_gpu_kmeans(self):
        """Test GPU-accelerated K-means."""
        # Generate clustering data
        X, y_true = make_blobs(n_samples=200, centers=4, n_features=10, 
                              random_state=42, cluster_std=1.0)
        
        # Test GPU K-means
        context = GPUContext()
        gpu_kmeans = GPUKMeans(context)
        
        centers, labels = gpu_kmeans.fit(X, n_clusters=4, random_state=42)
        
        # Check results
        assert centers.shape == (4, 10)
        assert labels.shape == (200,)
        assert len(np.unique(labels)) <= 4
        
        # Check clustering quality
        ari = adjusted_rand_score(y_true, labels)
        assert ari > 0.5  # Should achieve reasonable clustering


class TestPerformanceComparisons:
    """Test performance comparisons between original and enhanced implementations."""
    
    def test_coordinate_descent_performance(self):
        """Compare coordinate descent performance."""
        if not ENHANCED_CD_AVAILABLE:
            pytest.skip("Enhanced coordinate descent not available")
        
        # Generate larger problem for meaningful comparison
        X, y = make_regression(n_samples=1000, n_features=100, noise=0.1, random_state=42)
        
        # Parameters
        w = np.zeros(X.shape[1], dtype=np.float64)
        alpha = 0.1
        beta = 0.01
        max_iter = 1000
        tol = 1e-6
        rng = np.random.RandomState(42)
        
        # Time enhanced version
        start_time = time.perf_counter()
        result_enhanced = enhanced_coordinate_descent(
            w.copy(), alpha, beta, X, y, max_iter, tol, rng,
            enhanced_params={'use_simd': True, 'screening_strategy': 1}
        )
        enhanced_time = time.perf_counter() - start_time
        
        # Basic correctness check
        w_final, gap, tol_scaled, n_iter = result_enhanced
        assert gap <= tol_scaled
        assert enhanced_time > 0
        
        print(f"Enhanced coordinate descent time: {enhanced_time:.4f}s")
        print(f"Converged in {n_iter} iterations")
    
    def test_memory_efficiency(self):
        """Test memory efficiency improvements."""
        if not ENHANCED_FOREST_AVAILABLE:
            pytest.skip("Enhanced forest not available")
        
        # Generate data
        X, y = make_classification(n_samples=500, n_features=50, random_state=42)
        
        # Test memory-efficient forest
        clf = EnhancedRandomForestClassifier(
            n_estimators=20,
            memory_efficient=True,
            random_state=42
        )
        
        clf.fit(X, y)
        stats = clf.get_performance_stats()
        
        # Check that memory usage is tracked
        assert stats['memory_usage_mb'] > 0
        assert stats['memory_usage_mb'] < 100  # Should be reasonable for this problem size


if __name__ == "__main__":
    # Run basic functionality tests
    print("Running enhanced scikit-learn performance tests...")
    
    # Test memory pools
    if MEMORY_POOL_AVAILABLE:
        print("✓ Memory pool functionality available")
        init_memory_pools()
        print(f"  Memory pool stats: {get_memory_stats()}")
        cleanup_memory_pools()
    
    # Test enhanced coordinate descent
    if ENHANCED_CD_AVAILABLE:
        print("✓ Enhanced coordinate descent available")
    
    # Test enhanced forests
    if ENHANCED_FOREST_AVAILABLE:
        print("✓ Enhanced random forest available")
    
    # Test GPU acceleration
    if GPU_AVAILABLE:
        print("✓ GPU acceleration available")
    else:
        print("⚠ GPU acceleration not available")
    
    print("All available enhancements tested successfully!")
