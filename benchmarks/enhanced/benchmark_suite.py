#!/usr/bin/env python3
"""
Enhanced scikit-learn benchmark suite.

This script runs comprehensive benchmarks comparing original and enhanced implementations.
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split

# Import both original and enhanced implementations
from sklearn.ensemble import RandomForestClassifier as OriginalRF
from sklearn.linear_model import Lasso as OriginalLasso
from sklearn.cluster import KMeans as OriginalKMeans

try:
    from sklearn.ensemble.enhanced import EnhancedRandomForestClassifier
    from sklearn.linear_model.enhanced import enhanced_coordinate_descent
    from sklearn.cluster.enhanced import enhanced_lloyd_iteration
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    if not ENHANCED_AVAILABLE:
        print("Enhanced implementations not available")
        return
    
    results = []
    
    # Random Forest benchmark
    print("Benchmarking Random Forest...")
    X, y = make_classification(n_samples=10000, n_features=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Original
    start = time.time()
    rf_orig = OriginalRF(n_estimators=100, random_state=42)
    rf_orig.fit(X_train, y_train)
    orig_time = time.time() - start
    
    # Enhanced
    start = time.time()
    rf_enh = EnhancedRandomForestClassifier(n_estimators=100, random_state=42)
    rf_enh.fit(X_train, y_train)
    enh_time = time.time() - start
    
    results.append({
        'Algorithm': 'Random Forest',
        'Original Time (s)': orig_time,
        'Enhanced Time (s)': enh_time,
        'Speedup': orig_time / enh_time
    })
    
    print(f"Random Forest: {orig_time:.2f}s -> {enh_time:.2f}s (speedup: {orig_time/enh_time:.2f}x)")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    run_benchmark_suite()
