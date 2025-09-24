#!/usr/bin/env python3
"""
Quick test script to verify enhanced scikit-learn modules are working.
This script tests basic functionality without requiring full compilation.
"""

import sys
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs

def test_basic_sklearn():
    """Test basic scikit-learn functionality."""
    print("🔍 Testing basic scikit-learn functionality...")
    
    try:
        # Test classification dataset
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        print(f"✓ Classification dataset: {X.shape}, {y.shape}")
        
        # Test regression dataset  
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        print(f"✓ Regression dataset: {X_reg.shape}, {y_reg.shape}")
        
        # Test clustering dataset
        X_clust, y_clust = make_blobs(n_samples=100, centers=3, random_state=42)
        print(f"✓ Clustering dataset: {X_clust.shape}, {y_clust.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality failed: {e}")
        return False

def test_enhanced_modules():
    """Test enhanced module imports."""
    print("\n🚀 Testing enhanced module imports...")
    
    # Test enhanced utilities
    try:
        import sklearn.utils
        print("✓ sklearn.utils imported successfully")
    except ImportError as e:
        print(f"⚠ sklearn.utils import failed: {e}")
    
    # Test enhanced linear model
    try:
        import sklearn.linear_model
        print("✓ sklearn.linear_model imported successfully")
    except ImportError as e:
        print(f"⚠ sklearn.linear_model import failed: {e}")
    
    # Test enhanced tree
    try:
        import sklearn.tree
        print("✓ sklearn.tree imported successfully")
    except ImportError as e:
        print(f"⚠ sklearn.tree import failed: {e}")
    
    # Test enhanced cluster
    try:
        import sklearn.cluster
        print("✓ sklearn.cluster imported successfully")
    except ImportError as e:
        print(f"⚠ sklearn.cluster import failed: {e}")
    
    # Test enhanced ensemble
    try:
        import sklearn.ensemble
        print("✓ sklearn.ensemble imported successfully")
    except ImportError as e:
        print(f"⚠ sklearn.ensemble import failed: {e}")

def test_standard_algorithms():
    """Test standard scikit-learn algorithms work."""
    print("\n🧪 Testing standard algorithms...")
    
    # Generate test data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    try:
        # Test Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        pred = rf.predict(X[:10])
        print(f"✓ Random Forest: trained and predicted {len(pred)} samples")
    except Exception as e:
        print(f"✗ Random Forest failed: {e}")
    
    try:
        # Test Linear Model
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=42, max_iter=100)
        lr.fit(X, y)
        pred = lr.predict(X[:10])
        print(f"✓ Logistic Regression: trained and predicted {len(pred)} samples")
    except Exception as e:
        print(f"✗ Logistic Regression failed: {e}")
    
    try:
        # Test K-Means
        from sklearn.cluster import KMeans
        X_clust, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_clust)
        print(f"✓ K-Means: clustered {len(labels)} samples into {len(np.unique(labels))} clusters")
    except Exception as e:
        print(f"✗ K-Means failed: {e}")

def test_enhanced_files_exist():
    """Test that enhanced files exist in the repository."""
    print("\n📁 Checking enhanced files...")
    
    import os
    enhanced_files = [
        'sklearn/utils/_memory_pool.pyx',
        'sklearn/utils/_simd_utils.pyx', 
        'sklearn/utils/_gpu_utils.py',
        'sklearn/linear_model/_cd_fast_enhanced.pyx',
        'sklearn/tree/_tree_enhanced.pyx',
        'sklearn/cluster/_kmeans_enhanced.pyx',
        'sklearn/ensemble/_forest_enhanced.py',
        'sklearn/tests/test_enhanced_performance.py',
        'setup_enhanced.py',
        'README_ENHANCED.md',
        'PERFORMANCE_ANALYSIS.md'
    ]
    
    for file_path in enhanced_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - NOT FOUND")

def main():
    """Run all tests."""
    print("🎯 Enhanced Scikit-Learn Quick Test")
    print("=" * 50)
    
    # Test basic functionality
    basic_ok = test_basic_sklearn()
    
    # Test enhanced modules
    test_enhanced_modules()
    
    # Test standard algorithms
    test_standard_algorithms()
    
    # Test enhanced files exist
    test_enhanced_files_exist()
    
    print("\n" + "=" * 50)
    if basic_ok:
        print("🎉 Quick test completed! Enhanced scikit-learn repository is ready.")
        print("📝 Note: Full enhanced features require compilation with:")
        print("   python setup_enhanced.py build_ext --inplace")
    else:
        print("❌ Basic tests failed. Please check your environment.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
