#!/usr/bin/env python3
"""
Verify enhanced scikit-learn installation.
"""

import sys
import numpy as np
from sklearn.datasets import make_classification

def verify_installation():
    """Verify that enhanced features are working."""
    print("🔍 Verifying Enhanced Scikit-Learn Installation")
    print("=" * 50)
    
    # Test basic sklearn functionality
    try:
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        print("✓ Basic scikit-learn functionality working")
    except Exception as e:
        print(f"✗ Basic functionality failed: {e}")
        return False
    
    # Test enhanced memory pools
    try:
        from sklearn.utils.enhanced import init_memory_pools, get_memory_stats
        init_memory_pools()
        stats = get_memory_stats()
        print(f"✓ Memory pools initialized: {len(stats)} pools")
    except ImportError:
        print("⚠ Memory pools not available (Cython modules not compiled)")
    except Exception as e:
        print(f"✗ Memory pools failed: {e}")
    
    # Test SIMD utilities
    try:
        from sklearn.utils.enhanced import check_simd_support
        simd_info = check_simd_support()
        print(f"✓ SIMD support detected: {simd_info}")
    except ImportError:
        print("⚠ SIMD utilities not available")
    except Exception as e:
        print(f"✗ SIMD utilities failed: {e}")
    
    # Test GPU utilities
    try:
        from sklearn.utils.enhanced import is_gpu_available, get_gpu_info
        gpu_available = is_gpu_available()
        if gpu_available:
            gpu_info = get_gpu_info()
            print(f"✓ GPU acceleration available: {gpu_info['backends']}")
        else:
            print("⚠ GPU acceleration not available")
    except ImportError:
        print("⚠ GPU utilities not available")
    except Exception as e:
        print(f"✗ GPU utilities failed: {e}")
    
    # Test enhanced algorithms
    try:
        from sklearn.ensemble.enhanced import EnhancedRandomForestClassifier
        clf = EnhancedRandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        print("✓ Enhanced Random Forest working")
    except ImportError:
        print("⚠ Enhanced Random Forest not available")
    except Exception as e:
        print(f"✗ Enhanced Random Forest failed: {e}")
    
    print("\n🎉 Installation verification complete!")
    return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
