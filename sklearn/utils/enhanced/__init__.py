"""Enhanced utilities for scikit-learn."""

from .._memory_pool import init_memory_pools, cleanup_memory_pools, get_memory_stats
from .._simd_utils import check_simd_support, dot_product, euclidean_distance_squared
from .._gpu_utils import is_gpu_available, get_gpu_info, GPUContext

__all__ = [
    'init_memory_pools',
    'cleanup_memory_pools', 
    'get_memory_stats',
    'check_simd_support',
    'dot_product',
    'euclidean_distance_squared',
    'is_gpu_available',
    'get_gpu_info',
    'GPUContext'
]
