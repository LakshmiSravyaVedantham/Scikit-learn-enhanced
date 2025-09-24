# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
GPU acceleration utilities for scikit-learn.

This module provides GPU-accelerated implementations of common operations
using CUDA and OpenCL for maximum performance on modern hardware.
"""

import numpy as np
import warnings
from typing import Optional, Union, Tuple, Dict, Any

# GPU backend detection
_gpu_backends = {}
_default_backend = None

try:
    import cupy as cp
    _gpu_backends['cupy'] = cp
    if _default_backend is None:
        _default_backend = 'cupy'
except ImportError:
    pass

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    _gpu_backends['opencl'] = cl
    if _default_backend is None:
        _default_backend = 'opencl'
except ImportError:
    pass

try:
    import torch
    _gpu_backends['pytorch'] = torch
    if _default_backend is None:
        _default_backend = 'pytorch'
except ImportError:
    pass


class GPUContext:
    """GPU context manager for different backends."""
    
    def __init__(self, backend: Optional[str] = None, device_id: int = 0):
        self.backend = backend or _default_backend
        self.device_id = device_id
        self._context = None
        self._queue = None
        
        if self.backend is None:
            raise RuntimeError("No GPU backend available. Install CuPy, PyOpenCL, or PyTorch.")
        
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize GPU context based on backend."""
        if self.backend == 'cupy':
            cp.cuda.Device(self.device_id).use()
            self._context = cp.cuda.Device(self.device_id)
        
        elif self.backend == 'opencl':
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            
            devices = platforms[0].get_devices()
            if not devices:
                raise RuntimeError("No OpenCL devices found")
            
            self._context = cl.Context([devices[self.device_id]])
            self._queue = cl.CommandQueue(self._context)
        
        elif self.backend == 'pytorch':
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device_id)
                self._context = torch.cuda.current_device()
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """Transfer array to GPU."""
        if self.backend == 'cupy':
            return cp.asarray(array)
        elif self.backend == 'opencl':
            return cl_array.to_device(self._queue, array)
        elif self.backend == 'pytorch':
            return torch.from_numpy(array).cuda()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def to_cpu(self, gpu_array: Any) -> np.ndarray:
        """Transfer array from GPU to CPU."""
        if self.backend == 'cupy':
            return cp.asnumpy(gpu_array)
        elif self.backend == 'opencl':
            return gpu_array.get()
        elif self.backend == 'pytorch':
            return gpu_array.cpu().numpy()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


class GPULinearAlgebra:
    """GPU-accelerated linear algebra operations."""
    
    def __init__(self, context: GPUContext):
        self.context = context
        self.backend = context.backend
    
    def dot(self, a: Any, b: Any) -> Any:
        """GPU matrix multiplication."""
        if self.backend == 'cupy':
            return cp.dot(a, b)
        elif self.backend == 'opencl':
            # Would implement OpenCL GEMM here
            raise NotImplementedError("OpenCL GEMM not implemented")
        elif self.backend == 'pytorch':
            return torch.mm(a, b)
    
    def norm(self, a: Any, axis: Optional[int] = None) -> Any:
        """GPU vector/matrix norm."""
        if self.backend == 'cupy':
            return cp.linalg.norm(a, axis=axis)
        elif self.backend == 'pytorch':
            return torch.norm(a, dim=axis)
    
    def svd(self, a: Any) -> Tuple[Any, Any, Any]:
        """GPU singular value decomposition."""
        if self.backend == 'cupy':
            return cp.linalg.svd(a)
        elif self.backend == 'pytorch':
            return torch.svd(a)
    
    def solve(self, a: Any, b: Any) -> Any:
        """GPU linear system solver."""
        if self.backend == 'cupy':
            return cp.linalg.solve(a, b)
        elif self.backend == 'pytorch':
            return torch.solve(b, a)[0]


class GPUKMeans:
    """GPU-accelerated K-means clustering."""
    
    def __init__(self, context: GPUContext):
        self.context = context
        self.linalg = GPULinearAlgebra(context)
    
    def fit(self, X: np.ndarray, n_clusters: int, max_iter: int = 300, 
            tol: float = 1e-4, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """GPU K-means clustering."""
        
        # Transfer data to GPU
        X_gpu = self.context.to_gpu(X)
        n_samples, n_features = X.shape
        
        # Initialize centers using K-means++
        centers_gpu = self._kmeans_plus_plus_init(X_gpu, n_clusters, random_state)
        
        # Main K-means loop
        for iteration in range(max_iter):
            # Compute distances
            distances = self._compute_distances(X_gpu, centers_gpu)
            
            # Assign labels
            labels = self._assign_labels(distances)
            
            # Update centers
            new_centers = self._update_centers(X_gpu, labels, n_clusters)
            
            # Check convergence
            center_shift = self.linalg.norm(new_centers - centers_gpu)
            if self.context.to_cpu(center_shift) < tol:
                break
            
            centers_gpu = new_centers
        
        # Transfer results back to CPU
        centers = self.context.to_cpu(centers_gpu)
        labels = self.context.to_cpu(labels)
        
        return centers, labels
    
    def _kmeans_plus_plus_init(self, X_gpu: Any, n_clusters: int, 
                              random_state: Optional[int]) -> Any:
        """GPU K-means++ initialization."""
        if self.context.backend == 'cupy':
            return self._cupy_kmeans_plus_plus(X_gpu, n_clusters, random_state)
        elif self.context.backend == 'pytorch':
            return self._pytorch_kmeans_plus_plus(X_gpu, n_clusters, random_state)
    
    def _cupy_kmeans_plus_plus(self, X: Any, n_clusters: int, random_state: Optional[int]) -> Any:
        """CuPy implementation of K-means++."""
        n_samples, n_features = X.shape
        centers = cp.empty((n_clusters, n_features), dtype=X.dtype)
        
        # Set random seed
        if random_state is not None:
            cp.random.seed(random_state)
        
        # Choose first center randomly
        first_idx = cp.random.randint(0, n_samples)
        centers[0] = X[first_idx]
        
        # Choose remaining centers
        for c in range(1, n_clusters):
            # Compute distances to nearest center
            distances = cp.full(n_samples, cp.inf, dtype=X.dtype)
            for i in range(c):
                dist = cp.sum((X - centers[i]) ** 2, axis=1)
                distances = cp.minimum(distances, dist)
            
            # Choose next center with probability proportional to squared distance
            probabilities = distances / cp.sum(distances)
            cumulative_probs = cp.cumsum(probabilities)
            r = cp.random.random()
            next_idx = cp.searchsorted(cumulative_probs, r)
            centers[c] = X[next_idx]
        
        return centers
    
    def _pytorch_kmeans_plus_plus(self, X: Any, n_clusters: int, random_state: Optional[int]) -> Any:
        """PyTorch implementation of K-means++."""
        n_samples, n_features = X.shape
        centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device)
        
        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
        
        # Choose first center randomly
        first_idx = torch.randint(0, n_samples, (1,))
        centers[0] = X[first_idx]
        
        # Choose remaining centers
        for c in range(1, n_clusters):
            # Compute distances to nearest center
            distances = torch.full((n_samples,), float('inf'), dtype=X.dtype, device=X.device)
            for i in range(c):
                dist = torch.sum((X - centers[i]) ** 2, dim=1)
                distances = torch.minimum(distances, dist)
            
            # Choose next center with probability proportional to squared distance
            probabilities = distances / torch.sum(distances)
            next_idx = torch.multinomial(probabilities, 1)
            centers[c] = X[next_idx]
        
        return centers
    
    def _compute_distances(self, X: Any, centers: Any) -> Any:
        """Compute distances between samples and centers."""
        if self.context.backend == 'cupy':
            # Broadcasting approach for CuPy
            X_expanded = X[:, cp.newaxis, :]  # (n_samples, 1, n_features)
            centers_expanded = centers[cp.newaxis, :, :]  # (1, n_clusters, n_features)
            return cp.sum((X_expanded - centers_expanded) ** 2, axis=2)
        
        elif self.context.backend == 'pytorch':
            # Broadcasting approach for PyTorch
            X_expanded = X.unsqueeze(1)  # (n_samples, 1, n_features)
            centers_expanded = centers.unsqueeze(0)  # (1, n_clusters, n_features)
            return torch.sum((X_expanded - centers_expanded) ** 2, dim=2)
    
    def _assign_labels(self, distances: Any) -> Any:
        """Assign samples to nearest centers."""
        if self.context.backend == 'cupy':
            return cp.argmin(distances, axis=1)
        elif self.context.backend == 'pytorch':
            return torch.argmin(distances, dim=1)
    
    def _update_centers(self, X: Any, labels: Any, n_clusters: int) -> Any:
        """Update cluster centers."""
        if self.context.backend == 'cupy':
            n_features = X.shape[1]
            new_centers = cp.zeros((n_clusters, n_features), dtype=X.dtype)
            
            for k in range(n_clusters):
                mask = labels == k
                if cp.sum(mask) > 0:
                    new_centers[k] = cp.mean(X[mask], axis=0)
            
            return new_centers
        
        elif self.context.backend == 'pytorch':
            n_features = X.shape[1]
            new_centers = torch.zeros((n_clusters, n_features), dtype=X.dtype, device=X.device)
            
            for k in range(n_clusters):
                mask = labels == k
                if torch.sum(mask) > 0:
                    new_centers[k] = torch.mean(X[mask], dim=0)
            
            return new_centers


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPU resources."""
    info = {
        'backends': list(_gpu_backends.keys()),
        'default_backend': _default_backend,
        'devices': {}
    }
    
    if 'cupy' in _gpu_backends:
        try:
            info['devices']['cuda'] = {
                'count': cp.cuda.runtime.getDeviceCount(),
                'current': cp.cuda.Device().id,
                'memory': cp.cuda.MemoryInfo()
            }
        except Exception as e:
            info['devices']['cuda'] = {'error': str(e)}
    
    if 'pytorch' in _gpu_backends:
        try:
            if torch.cuda.is_available():
                info['devices']['pytorch_cuda'] = {
                    'count': torch.cuda.device_count(),
                    'current': torch.cuda.current_device(),
                    'memory': torch.cuda.get_device_properties(0)
                }
        except Exception as e:
            info['devices']['pytorch_cuda'] = {'error': str(e)}
    
    return info


def is_gpu_available(backend: Optional[str] = None) -> bool:
    """Check if GPU acceleration is available."""
    if backend is None:
        return _default_backend is not None
    return backend in _gpu_backends
