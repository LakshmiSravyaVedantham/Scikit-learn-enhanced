#!/usr/bin/env python3
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Enhanced setup script for scikit-learn with performance optimizations.

This setup script includes additional optimizations and features:
- SIMD vectorization support
- GPU acceleration capabilities
- Memory pool allocation
- Enhanced Cython compilation flags
- Performance profiling tools
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Compiler import Options

# Enable Cython optimizations
Options.docstrings = False
Options.embed_pos_in_docstring = False

# Detect system capabilities
def detect_system_capabilities():
    """Detect system capabilities for optimization."""
    capabilities = {
        'simd': False,
        'openmp': False,
        'cuda': False,
        'opencl': False,
        'avx2': False,
        'fma': False
    }
    
    # Check for SIMD support
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        flags = cpu_info.get('flags', [])
        
        capabilities['simd'] = any(flag in flags for flag in ['sse', 'sse2', 'avx'])
        capabilities['avx2'] = 'avx2' in flags
        capabilities['fma'] = 'fma' in flags
    except ImportError:
        # Fallback: assume modern CPU
        capabilities['simd'] = True
        capabilities['avx2'] = True
        capabilities['fma'] = True
    
    # Check for OpenMP
    try:
        result = subprocess.run(['gcc', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'],
                              input=b'int main(){return 0;}', 
                              capture_output=True)
        capabilities['openmp'] = result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        capabilities['openmp'] = False
    
    # Check for CUDA
    try:
        import cupy
        capabilities['cuda'] = True
    except ImportError:
        capabilities['cuda'] = False
    
    # Check for OpenCL
    try:
        import pyopencl
        capabilities['opencl'] = True
    except ImportError:
        capabilities['opencl'] = False
    
    return capabilities

def get_enhanced_compile_args():
    """Get enhanced compilation arguments."""
    capabilities = detect_system_capabilities()
    
    compile_args = [
        '-O3',  # Maximum optimization
        '-ffast-math',  # Fast math operations
        '-march=native',  # Optimize for current CPU
        '-mtune=native',
    ]
    
    # Add SIMD flags
    if capabilities['avx2']:
        compile_args.extend(['-mavx2', '-mfma'])
    elif capabilities['simd']:
        compile_args.extend(['-msse4.2'])
    
    # Add OpenMP support
    if capabilities['openmp']:
        compile_args.append('-fopenmp')
    
    # Platform-specific optimizations
    if platform.system() == 'Linux':
        compile_args.extend([
            '-fPIC',
            '-Wno-unused-function',
            '-Wno-unused-variable'
        ])
    elif platform.system() == 'Darwin':
        compile_args.extend([
            '-Wno-unused-function',
            '-Wno-unused-variable'
        ])
    
    return compile_args, capabilities

def get_enhanced_link_args():
    """Get enhanced linking arguments."""
    link_args = []
    
    # Add OpenMP linking
    if detect_system_capabilities()['openmp']:
        link_args.append('-fopenmp')
    
    # Add math library
    if platform.system() != 'Windows':
        link_args.append('-lm')
    
    return link_args

def create_enhanced_extensions():
    """Create enhanced Cython extensions."""
    compile_args, capabilities = get_enhanced_compile_args()
    link_args = get_enhanced_link_args()
    
    # Define macros based on capabilities
    define_macros = [
        ('NPY_NO_DEPRECATED_API', 'NPY_1_22_API_VERSION'),
    ]
    
    if capabilities['simd']:
        define_macros.append(('USE_SIMD', '1'))
    if capabilities['avx2']:
        define_macros.append(('USE_AVX2', '1'))
    if capabilities['fma']:
        define_macros.append(('USE_FMA', '1'))
    if capabilities['openmp']:
        define_macros.append(('USE_OPENMP', '1'))
    
    # Enhanced extensions
    extensions = [
        # Memory pool
        Extension(
            'sklearn.utils._memory_pool',
            ['sklearn/utils/_memory_pool.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=define_macros,
        ),
        
        # SIMD utilities
        Extension(
            'sklearn.utils._simd_utils',
            ['sklearn/utils/_simd_utils.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=define_macros,
        ),
        
        # Enhanced coordinate descent
        Extension(
            'sklearn.linear_model._cd_fast_enhanced',
            ['sklearn/linear_model/_cd_fast_enhanced.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=define_macros,
        ),
        
        # Enhanced tree
        Extension(
            'sklearn.tree._tree_enhanced',
            ['sklearn/tree/_tree_enhanced.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=define_macros,
        ),
        
        # Enhanced K-means
        Extension(
            'sklearn.cluster._kmeans_enhanced',
            ['sklearn/cluster/_kmeans_enhanced.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=define_macros,
        ),
    ]
    
    return extensions

def get_enhanced_requirements():
    """Get enhanced requirements including optional GPU dependencies."""
    base_requirements = [
        'numpy>=1.24.1',
        'scipy>=1.10.0',
        'joblib>=1.3.0',
        'threadpoolctl>=3.2.0',
    ]
    
    # Optional GPU requirements
    gpu_requirements = [
        'cupy-cuda12x; platform_system!="Darwin"',  # CUDA support
        'pyopencl; platform_system!="Windows"',     # OpenCL support
        'torch; python_version>="3.8"',             # PyTorch backend
    ]
    
    # Performance monitoring
    performance_requirements = [
        'psutil>=5.8.0',
        'memory_profiler>=0.60.0',
        'py-cpuinfo>=8.0.0',
    ]
    
    return {
        'base': base_requirements,
        'gpu': gpu_requirements,
        'performance': performance_requirements,
    }

def main():
    """Main setup function."""
    print("Setting up enhanced scikit-learn...")
    
    # Detect capabilities
    capabilities = detect_system_capabilities()
    print(f"Detected capabilities: {capabilities}")
    
    # Create extensions
    extensions = create_enhanced_extensions()
    
    # Cythonize with optimizations
    cythonized_extensions = cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
            'profile': False,
            'linetrace': False,
        },
        compile_time_env={
            'USE_SIMD': capabilities['simd'],
            'USE_AVX2': capabilities['avx2'],
            'USE_FMA': capabilities['fma'],
            'USE_OPENMP': capabilities['openmp'],
        }
    )
    
    # Get requirements
    requirements = get_enhanced_requirements()
    
    # Setup configuration
    setup(
        name='scikit-learn-enhanced',
        version='1.8.0-enhanced',
        description='Enhanced scikit-learn with performance optimizations',
        long_description=open('README_ENHANCED.md').read(),
        long_description_content_type='text/markdown',
        author='The scikit-learn developers (Enhanced)',
        author_email='scikit-learn@python.org',
        url='https://github.com/LakshmiSravya123/Scikit-learn-enhanced',
        license='BSD-3-Clause',
        packages=find_packages(),
        ext_modules=cythonized_extensions,
        install_requires=requirements['base'],
        extras_require={
            'gpu': requirements['gpu'],
            'performance': requirements['performance'],
            'all': requirements['gpu'] + requirements['performance'],
        },
        python_requires='>=3.10',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Programming Language :: Cython',
            'Programming Language :: C',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords='machine learning, scikit-learn, performance, optimization, gpu, simd',
        include_package_data=True,
        zip_safe=False,
    )

if __name__ == '__main__':
    main()
