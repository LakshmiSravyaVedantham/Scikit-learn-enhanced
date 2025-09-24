# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""
High-performance memory pool allocator for scikit-learn.

This module provides optimized memory allocation strategies to reduce
allocation overhead and improve cache performance.
"""

from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memset, memcpy
from libc.stdint cimport uintptr_t
import numpy as np
cimport numpy as cnp
cnp.import_array()

from ..utils._typedefs cimport intp_t, float64_t, float32_t

# Memory alignment for SIMD operations
cdef enum:
    MEMORY_ALIGNMENT = 64  # 64-byte alignment for AVX-512
    POOL_BLOCK_SIZE = 1024 * 1024  # 1MB blocks
    MAX_POOLS = 16

# Memory pool structure
cdef struct MemoryBlock:
    void* ptr
    size_t size
    size_t used
    MemoryBlock* next

cdef struct MemoryPool:
    MemoryBlock* blocks
    size_t block_size
    size_t total_allocated
    size_t total_used
    int pool_id

# Global memory pools for different allocation sizes
cdef MemoryPool* global_pools[MAX_POOLS]
cdef int num_pools = 0
cdef bint pools_initialized = False

cdef inline void* aligned_malloc(size_t size, size_t alignment) noexcept nogil:
    """Allocate aligned memory."""
    cdef void* ptr
    cdef uintptr_t addr
    
    # Allocate extra space for alignment
    ptr = malloc(size + alignment - 1 + sizeof(void*))
    if ptr == NULL:
        return NULL
    
    # Calculate aligned address
    addr = <uintptr_t>ptr + sizeof(void*) + alignment - 1
    addr = addr & ~(alignment - 1)
    
    # Store original pointer for freeing
    (<void**>addr)[-1] = ptr
    
    return <void*>addr

cdef inline void aligned_free(void* ptr) noexcept nogil:
    """Free aligned memory."""
    if ptr != NULL:
        free((<void**>ptr)[-1])

cdef inline MemoryPool* create_memory_pool(size_t block_size) noexcept nogil:
    """Create a new memory pool."""
    cdef MemoryPool* pool = <MemoryPool*>malloc(sizeof(MemoryPool))
    if pool == NULL:
        return NULL
    
    pool.blocks = NULL
    pool.block_size = block_size
    pool.total_allocated = 0
    pool.total_used = 0
    pool.pool_id = num_pools
    
    return pool

cdef inline MemoryBlock* create_memory_block(size_t size) noexcept nogil:
    """Create a new memory block."""
    cdef MemoryBlock* block = <MemoryBlock*>malloc(sizeof(MemoryBlock))
    if block == NULL:
        return NULL
    
    block.ptr = aligned_malloc(size, MEMORY_ALIGNMENT)
    if block.ptr == NULL:
        free(block)
        return NULL
    
    block.size = size
    block.used = 0
    block.next = NULL
    
    return block

cdef inline void* pool_alloc(MemoryPool* pool, size_t size) noexcept nogil:
    """Allocate memory from pool."""
    cdef MemoryBlock* block = pool.blocks
    cdef MemoryBlock* new_block
    cdef void* ptr
    
    # Find a block with enough space
    while block != NULL:
        if block.size - block.used >= size:
            ptr = <char*>block.ptr + block.used
            block.used += size
            pool.total_used += size
            return ptr
        block = block.next
    
    # Need a new block
    cdef size_t block_size = max(pool.block_size, size)
    new_block = create_memory_block(block_size)
    if new_block == NULL:
        return NULL
    
    # Add to front of list
    new_block.next = pool.blocks
    pool.blocks = new_block
    pool.total_allocated += block_size
    
    # Allocate from new block
    ptr = new_block.ptr
    new_block.used = size
    pool.total_used += size
    
    return ptr

cdef inline void destroy_memory_pool(MemoryPool* pool) noexcept nogil:
    """Destroy memory pool and free all blocks."""
    cdef MemoryBlock* block = pool.blocks
    cdef MemoryBlock* next_block
    
    while block != NULL:
        next_block = block.next
        aligned_free(block.ptr)
        free(block)
        block = next_block
    
    free(pool)

cdef inline void reset_memory_pool(MemoryPool* pool) noexcept nogil:
    """Reset pool for reuse without freeing blocks."""
    cdef MemoryBlock* block = pool.blocks
    
    while block != NULL:
        block.used = 0
        block = block.next
    
    pool.total_used = 0

def init_memory_pools():
    """Initialize global memory pools."""
    global pools_initialized, num_pools
    
    if pools_initialized:
        return
    
    # Create pools for common allocation sizes
    cdef size_t[8] pool_sizes = [
        1024,      # 1KB - small allocations
        8192,      # 8KB - medium allocations  
        65536,     # 64KB - large allocations
        262144,    # 256KB - very large allocations
        1048576,   # 1MB - huge allocations
        4194304,   # 4MB - massive allocations
        16777216,  # 16MB - enormous allocations
        67108864   # 64MB - gigantic allocations
    ]
    
    cdef int i
    for i in range(8):
        global_pools[i] = create_memory_pool(pool_sizes[i])
        if global_pools[i] == NULL:
            # Cleanup on failure
            for j in range(i):
                destroy_memory_pool(global_pools[j])
            return
    
    num_pools = 8
    pools_initialized = True

def cleanup_memory_pools():
    """Cleanup all memory pools."""
    global pools_initialized, num_pools
    
    if not pools_initialized:
        return
    
    cdef int i
    for i in range(num_pools):
        destroy_memory_pool(global_pools[i])
        global_pools[i] = NULL
    
    num_pools = 0
    pools_initialized = False

cdef inline MemoryPool* get_pool_for_size(size_t size) noexcept nogil:
    """Get appropriate pool for allocation size."""
    cdef int i
    
    for i in range(num_pools):
        if global_pools[i].block_size >= size:
            return global_pools[i]
    
    # Use largest pool for oversized allocations
    return global_pools[num_pools - 1]

cdef inline void* fast_alloc(size_t size) noexcept nogil:
    """Fast allocation using memory pools."""
    if not pools_initialized:
        return aligned_malloc(size, MEMORY_ALIGNMENT)
    
    cdef MemoryPool* pool = get_pool_for_size(size)
    return pool_alloc(pool, size)

cdef inline void fast_free(void* ptr) noexcept nogil:
    """Fast deallocation (no-op for pool allocations)."""
    # Pool memory is reused, so no need to free individual allocations
    pass

# Python interface
def get_memory_stats():
    """Get memory pool statistics."""
    if not pools_initialized:
        return {}
    
    stats = {}
    cdef int i
    cdef MemoryPool* pool
    
    for i in range(num_pools):
        pool = global_pools[i]
        stats[f"pool_{i}"] = {
            "block_size": pool.block_size,
            "total_allocated": pool.total_allocated,
            "total_used": pool.total_used,
            "utilization": pool.total_used / max(pool.total_allocated, 1)
        }
    
    return stats

def reset_all_pools():
    """Reset all memory pools for reuse."""
    if not pools_initialized:
        return
    
    cdef int i
    for i in range(num_pools):
        reset_memory_pool(global_pools[i])
