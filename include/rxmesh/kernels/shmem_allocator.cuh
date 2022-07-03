#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

namespace rxmesh {
namespace detail {

extern __shared__ char SHMEM_START[];

/**
 * @brief Shared memory allocator that makes it easy to allocate different
 * segments of the shared memory of different types.
 */
struct ShmemAllocator
{
    static constexpr uint32_t default_alignment = 8;

    __device__ ShmemAllocator() : m_ptr(SHMEM_START)
    {
    }

    /**
     * @brief deallocate by subtracting number of bytes from the base pointer
     * This is NOT true deallocation. This can only be used by deallocating the
     * last allocated num_bytes to avoid segmentation and overlap
     * @param num_bytes number of bytes to be deallocated
     */
    __device__ __forceinline__ void dealloc(const uint32_t num_bytes)
    {
        m_ptr = m_ptr - num_bytes;
        assert(m_ptr - SHMEM_START > 0);
    }

    /**
     * @brief Typed deallocation
     * @param count number of elements to be deallocated
     */
    template <typename T>
    __device__ __forceinline__ void dealloc(uint32_t count)
    {
        dealloc(count * sizeof(T));
    }

    /**
     * @brief Allocate num_bytes and return a pointer to the start of the
     * allocation. The returned pointer is aligned to bytes_alignment.
     * This function could be called by all threads if ShmemAllocator is in the
     * register. If ShmemAllocator is declared as __shared__, only one thread
     * per block should call this function.
     * @param num_bytes to allocate
     * @param byte_alignment alignment size
     */
    __device__ __forceinline__ char* alloc(
        uint32_t num_bytes,
        uint32_t byte_alignment = default_alignment)
    {
        this->align(byte_alignment);
        char* ret = m_ptr;
        m_ptr     = m_ptr + num_bytes;
        assert(get_allocated_size_bytes() <= get_max_size_bytes());
        return ret;
    }

    /**
     * @brief a typed version of alloc() where the input number of elements (not
     * number of bytes).
     * @tparam T type of the pointer
     * @param count number of elements to be allocated
     * @param byte_alignment alignment size
     */
    template <typename T>
    __device__ __forceinline__ T* alloc(uint32_t count)
    {
        static_assert(sizeof(T) <= default_alignment,
                      "default_alignment is less than the alignment boundary "
                      "of the input type");
        return reinterpret_cast<T*>(
            alloc(count * sizeof(T), default_alignment));
    }

    /**
     * @brief return the maximum allocation size which is the same as the number
     * of bytes passed during the kernel launch
     */
    __device__ __forceinline__ uint32_t get_max_size_bytes() const
    {
        uint32_t ret;
        asm("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
        return ret;
    }

    /**
     * @brief return the number of bytes that has been allocated
     */
    __device__ __forceinline__ uint32_t get_allocated_size_bytes() const
    {
        return m_ptr - SHMEM_START;
    }

   private:
    /**
     * @brief This function aligns m_ptr to the first location at the
     * boundary of a given alignment size. This what std:align does but it does
     * not work with CUDA so this a stripped down version of it.
     * @param byte_alignment number of bytes to get the pointer to be aligned to
     */
    __device__ __host__ __inline__ void align(
        const uint32_t byte_alignment) noexcept
    {
        const uint64_t intptr    = reinterpret_cast<uint64_t>(m_ptr);
        const uint64_t remainder = intptr % byte_alignment;
        if (remainder == 0) {
            return;
        }
        const uint64_t aligned = intptr + byte_alignment - remainder;
        m_ptr                  = reinterpret_cast<char*>(aligned);
    }
    char* m_ptr;
};
}  // namespace detail
}  // namespace rxmesh