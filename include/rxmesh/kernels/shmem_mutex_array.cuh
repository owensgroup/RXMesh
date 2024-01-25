#pragma once

namespace rxmesh {
struct ShmemMutexArray
{
#ifdef __CUDA_ARCH__

#if (__CUDA_ARCH__ < 700)
#error ShmemMutexArray requires compiling with sm70 or higher since it relies on Independent Thread Scheduling
#endif
#endif

    __device__ ShmemMutexArray()                                  = default;
    __device__ ShmemMutexArray(const ShmemMutexArray& other)      = default;
    __device__ ShmemMutexArray(ShmemMutexArray&&)                 = default;
    __device__ ShmemMutexArray& operator=(const ShmemMutexArray&) = default;
    __device__ ShmemMutexArray& operator=(ShmemMutexArray&&)      = default;
    __device__ ~ShmemMutexArray()                                 = default;

    /**
     * @brief s_array is a pre-allocated device memory with the give size
     */
    __device__ ShmemMutexArray(cooperative_groups::thread_block& block,
                               int                               size,
                               int*                              s_array)
        : m_size(size), m_mutex(s_array)
    {
        for (int i = threadIdx.x; i < size; i += block.size()) {
            s_array[i] = 0;
        }
    }

    __device__ __inline__ void lock(int loc)
    {
#ifdef __CUDA_ARCH__
        assert(m_mutex);
        while (::atomicCAS(m_mutex + loc, 0, 1) != 0) {
        }
        __threadfence();
#endif
    }


    __device__ __inline__ void unlock(int loc)
    {
#ifdef __CUDA_ARCH__
        assert(m_mutex);
        __threadfence();
        ::atomicExch(m_mutex + loc, 0);
#endif
    }

    int  m_size;
    int* m_mutex;
};
}  // namespace rxmesh
