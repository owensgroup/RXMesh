#pragma once

namespace rxmesh {
struct ShmemMutex
{
#ifdef __CUDA_ARCH__

#if (__CUDA_ARCH__ < 700)
#error ShmemMutex requires compiling with sm70 or higher since it relies on Independent Thread Scheduling
#endif
#endif

    __device__             ShmemMutex(const ShmemMutex& other) = default;
    __device__             ShmemMutex(ShmemMutex&&)            = default;
    __device__ ShmemMutex& operator=(const ShmemMutex&)        = default;
    __device__ ShmemMutex& operator=(ShmemMutex&&)             = default;
    __device__ ~ShmemMutex()                                   = default;

    __device__ ShmemMutex() : m_mutex(nullptr)
    {
    }


    __device__ __inline__ void alloc()
    {
#ifdef __CUDA_ARCH__
        __shared__ int s_mutex[1];
        m_mutex = s_mutex;
        if (threadIdx.x == 0) {
            m_mutex[0] = 0;
        }
#endif
    }

    __device__ __inline__ void lock()
    {
#ifdef __CUDA_ARCH__
        assert(m_mutex);
        __threadfence();
        while (::atomicCAS(m_mutex, 0, 1) != 0) {
            __threadfence();
        }        
        __threadfence();
#endif
    }


    __device__ __inline__ void unlock()
    {
#ifdef __CUDA_ARCH__
        assert(m_mutex);
        __threadfence();
        ::atomicExch(m_mutex, 0);
        __threadfence();
#endif
    }

    int* m_mutex;
};
}  // namespace rxmesh
