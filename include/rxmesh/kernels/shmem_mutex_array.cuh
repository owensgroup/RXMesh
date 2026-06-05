#pragma once

#include <cassert>
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include "rxmesh/util/cuda_to_hip.h"
#endif

namespace rxmesh {
struct ShmemMutexArray
{
// The sm70 requirement is a CUDA-only ISA check (Independent Thread Scheduling);
// on HIP/CDNA the equivalent atomics are available, so do not gate on it there.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#error ShmemMutexArray requires compiling with sm70 or higher since it relies on Independent Thread Scheduling
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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        assert(m_mutex);
        while (::atomicCAS(m_mutex + loc, 0, 1) != 0) {
        }
        __threadfence();
#endif
    }


    __device__ __inline__ void unlock(int loc)
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        assert(m_mutex);
        __threadfence();
        ::atomicExch(m_mutex + loc, 0);
#endif
    }

    // See ShmemMutex::critical_section for why a plain spin-lock shared by
    // lanes of one wavefront deadlocks on AMD CDNA. Serialize the wavefront's
    // contending lanes (one leader at a time), each locking its own loc, so
    // there is never intra-wavefront contention on a shared mutex word.
    template <typename FuncT>
    __device__ __inline__ void critical_section(int loc, FuncT&& func)
    {
#if defined(__HIP_DEVICE_COMPILE__)
        assert(m_mutex);
        const int          lane = __lane_id();
        unsigned long long need = __ballot(1);
        while (need) {
            const int leader = __ffsll(static_cast<long long>(need)) - 1;
            if (lane == leader) {
                while (::atomicCAS(m_mutex + loc, 0, 1) != 0) {
                }
                __threadfence();
                func();
                __threadfence();
                ::atomicExch(m_mutex + loc, 0);
            }
            need &= ~(1ull << leader);
        }
#else
        lock(loc);
        func();
        unlock(loc);
#endif
    }

    int  m_size;
    int* m_mutex;
};
}  // namespace rxmesh
