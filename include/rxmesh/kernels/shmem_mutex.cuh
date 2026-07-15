#pragma once

#include <cassert>
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#include "rxmesh/util/cuda_to_hip.h"
#endif

namespace rxmesh {
struct ShmemMutex
{
// The sm70 requirement is a CUDA-only ISA check (Independent Thread Scheduling);
// on HIP/CDNA the equivalent atomics are available, so do not gate on it there.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#error ShmemMutex requires compiling with sm70 or higher since it relies on Independent Thread Scheduling
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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        __shared__ int s_mutex[1];
        m_mutex = s_mutex;
        if (threadIdx.x == 0) {
            m_mutex[0] = 0;
        }
        // Ensure the 0-init is visible to every thread before any lock()/unlock()
        // (HIP does not guarantee it otherwise; a stale mutex word deadlocks).
        __syncthreads();
#endif
    }

    __device__ __inline__ void lock()
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
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
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        assert(m_mutex);
        __threadfence();
        ::atomicExch(m_mutex, 0);
        __threadfence();
#endif
    }

    // Run a critical section under the mutex. On CUDA (Volta+ Independent
    // Thread Scheduling) the plain lock()/unlock() bracket is forward-progress
    // safe even when several lanes of the same warp contend. AMD CDNA has no
    // per-lane forward-progress guarantee within a wavefront, so any spin-lock
    // shared by multiple lanes of the same wavefront deadlocks: a lane that
    // wins the CAS cannot reach the release because it is stuck at SIMT
    // reconvergence waiting on sibling lanes still spinning to acquire (and
    // folding the body into the acquire loop does not help -- the compiler
    // peels acquiring lanes off the exec mask and the lock stays held). The
    // fix is to serialize the wavefront's contending lanes: elect one active
    // lane at a time, let it fully acquire/run/release, then move on. Cross-
    // wavefront contention on the shared word is fine (wavefronts schedule
    // independently); only intra-wavefront contention is the hazard.
    template <typename FuncT>
    __device__ __inline__ void critical_section(FuncT&& func)
    {
#if defined(__HIP_DEVICE_COMPILE__)
        assert(m_mutex);
        const int          lane = __lane_id();
        unsigned long long need = __ballot(1);
        while (need) {
            const int leader = __ffsll(static_cast<long long>(need)) - 1;
            if (lane == leader) {
                while (::atomicCAS(m_mutex, 0, 1) != 0) {
                }
                __threadfence();
                func();
                __threadfence();
                ::atomicExch(m_mutex, 0);
            }
            need &= ~(1ull << leader);
        }
#else
        lock();
        func();
        unlock();
#endif
    }

    int* m_mutex;
};
}  // namespace rxmesh
