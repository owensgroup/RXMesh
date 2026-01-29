#include "rxmesh/patch_lock.h"

#include "rxmesh/kernels/util.cuh"
#include "rxmesh/util/macros.h"

namespace rxmesh {

__device__ bool PatchLock::acquire_lock(uint32_t id)
{
#ifdef __CUDA_ARCH__
    int attempt = 0;
    while (::atomicCAS(lock, FREE, LOCKED) == LOCKED) {
        __threadfence();
        if (attempt == MAX_ATTEMPT) {
            int other = ::atomicMin(spin, id);
            __threadfence();
            if (other < id) {
                return false;
            }
            attempt = 0;
        }
        attempt++;
    }
    atomicExch(spin, id);
    return true;
#else
    return true;
#endif
}

__device__ void PatchLock::release_lock()
{
#ifdef __CUDA_ARCH__
    atomicExch(spin, INVALID32);
    atomicExch(lock, FREE);
    __threadfence();
#endif
}

__device__ bool PatchLock::is_locked() const
{
#ifdef __CUDA_ARCH__
    return atomic_read(lock) == LOCKED;
#else
    return false;
#endif
}

__host__ void PatchLock::init()
{
    CUDA_ERROR(cudaMalloc((void**)&lock, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&spin, sizeof(uint32_t)));
    uint32_t h_lock = FREE, h_spin = INVALID32;
    CUDA_ERROR(cudaMemcpy(
        lock, &h_lock, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(
        spin, &h_spin, sizeof(uint32_t), cudaMemcpyHostToDevice));
}

__host__ void PatchLock::free()
{
    GPU_FREE(lock);
    GPU_FREE(spin);
}

}  // namespace rxmesh
