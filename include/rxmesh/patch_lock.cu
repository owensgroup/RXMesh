#include "rxmesh/patch_lock.h"

#include "rxmesh/kernels/util.cuh"
#include "rxmesh/util/macros.h"

namespace rxmesh {
__host__ void PatchLock::init()
{
    CUDA_ERROR(cudaMalloc((void**)&lock, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&spin, sizeof(uint32_t)));
    uint32_t h_lock = FREE, h_spin = INVALID32;
    CUDA_ERROR(
        cudaMemcpy(lock, &h_lock, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(
        cudaMemcpy(spin, &h_spin, sizeof(uint32_t), cudaMemcpyHostToDevice));
}

__host__ void PatchLock::free()
{
    GPU_FREE(lock);
    GPU_FREE(spin);
}

}  // namespace rxmesh
