#pragma once
#include <cuda_runtime.h>
#include "rxmesh/util/macros.h"

namespace rxmesh {
__global__ static void get_cude_arch_k(int* d_arch)
{

#if defined(__CUDA_ARCH__)
    *d_arch = __CUDA_ARCH__;
#else
    *d_arch = 0;
#endif
}

inline int cuda_arch()
{
    int* d_arch = 0;
    CUDA_ERROR(cudaMalloc((void**)&d_arch, sizeof(int)));
    get_cude_arch_k<<<1, 1>>>(d_arch);
    int h_arch = 0;
    CUDA_ERROR(
        cudaMemcpy(&h_arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_arch);
    return h_arch;
}
}  // namespace rxmesh