#pragma once

#include "rxmesh/patch_info.h"

namespace rxmesh {
namespace detail {

__global__ static void free_patch_info(const uint32_t num_patches,
                                       PatchInfo*     patches_info)
{
    /*uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint32_t p_id = tid; p_id < num_patches;
         p_id += blockDim.x * gridDim.x) {

        cudaFree(patches_info[p_id].ev);
        cudaFree(patches_info[p_id].fe);
        cudaFree(patches_info[p_id].not_owned_patch_v);
        cudaFree(patches_info[p_id].not_owned_patch_e);
        cudaFree(patches_info[p_id].not_owned_patch_f);
        cudaFree(patches_info[p_id].not_owned_id_v);
        cudaFree(patches_info[p_id].not_owned_id_e);
        cudaFree(patches_info[p_id].not_owned_id_f);
    }*/
}

}  // namespace detail
}  // namespace rxmesh