#pragma once
#include "rxmesh/patch_info.h"
#include "rxmesh/util/bitmask_util.h"
namespace rxmesh {
namespace detail {
template <typename LambdaT>
__device__ __inline__ void for_each_vertex_kernel(const uint32_t   num_patches,
                                                  const PatchInfo* patch_info,
                                                  LambdaT          apply)
{
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t num_v = patch_info[p_id].num_vertices[0];
        for (uint16_t v = threadIdx.x; v < num_v; v += blockDim.x) {
            if (detail::is_owned(v, patch_info[p_id].owned_mask_v) &&
                !detail::is_deleted(v, patch_info[p_id].active_mask_v)) {
                VertexHandle v_handle(p_id, v);
                apply(v_handle);
            }
        }
    }
}
template <typename LambdaT>
__global__ void for_each_vertex(const uint32_t   num_patches,
                                const PatchInfo* patch_info,
                                LambdaT          apply)
{
    for_each_vertex_kernel(num_patches, patch_info, apply);
}


template <typename LambdaT>
__device__ __inline__ void for_each_edge_kernel(const uint32_t   num_patches,
                                                const PatchInfo* patch_info,
                                                LambdaT          apply)
{
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t num_e = patch_info[p_id].num_edges[0];
        for (uint16_t e = threadIdx.x; e < num_e; e += blockDim.x) {
            if (detail::is_owned(e, patch_info[p_id].owned_mask_e) &&
                !detail::is_deleted(e, patch_info[p_id].active_mask_e)) {
                EdgeHandle e_handle(p_id, e);
                apply(e_handle);
            }
        }
    }
}
template <typename LambdaT>
__global__ void for_each_edge(const uint32_t   num_patches,
                              const PatchInfo* patch_info,
                              LambdaT          apply)
{
    for_each_edge_kernel(num_patches, patch_info, apply);
}



template <typename LambdaT>
__device__ __inline__ void for_each_face_kernel(const uint32_t   num_patches,
                                                const PatchInfo* patch_info,
                                                LambdaT          apply)
{
    const uint32_t p_id = blockIdx.x;
    if (p_id < num_patches) {
        const uint16_t num_f = patch_info[p_id].num_faces[0];
        for (uint16_t f = threadIdx.x; f < num_f; f += blockDim.x) {
            if (detail::is_owned(f, patch_info[p_id].owned_mask_f) &&
                !detail::is_deleted(f, patch_info[p_id].active_mask_f)) {
                FaceHandle f_handle(p_id, f);
                apply(f_handle);
            }
        }
    }
}
template <typename LambdaT>
__global__ void for_each_face(const uint32_t   num_patches,
                              const PatchInfo* patch_info,
                              LambdaT          apply)
{
    for_each_face_kernel(num_patches, patch_info, apply);
}

}  // namespace detail
}  // namespace rxmesh