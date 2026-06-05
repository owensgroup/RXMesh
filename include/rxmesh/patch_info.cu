#include "rxmesh/patch_info.h"

#include <cassert>
#include <type_traits>
#include <utility>



namespace rxmesh {

// Explicit instantiations
// The instantiated PatchInfo members are __host__ __device__. clang requires the
// explicit instantiation to carry the same target attributes (nvcc does not).
#define PATCH_INFO_INSTANTIATE(HandleT)                                          \
    template __host__ __device__ HandleT PatchInfo::find<HandleT>(               \
        const LPPair::KeyT, const LPPair*, const LPPair*, const PatchStash&)     \
        const;                                                                   \
    template __host__ __device__ HandleT PatchInfo::find<HandleT>(               \
        const LPPair::KeyT, const LPPair*, const LPPair*) const;                 \
    template __host__ __device__ HandleT                                         \
    PatchInfo::get_handle<HandleT>(const LPPair) const;                          \
    template __host__ __device__ const uint16_t*                                 \
    PatchInfo::get_num_elements<HandleT>() const;                                \
    template __host__ __device__ uint16_t                                        \
    PatchInfo::get_capacity<HandleT>() const;                                    \
    template __host__ __device__ const uint32_t*                                 \
    PatchInfo::get_active_mask<HandleT>() const;                                 \
    template __host__ __device__ uint32_t*                                       \
    PatchInfo::get_active_mask<HandleT>();                                       \
    template __host__ __device__ const uint32_t*                                 \
    PatchInfo::get_owned_mask<HandleT>() const;                                  \
    template __host__ __device__ uint32_t*                                       \
    PatchInfo::get_owned_mask<HandleT>();                                        \
    template __host__ __device__ const LPHashTable&                             \
    PatchInfo::get_lp<HandleT>() const;                                          \
    template __host__ __device__ LPHashTable& PatchInfo::get_lp<HandleT>();      \
    template __host__ __device__ uint16_t                                        \
    PatchInfo::get_num_owned<HandleT>() const;

PATCH_INFO_INSTANTIATE(VertexHandle)
PATCH_INFO_INSTANTIATE(EdgeHandle)
PATCH_INFO_INSTANTIATE(FaceHandle)

#undef PATCH_INFO_INSTANTIATE

}  // namespace rxmesh
