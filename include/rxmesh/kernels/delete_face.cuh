#pragma once

#include "rxmesh/kernels/dynamic_util.cuh"

namespace rxmesh {
namespace detail {
/**
 * @brief delete face
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void delete_face(PatchInfo&       patch_info,
                                       const predicateT predicate)
{
    // Extract the argument in the predicate lambda function
    using PredicateTTraits = detail::FunctionTraits<predicateT>;
    using HandleT          = typename PredicateTTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, FaceHandle>,
        "First argument in predicate lambda function should be FaceHandle");

    // patch basic info
    const uint32_t patch_id        = patch_info.patch_id;
    const uint16_t num_owned_faces = patch_info.num_owned_faces;
    const uint16_t num_faces       = patch_info.num_faces;

    // load face's bitmask to shared memory
    const uint16_t             mask_size = DIVIDE_UP(num_faces, 32);
    extern __shared__ uint32_t shrd_mem32[];
    uint32_t*                  s_mask_f = shrd_mem32;

    load_async(patch_info.mask_f, mask_size, s_mask_f, true);
    __syncthreads();

    // update the bitmask based on user-defined predicate
    update_bitmask<blockThreads>(
        num_owned_faces, s_mask_f, [&](const uint16_t local_f) {
            return predicate({patch_id, local_f});
        });
    __syncthreads();

    // store the bitmask back to global memory
    store<blockThreads>(s_mask_f, mask_size, patch_info.mask_f);
    __syncthreads();
}
}  // namespace detail
}  // namespace rxmesh