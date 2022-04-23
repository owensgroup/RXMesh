#pragma once

#include "rxmesh/kernels/dynamic_util.cuh"
#include "rxmesh/kernels/is_deleted.cuh"
#include "rxmesh/kernels/warp_update_mask.cuh"

namespace rxmesh {
namespace detail {
/**
 * @brief delete edge
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void delete_edge(PatchInfo&       patch_info,
                                       const predicateT predicate)
{
    // Extract the argument in the predicate lambda function
    using PredicateTTraits = detail::FunctionTraits<predicateT>;
    using HandleT          = typename PredicateTTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, EdgeHandle>,
        "First argument in predicate lambda function should be EdgeHandle");

    // patch basic info
    const uint16_t num_owned_faces = patch_info.num_owned_faces;
    const uint16_t num_owned_edges = patch_info.num_owned_edges;
    const uint16_t num_edges       = patch_info.num_edges;

    // shared memory used to store EV and FE
    const uint16_t             mask_size = DIVIDE_UP(num_edges, 32);
    extern __shared__ uint32_t shrd_mem32[];
    extern __shared__ uint16_t shrd_mem[];
    uint32_t*                  s_mask_e = shrd_mem32;
    uint16_t*                  s_fe     = &shrd_mem[2 * mask_size];

    // load edges mask into shared memory
    load_async(patch_info.mask_e, mask_size, s_mask_e, true);
    __syncthreads();

    // start loading FE without sync
    load_mesh_async<Op::FE>(patch_info, s_fe, s_fe, false);

    // update the bitmask based on user-defined predicate
    update_bitmask<blockThreads>(
        num_owned_edges, s_mask_e, [&](const uint16_t local_e) {
            return predicate({patch_info.patch_id, local_e});
        });
    __syncthreads();

    // store edge mask into global memory
    store<blockThreads>(s_mask_e, mask_size, patch_info.mask_e);

    // delete faces incident to deleted edges
    // we only delete faces owned by the patch
    // we need to make sure that the whole warp go into this loop
    // TODO here we read the face bitmask from global memory to check if the
    // face is already deleted and then update the mask to the global memory.
    // While the memory access is coalesced, the whole warp read only 4 bytes.
    // We may (or may not) get better performance if we read the bitmask into
    // shared memory, operate on it, then store it to global memory
    update_bitmask<blockThreads>(
        num_owned_faces, patch_info.mask_f, [&](const uint16_t local_f) {
            const uint16_t e0 = s_fe[3 * local_f + 0] >> 1;
            const uint16_t e1 = s_fe[3 * local_f + 1] >> 1;
            const uint16_t e2 = s_fe[3 * local_f + 2] >> 1;

            return is_deleted(e0, s_mask_e) || is_deleted(e1, s_mask_e) ||
                   is_deleted(e2, s_mask_e);
        });
    __syncthreads();
}
}  // namespace detail
}  // namespace rxmesh