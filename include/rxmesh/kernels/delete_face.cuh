#pragma once

#include "rxmesh/kernels/warp_update_mask.cuh"

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
    const uint16_t num_owned_faces = patch_info.num_owned_faces;

    // shared memory used to store FE
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_fe = shrd_mem;

    // load FE and make sure to sync before
    load_mesh_async<Op::FE>(patch_info, s_fe, s_fe, true);
    //__syncthreads();

    // load over all face-one thread per face
    uint16_t local_id = threadIdx.x;

    // we need to make sure that the whole warp go into the loop
    uint16_t len = round_to_next_multiple_32(num_owned_faces);

    while (local_id < len) {
        // check if face with local_id should be deleted and make sure we only
        // check owned faces
        bool to_delete = false;
        if (local_id < num_owned_faces) {
            to_delete = predicate({patch_info.patch_id, local_id});
        }

        // update the face's bit mask. This function should be called by the
        // whole warp
        warp_update_mask(to_delete, local_id, patch_info.mask_f);

        local_id += blockThreads;
    }

    __syncthreads();
    store<blockThreads>(s_fe,
                        patch_info.num_faces * 3,
                        reinterpret_cast<uint16_t*>(patch_info.fe));

    __syncthreads();
    
}
}  // namespace detail
}  // namespace rxmesh