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
    uint16_t*                  s_ev     = &shrd_mem[2 * mask_size];
    uint16_t*                  s_fe     = s_ev;

    // load edges mask into shared memory
    load_async(patch_info.mask_e, mask_size, s_mask_e, false);

    // we only need to load s_ev, operate on it, then load s_fe
    load_mesh_async<Op::EV>(patch_info, s_ev, s_fe, false);
    __syncthreads();

    // load over all edges---one thread per edge
    uint16_t local_id = threadIdx.x;

    // we need to make sure that the whole warp go into the loop
    uint16_t len = round_to_next_multiple_32(num_owned_edges);

    while (local_id < len) {

        // check if edge with local_id should be deleted and make sure we only
        // check owned edges
        bool to_delete = false;

        if (local_id < num_owned_edges) {
            to_delete = predicate({patch_info.patch_id, local_id});
        }

        // reset the connectivity of deleted edge
        if (to_delete) {
            s_ev[2 * local_id + 0] = INVALID16;
            s_ev[2 * local_id + 1] = INVALID16;
        }

        // update the edge's bit mask. This function should be called by the
        // whole warp
        warp_update_mask(to_delete, local_id, s_mask_e);

        local_id += blockThreads;
    }

    // we can now store EV to global memory
    __syncthreads();
    store<blockThreads>(s_ev,
                        patch_info.num_edges * 2,
                        reinterpret_cast<uint16_t*>(patch_info.ev));

    //load FE and make sure we don't overwrite EV
    __syncthreads();
    load_mesh_async<Op::FE>(patch_info, s_ev, s_fe, false);
    __syncthreads();


    // delete faces incident to deleted edges
    // we only delete faces owned by the patch
    local_id = 0;
    // we need to make sure that the whole warp go into the loop
    len = round_to_next_multiple_32(num_owned_faces);
    while (local_id < len) {
        uint16_t e0 = s_fe[3 * local_id + 0] >> 1;
        uint16_t e1 = s_fe[3 * local_id + 1] >> 1;
        uint16_t e2 = s_fe[3 * local_id + 2] >> 1;

        bool to_delete = is_deleted(e0, s_mask_e) || is_deleted(e1, s_mask_e) ||
                         is_deleted(e2, s_mask_e);

        if (to_delete) {
            s_fe[3 * local_id + 0] = INVALID16;
            s_fe[3 * local_id + 1] = INVALID16;
            s_fe[3 * local_id + 2] = INVALID16;
        }

        // update the face's bit mask. This function should be called by the
        // whole warp
        warp_update_mask(to_delete, local_id, patch_info.mask_f);

        local_id += blockThreads;
    }
    __syncthreads();

    // store edge mask into global memory
    store<blockThreads>(s_mask_e, mask_size, patch_info.mask_e);

    // store FE in global memory
    store<blockThreads>(s_fe,
                        patch_info.num_faces * 3,
                        reinterpret_cast<uint16_t*>(patch_info.fe));
    __syncthreads();
}
}  // namespace detail
}  // namespace rxmesh