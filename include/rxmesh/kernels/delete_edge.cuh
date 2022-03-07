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
    const uint16_t num_owned_edges = patch_info.num_owned_edges;
    const uint16_t num_edges       = patch_info.num_edges;

    // shared memory used to store EV and FE
    const uint32_t             mask_size = DIVIDE_UP(num_edges, 32);
    extern __shared__ uint32_t shrd_mem32[];
    extern __shared__ uint16_t shrd_mem[];
    uint32_t*                  s_mask_e = shrd_mem32;
    uint16_t*                  s_ev     = &shrd_mem[2 * mask_size];
    uint16_t*                  s_fe     = s_ev;

    // load Fe and EV and edges mask into shared memory
    load_async(patch_info.mask_e, mask_size, s_mask_e, false);

    //TODO we only need to load s_ev, operate on it, then load s_fe. 
    load_mesh_async<Op::FV>(patch_info, s_ev, s_fe, false);
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

        // update the face's bit mask. This function should be called by the
        // whole warp
        warp_update_mask(to_delete, local_id, s_mask_e);

        local_id += blockThreads;
    }
}
}  // namespace detail
}  // namespace rxmesh