namespace rxmesh {

namespace detail {
/**
 * @brief delete edge
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void delete_vertex(PatchInfo&       patch_info,
                                         const predicateT predicate)
{
    // Extract the argument in the predicate lambda function
    using PredicateTTraits = detail::FunctionTraits<predicateT>;
    using HandleT          = typename PredicateTTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, VertexHandle>,
        "First argument in predicate lambda function should be vertexHandle");


    // patch basic info
    const uint16_t num_owned_faces    = patch_info.num_owned_faces;
    const uint16_t num_owned_edges    = patch_info.num_owned_edges;
    const uint16_t num_owned_vertices = patch_info.num_owned_vertices;
    const uint16_t num_vertices       = patch_info.num_vertices;
    const uint16_t num_edges          = patch_info.num_edges;

    // shared memory used to store EV and FE
    const uint16_t             mask_size_v = DIVIDE_UP(num_vertices, 32);
    const uint16_t             mask_size_e = DIVIDE_UP(num_edges, 32);
    extern __shared__ uint32_t shrd_mem32[];
    extern __shared__ uint16_t shrd_mem[];
    uint32_t*                  s_mask_v = shrd_mem32;
    uint16_t* s_ev = &shrd_mem[2 * std::max(mask_size_v, mask_size_e)];
    uint16_t* s_fe = s_ev;

    // load edges mask into shared memory
    load_async(patch_info.mask_v, mask_size_v, s_mask_v, false);
    __syncthreads();

    // we only need to load s_ev, operate on it, then load s_fe
    // we don't wait here
    load_mesh_async<Op::EV>(patch_info, s_ev, s_fe, false);

    // load over all vertices---one thread per vertex
    uint16_t local_id = threadIdx.x;

    // we need to make sure that the whole warp go into the loop
    uint16_t len = round_to_next_multiple_32(num_owned_vertices);

    while (local_id < len) {
        // check if vertex with local_id should be deleted and make sure we only
        // check owned edges
        bool to_delete = false;

        if (local_id < num_owned_vertices) {
            to_delete = predicate({patch_info.patch_id, local_id});
        }

        // update the vertex's bit mask. This function should be called by the
        // whole warp
        warp_update_mask(to_delete, local_id, s_mask_v);

        local_id += blockThreads;
    }

    // now we loop over EV and delete edges incident to deleted vertices 

    // store EV to global memory and vertices mask and load edges mask 

    // load FE and make sure we don't overwrite EV

    // now we loop over FE and deleted faces incident to deleted edges 

    // store FE to global memory 
}
}  // namespace detail
}  // namespace rxmesh