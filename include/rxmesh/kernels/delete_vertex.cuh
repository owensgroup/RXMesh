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
    uint32_t*                  s_mask_e = shrd_mem32;
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

    // wait till all s_mask_v has been written by all threads and also EV
    // has been loaded into shared memory
    __syncthreads();

    // store vertices mask but don't wait
    store<blockThreads>(s_mask_v, mask_size_v, patch_info.mask_v);

    // now we loop over EV and delete edges incident to deleted vertices by
    // checking on vertices mask in shared memory and writing edges mask in
    // global memory
    len      = round_to_next_multiple_32(num_owned_edges);
    local_id = threadIdx.x;
    while (local_id < len) {
        bool to_delete = false;

        if (local_id < num_owned_edges) {
            const uint16_t v0 = s_ev[2 * local_id + 0];
            const uint16_t v1 = s_ev[2 * local_id + 1];
            to_delete = is_deleted(v0, s_mask_v) || is_deleted(v1, s_mask_v);
        }

        if (to_delete) {            
            s_ev[2 * local_id + 0] = INVALID16;
            s_ev[2 * local_id + 1] = INVALID16;
        }

        // update the face's bit mask. This function should be called by the
        // whole warp
        warp_update_mask(to_delete, local_id, patch_info.mask_e);

        local_id += blockThreads;
    }

    // wait so all threads has written edges mask
    __syncthreads();

    // load edges mask into shared memory
    load_async(patch_info.mask_e, mask_size_e, s_mask_e, false);

    // store EV to global memory and vertices mask and load edges mask
    store<blockThreads>(s_ev,
                        patch_info.num_edges * 2,
                        reinterpret_cast<uint16_t*>(patch_info.ev));

    // wait so we don't overwrite s_ev
    __syncthreads();

    // load FE (and overwrite EV)
    load_mesh_async<Op::FE>(patch_info, s_ev, s_fe, false);
    __syncthreads();

    // now we loop over FE and deleted faces incident to deleted edges
    len      = round_to_next_multiple_32(num_owned_faces);
    local_id = threadIdx.x;
    while (local_id < len) {
        bool to_delete = false;

        if (local_id < num_owned_faces) {
            const uint16_t e0 = s_fe[3 * local_id + 0] >> 1;
            const uint16_t e1 = s_fe[3 * local_id + 1] >> 1;
            const uint16_t e2 = s_fe[3 * local_id + 2] >> 1;

            to_delete = is_deleted(e0, s_mask_e) || is_deleted(e1, s_mask_e) ||
                        is_deleted(e2, s_mask_e);
        }

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

    // store FE to global memory
    store<blockThreads>(s_fe,
                        patch_info.num_faces * 3,
                        reinterpret_cast<uint16_t*>(patch_info.fe));
    __syncthreads();
}
}  // namespace detail
}  // namespace rxmesh