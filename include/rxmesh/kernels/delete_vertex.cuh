#pragma once

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
    uint32_t*                  s_mask_e = &shrd_mem32[mask_size_v];
    uint16_t* s_ev = &shrd_mem[2 * (mask_size_v + mask_size_e)];
    uint16_t* s_fe = s_ev;

    // load edges mask into shared memory
    load_async(patch_info.mask_v, mask_size_v, s_mask_v, true);
    __syncthreads();

    // we only need to load s_ev, operate on it, then load s_fe
    // we start loading EV here without sync
    load_mesh_async<Op::EV>(patch_info, s_ev, s_fe, false);

    // start loading edges mask without sync
    load_async(patch_info.mask_e, mask_size_e, s_mask_e, false);

    // update the bitmask based on user-defined predicate
    update_bitmask<blockThreads>(
        num_owned_vertices, s_mask_v, [&](const uint16_t local_v) {
            return predicate({patch_info.patch_id, local_v});
        });
    __syncthreads();

    // store vertices mask
    store<blockThreads>(s_mask_v, mask_size_v, patch_info.mask_v);

    // now we loop over EV and delete edges incident to deleted vertices by
    // checking on vertices mask in shared memory and writing edges mask in
    // global memory
    update_bitmask<blockThreads>(
        num_owned_edges, s_mask_e, [&](const uint16_t local_e) {
            const uint16_t v0 = s_ev[2 * local_e + 0];
            const uint16_t v1 = s_ev[2 * local_e + 1];
            return is_deleted(v0, s_mask_v) || is_deleted(v1, s_mask_v);
        });
    __syncthreads();

    // load FE (and overwrite EV)
    load_mesh_async<Op::FE>(patch_info, s_ev, s_fe, true);
    __syncthreads();

    // store edge mask to global memory
    store<blockThreads>(s_mask_e, mask_size_e, patch_info.mask_e);

    // now we loop over FE and deleted faces incident to deleted edges
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