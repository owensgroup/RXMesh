#pragma once

#include "rxmesh/kernels/is_deleted.cuh"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/kernels/warp_update_mask.cuh"

namespace rxmesh {
namespace detail {

/**
 * @brief edge collapse
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void edge_collapse(PatchInfo&       patch_info,
                                         const predicateT predicate)
{
    // Extract the argument in the predicate lambda function
    using PredicateTTraits = detail::FunctionTraits<predicateT>;
    using HandleT          = typename PredicateTTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, EdgeHandle>,
        "First argument in predicate lambda function should be EdgeHandle");

    // patch basic info
    const uint16_t num_owned_faces    = patch_info.num_owned_faces;
    const uint16_t num_owned_edges    = patch_info.num_owned_edges;
    const uint16_t num_owned_vertices = patch_info.num_owned_vertices;
    const uint16_t num_vertices       = patch_info.num_vertices;
    const uint16_t num_edges          = patch_info.num_edges;

    const uint16_t             mask_size_e = DIVIDE_UP(num_edges, 32);
    extern __shared__ uint32_t shrd_mem32[];
    extern __shared__ uint16_t shrd_mem[];
    uint32_t*                  s_mask_e = shrd_mem32;
    uint16_t* s_edge_source_vertex      = &shrd_mem[2 * mask_size_e];
    uint16_t* s_vertex_glue = &s_edge_source_vertex[num_owned_edges];
    uint16_t* s_edge_glue   = s_vertex_glue;
    uint16_t* s_ev          = &shrd_mem[std::max(num_vertices, num_edges)];
    uint16_t* s_fe          = s_ev;

    // set the glued vertices to itself
    // no need to sync here. will rely on syn from the call to load_mesh_async
    for (uint16_t v = threadIdx.x; v < num_owned_vertices; v += blockThreads) {
        s_vertex_glue[v] = v;
    }

    for (uint16_t e = threadIdx.x; e < num_owned_edges; e += blockThreads) {
        s_mask_e[e] = INVALID32;
    }


    // we only need to load s_ev, operate on it, then load s_fe
    load_mesh_async<Op::EV>(patch_info, s_ev, s_fe, true);


    // we need to make sure that the whole warp go into the loop
    uint16_t len = round_to_next_multiple_32(num_owned_edges);

    // loop over all owned edges. if an edge should be collapsed do:
    // 1. mark the edge as collapsed in s_mask_e (set its bit to zero)
    // 2. for the two end vertices of an edge (v0,v1), add v1 as the vertex to
    // glue to in s_vertex_glue[v0]
    // 3. Mark the collapsed edge vertices as INVALID16 in s_ev
    uint16_t local_id = threadIdx.x;
    while (local_id < len) {

        bool to_collapse = false;

        if (local_id < num_owned_edges) {
            to_collapse = predicate({patch_info.patch_id, local_id});
        }

        if (to_collapse) {
            uint16_t v0 = s_ev[2 * local_id];
            uint16_t v1 = s_ev[2 * local_id + 1];

            // we will mark v0 as deleted later
            s_vertex_glue[v0] = v1;

            s_ev[2 * local_id]     = INVALID16;
            s_ev[2 * local_id + 1] = INVALID16;
        }

        warp_update_mask(to_collapse, local_id, s_mask_e);

        local_id += blockThreads;
    }

    __syncthreads();

    // we loop again over s_ev, and start glue vertices i.e., replace an edge's
    // vertex with its entry in s_vertex_glue. Note that we initialized
    // s_vertex_glue sequentially so no change happens for vertices that should
    // not be glued
    local_id = threadIdx.x;
    while (local_id < num_owned_edges) {
        s_ev[2 * local_id]     = s_vertex_glue[s_ev[2 * local_id]];
        s_ev[2 * local_id + 1] = s_vertex_glue[s_ev[2 * local_id + 1]];

        s_edge_source_vertex[local_id] = s_ev[2 * local_id];

        local_id += blockThreads;
    }

    __syncthreads();

    // we are done with s_ev so we store it in global memory
    store<blockThreads>(s_ev,
                        patch_info.num_edges * 2,
                        reinterpret_cast<uint16_t*>(patch_info.ev));

    // We mark glued vertices as deleted in their bitmask in global memory
    // by checking on which vertex it has been glued to. If it glued to itself,
    // then it has not be deleted
    local_id = threadIdx.x;
    while (local_id < num_owned_vertices) {
        bool is_deleted_v = (s_vertex_glue[local_id] != local_id);

        warp_update_mask(is_deleted_v, local_id, patch_info.mask_v);
        local_id += blockThreads;
    }

    // sync so s_ev is not overwritten
    __syncthreads();

    for (uint16_t e = threadIdx.x; e < num_owned_edges; e += blockThreads) {
        s_edge_glue[e] = e;
    }
    // load s_fe where s_ev was stored
    load_mesh_async<Op::FE>(patch_info, s_ev, s_fe, true);

    // loop over FE and find glue-able edges
    // 1. check if any edge is collapsed
    // 2. if an edge is collapsed, then the next edge in the face should be
    // glued to the previous one
    // 3. if an edge is collapsed, then the face should be deleted
    local_id = threadIdx.x;
    while (local_id < num_owned_faces) {
        uint16_t e[3];
        uint8_t  d[3];
        Context::unpack_edge_dir(s_fe[3 * local_id + 0], e[0], d[0]);
        Context::unpack_edge_dir(s_fe[3 * local_id + 1], e[1], d[1]);
        Context::unpack_edge_dir(s_fe[3 * local_id + 2], e[2], d[2]);

        bool is_collapsed[3];
        is_collapsed[0] = is_deleted(e[0], s_mask_e);
        is_collapsed[1] = is_deleted(e[1], s_mask_e);
        is_collapsed[2] = is_deleted(e[2], s_mask_e);

        if (is_collapsed[0]) {
            s_edge_glue[e[1]] = e[2];
        }

        if (is_collapsed[1]) {
            s_edge_glue[e[2]] = e[0];
        }

        if (is_collapsed[2]) {
            s_edge_glue[e[0]] = e[1];
        }

        bool delete_face =
            (is_collapsed[0] || is_collapsed[1] || is_collapsed[2]);

        if (delete_face) {
            s_fe[3 * local_id + 0] = INVALID16;
            s_fe[3 * local_id + 1] = INVALID16;
            s_fe[3 * local_id + 2] = INVALID16;
        }

        warp_update_mask(delete_face, local_id, patch_info.mask_f);

        local_id += blockThreads;
    }

    __syncthreads();

    // we loop over s_fe again, and start glue edges i.e., replace a face's
    // edge with its entry in s_edge_glue. Note that we initialized s_edge_glue
    // sequentially so no change happens for edges that should not be glued
    local_id = threadIdx.x;
    while (local_id < num_owned_faces) {
        for (int i = 0; i < 3; ++i) {
            uint16_t e;
            uint8_t  d;
            Context::unpack_edge_dir(s_fe[3 * local_id + i], e, d);
            uint16_t e_glue = s_edge_glue[e];
            if (e != e_glue) {
                // toggle the edge direction if the source vertices don't match
                if (s_edge_source_vertex[e] != s_edge_source_vertex[e_glue]) {
                    d ^= 1;
                }
                e = e << 1;
                e |= d;
                s_fe[3 * local_id + i] = e;
            }
        }

        local_id += blockThreads;
    }

    __syncthreads();
    // store FE to global memory
    store<blockThreads>(s_fe,
                        patch_info.num_faces * 3,
                        reinterpret_cast<uint16_t*>(patch_info.fe));

    // Store edge mask
    store<blockThreads>(s_mask_e, mask_size_e, patch_info.mask_e);

    __syncthreads();
}
}  // namespace detail
}  // namespace rxmesh