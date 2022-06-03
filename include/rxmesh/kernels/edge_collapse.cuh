#pragma once

#include "rxmesh/kernels/dynamic_util.cuh"
#include "rxmesh/kernels/loader.cuh"

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
    const uint32_t patch_id           = patch_info.patch_id;
    const uint16_t num_owned_faces    = patch_info.num_owned_faces;
    const uint16_t num_owned_edges    = patch_info.num_owned_edges;
    const uint16_t num_owned_vertices = patch_info.num_owned_vertices;
    const uint16_t num_vertices       = patch_info.num_vertices;
    const uint16_t num_edges          = patch_info.num_edges;

    const uint16_t             active_mask_e_size = DIVIDE_UP(num_edges, 32);
    extern __shared__ uint32_t shrd_mem32[];
    extern __shared__ uint16_t shrd_mem[];
    uint32_t*                  s_active_mask_e = shrd_mem32;
    uint16_t* s_edge_source_vertex = &shrd_mem[2 * active_mask_e_size];
    uint16_t* s_vertex_glue =
        &s_edge_source_vertex[num_edges + (num_edges % 2)];
    uint16_t* s_edge_glue = s_vertex_glue;
    uint16_t  m           = std::max(num_vertices, num_edges);
    uint16_t* s_ev        = &s_edge_glue[m + (m % 2)];
    uint16_t* s_fe        = s_ev;


    // set the glued vertices to itself
    // no need to sync here. will rely on sync from the call to load_mesh_async
    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockThreads) {
        s_vertex_glue[v] = v;
    }

    // set the mask bit for all edges to indicate that all edges are not
    // collapsed
    for (uint16_t e = threadIdx.x; e < active_mask_e_size; e += blockThreads) {
        s_active_mask_e[e] = INVALID32;
    }


    // we only need to load s_ev, operate on it, then load s_fe
    load_mesh_async<Op::EV>(patch_info, s_ev, s_fe, true);


    // loop over all owned edges. if an edge should be collapsed do:
    // 1. mark the edge as collapsed in s_active_mask_e (set its bit to zero)
    // 2. for the two end vertices of an edge (v0,v1), add v1 as the vertex to
    // glue to in s_vertex_glue[v0]
    // We only do this for existing edges i.e., we need to check if the edge has
    // be previously deleted. This info is not what is stored in s_active_mask_e
    // but instead in the bitmask in global memory
    update_bitmask<blockThreads>(
        num_owned_edges,
        s_active_mask_e,
        [&](const uint16_t local_e) {
            if (is_deleted(local_e, patch_info.active_mask_e)) {
                return false;
            } else {
                bool to_collapse = predicate({patch_id, local_e});
                if (to_collapse) {
                    uint16_t v0 = s_ev[2 * local_e];
                    uint16_t v1 = s_ev[2 * local_e + 1];

                    // we will mark v0 as deleted later
                    s_vertex_glue[v0] = v1;
                }
                return to_collapse;
            }
        },
        false);

    __syncthreads();


    // we loop over s_ev, and start glue vertices i.e., replace an edge's
    // vertex with its entry in s_vertex_glue. Note that we initialized
    // s_vertex_glue sequentially so no change happens for vertices that should
    // not be glued
    block_loop<uint16_t, blockThreads, false>(
        num_edges, [&](const uint16_t local_e) {
            uint16_t src_v    = s_vertex_glue[s_ev[2 * local_e]];
            s_ev[2 * local_e] = src_v;

            s_ev[2 * local_e + 1] = s_vertex_glue[s_ev[2 * local_e + 1]];
            s_edge_source_vertex[local_e] = src_v;
        });
    __syncthreads();


    // we are done with s_ev so we store it in global memory
    store<blockThreads>(s_ev,
                        patch_info.num_edges * 2,
                        reinterpret_cast<uint16_t*>(patch_info.ev));


    // We mark glued vertices as deleted in their bitmask in global memory
    // by checking on which vertex it has been glued to. If it was glued to
    // itself, then it has not be deleted
    update_bitmask<blockThreads>(num_owned_vertices,
                                 patch_info.active_mask_v,
                                 [&](const uint16_t local_v) {
                                     return s_vertex_glue[local_v] != local_v;
                                 });
    // sync so s_ev (or s_vertex_glue) is not overwritten
    __syncthreads();

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        s_edge_glue[e] = e;
    }
    // load s_fe where s_ev was stored
    load_mesh_async<Op::FE>(patch_info, s_ev, s_fe, true);

    // loop over FE and find glue-able edges
    // 1. check if any edge is collapsed
    // 2. if an edge is collapsed, then the next edge in the face should be
    // glued to the previous one
    // 3. if an edge is collapsed, then the face should be deleted
    update_bitmask<blockThreads>(
        num_owned_faces, patch_info.active_mask_f, [&](const uint16_t local_f) {
            uint16_t e[3];

            e[0] = s_fe[3 * local_f + 0] >> 1;
            e[1] = s_fe[3 * local_f + 1] >> 1;
            e[2] = s_fe[3 * local_f + 2] >> 1;

            bool is_collapsed[3];
            is_collapsed[0] = is_deleted(e[0], s_active_mask_e);
            is_collapsed[1] = is_deleted(e[1], s_active_mask_e);
            is_collapsed[2] = is_deleted(e[2], s_active_mask_e);

            if (is_collapsed[0]) {
                s_edge_glue[e[1]] = e[2];
            }

            if (is_collapsed[1]) {
                s_edge_glue[e[2]] = e[0];
            }

            if (is_collapsed[2]) {
                s_edge_glue[e[0]] = e[1];
            }

            return (is_collapsed[0] || is_collapsed[1] || is_collapsed[2]);
        });

    __syncthreads();

    // glued edges should update it edge mask to indicate that they are deleted
    update_bitmask<blockThreads>(
        num_edges, s_active_mask_e, [&](const uint16_t local_e) {
            return (s_edge_glue[local_e] != local_e);
        });


    // we loop over s_fe again, and start glue edges i.e., replace a face's
    // edge with its entry in s_edge_glue. Note that we initialized s_edge_glue
    // sequentially so no change happens for edges that should not be glued
    block_loop<uint16_t, blockThreads, false>(
        num_owned_faces, [&](const uint16_t local_f) {
            for (int i = 0; i < 3; ++i) {
                uint16_t e;
                uint8_t  d;
                Context::unpack_edge_dir(s_fe[3 * local_f + i], e, d);
                uint16_t e_glue = s_edge_glue[e];
                if (e != e_glue) {
                    // toggle the edge direction if the source vertices don't
                    // match
                    if (s_edge_source_vertex[e] !=
                        s_edge_source_vertex[e_glue]) {
                        d ^= 1;
                    }
                    e = e_glue << 1;
                    e |= d;
                    s_fe[3 * local_f + i] = e;
                }
            }
        });
    __syncthreads();
    // store FE to global memory
    store<blockThreads>(s_fe,
                        patch_info.num_faces * 3,
                        reinterpret_cast<uint16_t*>(patch_info.fe));

    // Store edge mask by AND'ing the mask in shared memory with that in
    // global memory
    for (uint16_t e = threadIdx.x; e < active_mask_e_size; e += blockThreads) {
        patch_info.active_mask_e[e] &= s_active_mask_e[e];
    }

    __syncthreads();
}
}  // namespace detail
}  // namespace rxmesh