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
    const uint16_t num_owned_edges    = patch_info.num_owned_edges;
    const uint16_t num_owned_vertices = patch_info.num_owned_vertices;
    const uint16_t num_vertices       = patch_info.num_vertices;
    const uint16_t num_edges          = patch_info.num_edges;


    const uint16_t             mask_size_e = DIVIDE_UP(num_edges, 32);
    extern __shared__ uint32_t shrd_mem32[];
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_vertex_glue = shrd_mem;
    uint16_t*                  s_edge_glue   = s_vertex_glue;
    uint16_t* s_ev = &shrd_mem[std::max(num_vertices, num_edges)];
    uint16_t* s_fe = s_ev;

    // set the glued vertices to itself
    // no need to sync here. will rely on syn from the call to load_mesh_async
    for (uint16_t v = threadIdx.x; v < num_owned_vertices; v += blockThreads) {
        s_vertex_glue[v] = v;
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

        warp_update_mask(to_collapse, local_id, patch_info.mask_e);

        local_id += blockThreads;
    }

    __syncthreads();

    // we loop again over s_ev, and start glue edges i.e., replace an edge's
    // vertex with its entry i s_vertex_glue. Note that we initialized
    // s_vertex_glue sequentially so now change happens for vertices that should
    // not be glued
    local_id = threadIdx.x;
    while (local_id < num_owned_edges) {
        s_ev[2 * local_id]     = s_vertex_glue[s_ev[2 * local_id]];
        s_ev[2 * local_id + 1] = s_vertex_glue[s_ev[2 * local_id + 1]];

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

    

    
}
}  // namespace detail
}  // namespace rxmesh