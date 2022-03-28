#pragma once

#include "rxmesh/kernels/is_deleted.cuh"
#include "rxmesh/kernels/warp_update_mask.cuh"

namespace rxmesh {
namespace detail {

/**
 * @brief edge split
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void edge_split(PatchInfo&       patch_info,
                                      const predicateT predicate)
{
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
    const uint16_t num_faces          = patch_info.num_faces;

    __shared__ uint32_t        s_num_vertices;
    __shared__ uint32_t        s_num_edges;
    __shared__ uint32_t        s_num_faces;
    const uint16_t             mask_size_e = DIVIDE_UP(num_edges, 32);
    extern __shared__ uint32_t shrd_mem32[];
    extern __shared__ uint16_t shrd_mem[];
    uint32_t*                  s_mask_e = shrd_mem32;
    uint16_t* s_add_edges_and_vertices  = &shrd_mem[2 * mask_size_e];
    uint16_t* s_ev = &s_add_edges_and_vertices[patch_info.edges_capacity +
                                               (patch_info.edges_capacity % 2)];
    uint16_t* s_fe = s_ev;

    if (threadIdx.x == 0) {
        s_num_vertices = num_vertices;
        s_num_edges    = num_edges;
        s_num_faces    = num_faces;
    }
    // set the mask bit for all edges to indicate that all edges are not
    // split
    for (uint16_t e = threadIdx.x; e < mask_size_e; e += blockThreads) {
        s_mask_e[e] = INVALID32;
    }

    // we only need to load s_ev, operate on it, then load s_fe
    load_mesh_async<Op::FE>(patch_info, s_ev, s_fe, true);

    // we need to make sure that the whole warp go into the loop
    uint16_t len = round_to_next_multiple_32(num_owned_edges);

    // loop over all edges, and mark the split edges in s_mask_e
    uint16_t local_id = threadIdx.x;
    while (local_id < len) {
        bool to_split = false;
        if (local_id < num_owned_edges) {
            to_split = predicate({patch_info.patch_id, local_id});

            uint32_t new_edge_id =
                static_cast<uint16_t>(::atomicAdd(&s_num_edges, 1));

            s_add_edges_and_vertices[local_id] = new_edge_id;

            uint32_t new_vertex_id =
                static_cast<uint16_t>(::atomicAdd(&s_num_vertices, 1));

            s_add_edges_and_vertices[new_edge_id] = new_vertex_id;
        }
        warp_update_mask(to_split, local_id, s_mask_e);

        local_id += blockThreads;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        patch_info.num_vertices = s_num_vertices;
    }


    // loop FE where a face check if any of its edges are split,
    // in which case, the face will create a new edge, change one of the face's
    // edge to be this new edge, and create a new face
    local_id = threadIdx.x;
    while (local_id < num_owned_faces) {
        uint16_t e[3];
        uint8_t  d[3];
        bool     has_split_edge[3];

        Context::unpack_edge_dir(s_fe[3 * local_id + 0], e[0], d[0]);
        Context::unpack_edge_dir(s_fe[3 * local_id + 1], e[1], d[1]);
        Context::unpack_edge_dir(s_fe[3 * local_id + 2], e[2], d[2]);
        has_split_edge[0] = is_deleted(e[0], s_mask_e);
        has_split_edge[1] = is_deleted(e[1], s_mask_e);
        has_split_edge[2] = is_deleted(e[2], s_mask_e);

        if (has_split_edge[0] || has_split_edge[1] || has_split_edge[2]) {
            // this face will create new face (f_new) and a new edge (e_new). it
            // will also change one of it edges to be the e_new. f_new will
            // connect
            // e_new, the edge that changed in this face,

            // update the connectivity of the local_id face—only one edge needs
            // to be updates

            // TODO need to make sure that the face's edge orientation are
            // set correctly. same goes to the new face
            const uint16_t split_edge_id =
                (has_split_edge[0]) ? 0 : (has_split_edge[1] ? 1 : 2);

            const uint32_t update_edge_id = (split_edge_id + 1) % 3;

            const uint32_t update_edge = s_fe[3 * local_id + update_edge_id];

            const uint32_t new_edge =
                static_cast<uint16_t>(::atomicAdd(&s_num_edges, 1));

            s_fe[3 * local_id + update_edge_id] = (new_edge << 1);

            // create new face
            const uint32_t new_face =
                static_cast<uint16_t>(::atomicAdd(&s_num_faces, 1));

            const uint16_t added_edge = s_add_edges_and_vertices[split_edge_id];

            s_fe[3 * new_face + 0] = update_edge << 1;
            s_fe[3 * new_face + 1] = added_edge << 1;
            s_fe[3 * new_face + 2] = new_edge << 1;
        }

        local_id += blockThreads;
    }


    __syncthreads();

    if (threadIdx.x == 0) {
        patch_info.num_edges = s_num_edges;
        patch_info.num_faces = s_num_faces;
    }

    // store FE in global memory
    store<blockThreads>(
        s_fe, s_num_faces * 3, reinterpret_cast<uint16_t*>(patch_info.fe));

    __syncthreads();

    // load EV (overwrites FE)
    load_mesh_async<Op::EV>(patch_info, s_ev, s_fe, true);
}

}  // namespace detail
}  // namespace rxmesh