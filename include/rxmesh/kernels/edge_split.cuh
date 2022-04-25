#pragma once

#include "rxmesh/kernels/dynamic_util.cuh"

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

/**
 * @brief edge split
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void edge_split_2(PatchInfo&       patch_info,
                                        const predicateT predicate)
{
    using PredicateTTraits = detail::FunctionTraits<predicateT>;
    using HandleT          = typename PredicateTTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, EdgeHandle>,
        "First argument in predicate lambda function should be EdgeHandle");

    // patch basic info
    const uint32_t patch_id        = patch_info.patch_id;
    const uint16_t num_faces       = patch_info.num_faces;
    const uint16_t num_edges       = patch_info.num_edges;
    const uint16_t num_vertices    = patch_info.num_vertices;
    const uint16_t num_owned_edges = patch_info.num_owned_edges;


    // shared memory used to store EF, EV, FE, and info about the split edge
    __shared__ uint16_t        s_num_split_edges;
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_fe = shrd_mem;
    uint16_t* s_ef    = &shrd_mem[3 * num_faces + (3 * num_faces) % 2];
    uint16_t* s_ev    = s_ef;
    uint16_t* s_split = &s_ef[2 * num_edges];

    if (threadIdx.x == 0) {
        s_num_split_edges = 0;
    }

    // Initialize EF to invalid values
    // Cast EF into 32-but to fix the bank conflicts
    uint32_t* s_ef32 = reinterpret_cast<uint32_t*>(s_ef);
    for (uint16_t i = threadIdx.x; i < num_edges; i += blockThreads) {
        s_ef32[i] = INVALID32;
    }

    // load FE into shared memory
    load_mesh_async<Op::FE>(patch_info, s_ev, s_fe, true);
    __syncthreads();


    // Transpose FE into EF so we obtain the two incident faces to
    // to-be-flipped edges. We use the version that is optimized for
    // manifolds (we don't flip non-manifold edges)
    e_f_manifold<blockThreads>(
        num_edges, num_faces, s_fe, s_ef, patch_info.mask_e);
    __syncthreads();


    // loop over all edge
    block_loop<uint16_t, blockThreads, false>(
        num_owned_edges, [&](const uint16_t local_e) {
            // Do nothing if this edge is deleted
            if (!is_deleted(local_e, patch_info.mask_e)) {
                // check if we should split this edge based on the user-supplied
                // predicate
                if (predicate({patch_id, local_e})) {
                    const uint16_t split_id = atomicAdd(&s_num_split_edges, 1);
                    s_split[split_id]       = local_e;

                    // Added vertex id -> num_vertices + split_id
                    // The three added edges ids -> num_edges + 3*split_id + 0
                    //                             num_edges + 3*split_id + 1
                    //                             num_edges + 3*split_id + 2
                    // The two added faces ids -> num_faces + 2*split_id + 0
                    //                           num_faces + 2*split_id + 1
                }
            }
        });


    //TODO Set the active mask for added vertices, edges, and faces
    
    // Increment number of vertices, edges, and faces
    if (threadIdx.x == 0) {
        uint16_t num_split_edges = s_num_split_edges;
        patch_info.num_vertices += num_split_edges;
        patch_info.num_edges += 3 * num_split_edges;
        patch_info.num_faces += 2 * num_split_edges;
    }
}

}  // namespace detail
}  // namespace rxmesh