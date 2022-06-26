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
    //TODO
    /*using PredicateTTraits = detail::FunctionTraits<predicateT>;
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
    __shared__ uint32_t        s_num_split_edges;
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
        num_edges, num_faces, s_fe, s_ef, patch_info.active_mask_e);
    __syncthreads();


    // loop over all edge
    block_loop<uint16_t, blockThreads, false>(
        num_owned_edges, [&](const uint16_t local_e) {
            // Do nothing if this edge is deleted
            if (!is_deleted(local_e, patch_info.active_mask_e)) {
                // check if we should split this edge based on the user-supplied
                // predicate
                if (predicate({patch_id, local_e})) {
                    const uint16_t split_id = static_cast<uint16_t>(
                        ::atomicAdd(&s_num_split_edges, 3));
                    s_split[3 * split_id]    = local_e;
                    const uint16_t split_num = (split_id - num_edges) / 3;

                    // Added vertex id           -> num_vertices + split_num (V)
                    // The three added edges ids ->
                    //                            num_edges + split_id + 0 (Ea)
                    //                            num_edges + split_id + 1 (Eb)
                    //                            num_edges + split_id + 2 (Ec)
                    // The two added faces ids ->
                    //                         num_faces + 2*split_num + 0 (Fa)
                    //                         num_faces + 2*split_num + 1 (Fb)
                    //
                    // Fa will be adjacent to s_ef[2 * local_e + 0]
                    // Fb will be adjacent to s_ef[2 * local_e + 1]
                    //
                    // TODO in case of boundary split edge, we still add 3 edges
                    // and 2 faces but we mark the extra ones as inactive


                    const uint16_t Ea = num_edges + split_id + 0;

                    // Assign edges for Fa and Fb and update faces incident to
                    // local_e
                    for (uint16_t i = 0; i < 2; ++i) {
                        const uint16_t f = s_ef[2 * local_e + i];

                        if (f == INVALID16) {
                            // TODO make sure to mark the extra face and edge
                            // that is added for this boundary edge is marked as
                            // inactive
                            continue;
                        }

                        const uint16_t fx = num_faces + 2 * split_num + i;

                        const uint16_t ex = num_edges + split_id + i + 1;

                        // copy the connectivity of f into fx
                        // f is one of the faces incident to local_e
                        // fx is the face that will be adjacent to f along Ea

                        uint16_t fe[3];
                        fe[0] = s_fe[3 * f];
                        fe[1] = s_fe[3 * f + 1];
                        fe[2] = s_fe[3 * f + 2];

                        s_fe[3 * fx + 0] = fe[0];
                        s_fe[3 * fx + 1] = fe[1];
                        s_fe[3 * fx + 2] = fe[2];


                        // find local_e index within f's incident edges
                        const uint16_t l =
                            ((fe[0] >> 1) == local_e) ?
                                0 :
                                (((fe[1] >> 1) == local_e) ? 1 : 2);

                        // f's l-th edge stays the same
                        // but fx's l-th changes to Ea with same direction
                        s_fe[3 * fx + l] = (Ea << 1) | (fe[l] & 1);

                        // index within fe of next and previous edge to local_e
                        const uint16_t n = (l + 1) % 3;
                        const uint16_t p = (l + 2) % 3;

                        // f's next edge after local_e becomes Ex
                        // and preserve the direction of the original f's next
                        // edge
                        s_fe[3 * f + n] = (ex << 1) | (fe[n] & 1);

                        // fx's previous edge before local_e becomes Ex
                        // and its in the opposite direction as f's next edge
                        // after local_e
                        s_fe[3 * fx + p] = (ex << 1) | (fe[n] ^ 1);

                        // finally we store the edge next to local_e in the
                        // original f connectivity i.e., the edge we have
                        // moved/changed in f
                        s_split[3 * split_id + i + 1] = fe[n] >> 1;
                    }
                }
            }
        });

    __syncthreads();

    // load EV into shared memory and overwrite EF
    load_mesh_async<Op::EV>(patch_info, s_ev, s_fe, true);
    __syncthreads();

    // store FE
    store<blockThreads>(
        s_fe,
        3 * (num_faces + 2 * s_num_split_edges),  // new #faces after all split
        reinterpret_cast<uint16_t*>(patch_info.fe));

    block_loop<uint16_t, blockThreads, false>(
        s_num_split_edges, [&](const uint16_t split_id) {
            const uint16_t split_num = (split_id - num_edges) / 3;
            // Ea1 is local_e (the edge that was split)
            const uint16_t Ea1 = s_split[3 * split_id];

            // Ea is second half of the split edge Ea
            const uint16_t Ea = num_edges + split_id + 0;

            // Eb is the new edge created due to the split on the first incident
            // face to Ea1
            const uint16_t Eb = num_edges + split_id + 1;
            // Eb1, Eb, and Ea forms the new face due to split. This face is
            // adjacent to the first face incident to Ea1.
            const uint16_t Eb1 = s_split[3 * split_id + 1];

            // Ec is the new edge created due to the split on the second
            // incident face to Ea1
            const uint16_t Ec = num_edges + split_id + 2;
            // Ec1, Ec, and Ea1 forms the new face due to split. This face is
            // adjacent to the first second incident to Ea1.
            const uint16_t Ec1 = s_split[3 * split_id + 2];

            const uint16_t v = num_vertices + split_num;


            s_ev[2 * Ea + 0] = s_ev[2 * Ea1 + 0];
            s_ev[2 * Ea + 1] = s_ev[2 * Ea1 + 1];

            s_ev[2 * Eb + 0] = s_ev[2 * Eb1 + 0];
            s_ev[2 * Eb + 1] = s_ev[2 * Eb1 + 1];

            s_ev[2 * Ec + 0] = s_ev[2 * Ec1 + 0];
            s_ev[2 * Ec + 1] = s_ev[2 * Ec1 + 1];
        });

    // TODO Set the active mask for added vertices, edges, and faces


    // Increment number of vertices, edges, and faces
    if (threadIdx.x == 0) {
        uint16_t num_split_edges = s_num_split_edges;
        patch_info.num_vertices += num_split_edges;
        patch_info.num_edges += 3 * num_split_edges;
        patch_info.num_faces += 2 * num_split_edges;
    }*/
}

}  // namespace detail
}  // namespace rxmesh