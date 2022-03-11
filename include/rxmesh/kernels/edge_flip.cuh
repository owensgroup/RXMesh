namespace rxmesh {

namespace detail {
/**
 * @brief edge flip
 * TODO check first if the edge is deleted
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void edge_flip(PatchInfo&       patch_info,
                                     const predicateT predicate)
{
    // Extract the argument in the predicate lambda function
    using PredicateTTraits = FunctionTraits<predicateT>;
    using HandleT          = typename PredicateTTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, EdgeHandle>,
        "First argument in predicate lambda function should be EdgeHandle");

    // patch basic info
    const uint16_t num_faces       = patch_info.num_faces;
    const uint16_t num_edges       = patch_info.num_edges;
    const uint16_t num_owned_edges = patch_info.num_owned_edges;

    // shared memory used to store EF, EV, FE, and info about the flipped edge
    __shared__ uint16_t        s_num_flipped_edges;
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_fe = shrd_mem;
    uint16_t* s_ef      = &shrd_mem[3 * num_faces + (3 * num_faces) % 2];
    uint16_t* s_ev      = s_ef;
    uint16_t* s_flipped = &s_ef[2 * num_edges];

    if (threadIdx.x == 0) {
        s_num_flipped_edges = 0;
    }
    // Initialize EF to invalid values
    // Cast EF into 32-but to fix the bank conflicts
    uint32_t* s_ef32 = reinterpret_cast<uint32_t*>(s_ef);
    for (uint16_t i = threadIdx.x; i < num_edges; i += blockThreads) {
        s_ef32[i] = INVALID32;
    }

    // load FE into shared memory
    load_async(reinterpret_cast<const uint16_t*>(patch_info.fe),
               num_faces * 3,
               reinterpret_cast<uint16_t*>(s_fe),
               true);
    __syncthreads();

    // Transpose FE into EF so we obtain the two incident triangles to
    // to-be-flipped edges. We use the version that is optimized for
    // manifolds (we don't flip non-manifold edges)
    e_f_manifold<blockThreads>(num_edges, num_faces, s_fe, s_ef);
    __syncthreads();

    // load over all edges---one thread per edge
    uint16_t local_id = threadIdx.x;
    while (local_id < num_owned_edges) {

        // check if we should flip this edge based on the user-supplied
        // predicate
        if (predicate({patch_info.patch_id, local_id})) {

            // read the two faces incident to this edge
            const uint16_t f0 = s_ef[2 * local_id];
            const uint16_t f1 = s_ef[2 * local_id + 1];

            // if the edge is boundary (i.e., only incident to one face), then
            // we don't flip
            if (f0 != INVALID16 && f1 != INVALID16) {
                const uint16_t flipped_id = atomicAdd(&s_num_flipped_edges, 1);

                // for each flipped edge, we add it to shared memory buffer
                // along with one of its incident faces
                s_flipped[2 * flipped_id]     = local_id;
                s_flipped[2 * flipped_id + 1] = f0;

                // the three edges incident to the first face
                uint16_t f0_e[3];
                f0_e[0] = s_fe[3 * f0];
                f0_e[1] = s_fe[3 * f0 + 1];
                f0_e[2] = s_fe[3 * f0 + 2];

                // the flipped edge position in first face
                const uint16_t l0 = ((f0_e[0] >> 1) == local_id) ?
                                        0 :
                                        (((f0_e[1] >> 1) == local_id) ? 1 : 2);

                // the three edges incident to the second face
                uint16_t f1_e[3];
                f1_e[0] = s_fe[3 * f1];
                f1_e[1] = s_fe[3 * f1 + 1];
                f1_e[2] = s_fe[3 * f1 + 2];

                // the flipped edge position in second face
                const uint16_t l1 = ((f1_e[0] >> 1) == local_id) ?
                                        0 :
                                        (((f1_e[1] >> 1) == local_id) ? 1 : 2);

                const uint16_t f0_shift = 3 * f0;
                const uint16_t f1_shift = 3 * f1;

                s_fe[f0_shift + ((l0 + 1) % 3)] = f0_e[l0];
                s_fe[f1_shift + ((l1 + 1) % 3)] = f1_e[l1];

                s_fe[f1_shift + l1] = f0_e[(l0 + 1) % 3];
                s_fe[f0_shift + l0] = f1_e[(l1 + 1) % 3];
            }
        }
        local_id += blockThreads;
    }
    __syncthreads();

    // If flipped at least one edge
    if (s_num_flipped_edges > 0) {
        // we store the changes we made to FE in global memory
        store<blockThreads>(
            s_fe, 3 * num_faces, reinterpret_cast<uint16_t*>(patch_info.fe));

        // load EV in the same place that was used to store EF
        load_async(reinterpret_cast<const uint16_t*>(patch_info.ev),
                   num_edges * 2,
                   reinterpret_cast<uint16_t*>(s_ev),
                   true);
        __syncthreads();

        // Now, we go over all edge that has been flipped
        for (uint32_t e = threadIdx.x; e < s_num_flipped_edges;
             e += blockThreads) {

            // grab the edge that was flipped and one of its incident faces
            const uint16_t edge = s_flipped[2 * e];
            const uint16_t face = s_flipped[2 * e + 1];

            // grab the three edges incident to this face
            uint16_t fe[3];
            fe[0] = s_fe[3 * face];
            fe[1] = s_fe[3 * face + 1];
            fe[2] = s_fe[3 * face + 2];

            // find the flipped edge's position in this face incident edges
            const uint16_t l =
                ((fe[0] >> 1) == edge) ? 0 : (((fe[1] >> 1) == edge) ? 1 : 2);

            // find which edge is next and which is before the flipped edge
            // in this face incident edge
            const uint16_t next_edge = fe[(l + 1) % 3];
            const uint16_t prev_edge = fe[(l + 2) % 3];

            // grab the two end vertices of the next edge
            uint16_t next_edge_v[2];
            next_edge_v[0] = s_ev[2 * (next_edge >> 1)];
            next_edge_v[1] = s_ev[2 * (next_edge >> 1) + 1];

            // grab the two end vertices of the previous edge
            uint16_t prev_edge_v[2];
            prev_edge_v[0] = s_ev[2 * (prev_edge >> 1)];
            prev_edge_v[1] = s_ev[2 * (prev_edge >> 1) + 1];

            // Find which vertex from the next edge that is not common between
            // the two incident vertices in the next and previous edges
            const uint16_t n = (next_edge_v[0] == prev_edge_v[0] ||
                                next_edge_v[0] == prev_edge_v[1]) ?
                                   next_edge_v[1] :
                                   next_edge_v[0];

            // Find which vertex from the previous edge that is not common
            // between the two incident vertices in the next and previous edges
            const uint16_t p = (prev_edge_v[0] == next_edge_v[0] ||
                                prev_edge_v[0] == next_edge_v[1]) ?
                                   prev_edge_v[1] :
                                   prev_edge_v[0];

            // The flipped edge connects p and n vertices. The order depends on
            // the edge direction
            if ((fe[l] & 1)) {
                s_ev[2 * edge]     = n;
                s_ev[2 * edge + 1] = p;
            } else {
                s_ev[2 * edge]     = p;
                s_ev[2 * edge + 1] = n;
            }
        }

        // We store the changes in EV to global memory
        __syncthreads();
        store<blockThreads>(
            s_ev, num_edges * 2, reinterpret_cast<uint16_t*>(patch_info.ev));
    }
}
}  // namespace detail
}  // namespace rxmesh