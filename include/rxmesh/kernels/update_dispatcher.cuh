#pragma once

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
#include "rxmesh/kernels/loader.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/util/meta.h"

namespace rxmesh {

namespace detail {
/**
 * @brief
 */
template <uint32_t blockThreads, typename flipT>
__device__ __inline__ void edge_flip_block_dispatcher(PatchInfo&  patch_info,
                                                      const flipT flip)
{
    __shared__ uint16_t s_num_flipped_edges;
    const uint16_t      num_faces       = patch_info.num_faces;
    const uint16_t      num_edges       = patch_info.num_edges;
    const uint16_t      num_owned_edges = patch_info.num_owned_edges;

    // load the patch
    extern __shared__ uint16_t shrd_mem[];
    uint16_t*                  s_fe = shrd_mem;
    uint16_t* s_ef      = &shrd_mem[3 * num_faces + (3 * num_faces) % 2];
    uint16_t* s_ev      = s_ef;
    uint16_t* s_flipped = &s_ef[2 * num_edges];

    if (threadIdx.x == 0) {
        s_num_flipped_edges = 0;
    }
    // to fix the bank conflicts
    uint32_t* s_ef32 = reinterpret_cast<uint32_t*>(s_ef);
    for (uint16_t i = threadIdx.x; i < num_edges; i += blockThreads) {
        s_ef32[i] = INVALID32;
    }
    load_patch_FE<blockThreads>(patch_info,
                                reinterpret_cast<LocalEdgeT*>(s_fe));
    __syncthreads();

    // Transpose FE into EF so we obtain the two incident triangles to
    // the flipped edges. We use the version that is optimized for
    // manifolds
    e_f_manifold<blockThreads>(num_edges, num_faces, s_fe, s_ef);
    __syncthreads();


    uint16_t local_id = threadIdx.x;
    while (local_id < num_owned_edges) {

        if (flip({patch_info.patch_id, local_id})) {
            const uint16_t f0 = s_ef[2 * local_id];
            const uint16_t f1 = s_ef[2 * local_id + 1];
            if (f0 != INVALID16 && f1 != INVALID16) {

                const uint16_t flipped_id = atomicAdd(&s_num_flipped_edges, 1);

                s_flipped[2 * flipped_id]     = local_id;
                s_flipped[2 * flipped_id + 1] = f0;

                uint16_t f0_e[3];
                f0_e[0] = s_fe[3 * f0];
                f0_e[1] = s_fe[3 * f0 + 1];
                f0_e[2] = s_fe[3 * f0 + 2];

                const uint16_t l0 = ((f0_e[0] >> 1) == local_id) ?
                                        0 :
                                        (((f0_e[1] >> 1) == local_id) ? 1 : 2);

                uint16_t f1_e[3];
                f1_e[0] = s_fe[3 * f1];
                f1_e[1] = s_fe[3 * f1 + 1];
                f1_e[2] = s_fe[3 * f1 + 2];

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
    load_uint16<blockThreads>(
        s_fe, 3 * num_faces, reinterpret_cast<uint16_t*>(patch_info.fe));

    if (s_num_flipped_edges > 0) {
        load_patch_EV<blockThreads>(patch_info,
                                    reinterpret_cast<LocalVertexT*>(s_ev));
        __syncthreads();
    }
    for (uint32_t e = threadIdx.x; e < s_num_flipped_edges; e += blockThreads) {
        const uint16_t edge = s_flipped[2 * e];
        const uint16_t face = s_flipped[2 * e + 1];

        uint16_t fe[3];
        fe[0] = s_fe[3 * face];
        fe[1] = s_fe[3 * face + 1];
        fe[2] = s_fe[3 * face + 2];

        const uint16_t l =
            ((fe[0] >> 1) == edge) ? 0 : (((fe[1] >> 1) == edge) ? 1 : 2);

        const uint16_t next_edge = fe[(l + 1) % 3];
        const uint16_t prev_edge = fe[(l + 2) % 3];

        uint16_t next_edge_v[2];
        next_edge_v[0] = s_ev[2 * (next_edge >> 1)];
        next_edge_v[1] = s_ev[2 * (next_edge >> 1) + 1];

        uint16_t prev_edge_v[2];
        prev_edge_v[0] = s_ev[2 * (prev_edge >> 1)];
        prev_edge_v[1] = s_ev[2 * (prev_edge >> 1) + 1];

        const uint16_t n = (next_edge_v[0] == prev_edge_v[0] ||
                            next_edge_v[0] == prev_edge_v[1]) ?
                               next_edge_v[1] :
                               next_edge_v[0];

        const uint16_t p = (prev_edge_v[0] == next_edge_v[0] ||
                            prev_edge_v[0] == next_edge_v[1]) ?
                               prev_edge_v[1] :
                               prev_edge_v[0];

        s_ev[2 * e]     = n;
        s_ev[2 * e + 1] = p;
    }

    __syncthreads();

    load_uint16<blockThreads>(
        s_ev, num_edges * 2, reinterpret_cast<uint16_t*>(patch_info.ev));
}
}  // namespace detail

/**
 * @brief
 */
template <uint32_t blockThreads, typename flipT>
__device__ __inline__ void edge_flip_block_dispatcher(const Context& context,
                                                      const flipT    flip)
{
    // Extract the argument in the flip lambda function
    using FlipTraits = detail::FunctionTraits<flipT>;
    using HandleT    = typename FlipTraits::template arg<0>::type;
    static_assert(
        std::is_same_v<HandleT, EdgeHandle>,
        "First argument in flip lambda function should be EdgeHandle");


    if (blockIdx.x >= context.get_num_patches()) {
        return;
    }

    const uint32_t patch_id = blockIdx.x;
    detail::edge_flip_block_dispatcher<blockThreads>(
        context.get_patches_info()[patch_id], flip);
}

}  // namespace rxmesh