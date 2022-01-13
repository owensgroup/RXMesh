#pragma

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
    const uint16_t num_faces       = patch_info.num_faces;
    const uint16_t num_edges       = patch_info.num_edges;
    const uint16_t num_owned_edges = patch_info.num_owned_edges;

    bool do_flip[EDGE_FLIP_PER_THREAD];

    for (uint32_t i = 0; i < EDGE_FLIP_PER_THREAD; ++i) {
        do_flip[i] = false;
    }

    assert(EDGE_FLIP_PER_THREAD * blockThreads >= num_owned_edges);

    // compute the predicate for each edge assigned to this thread and cache
    // the results in do_flip.
    bool     any_flip = false;
    uint16_t local_id = threadIdx.x;
    uint32_t e        = 0;
    while (local_id < num_owned_edges) {
        do_flip[e] = flip({patch_info.patch_id, local_id});
        any_flip   = any_flip || do_flip[e];

        local_id += blockThreads;
        e++;
    }

    if (__syncthreads_or(any_flip) == 0) {
        return;
    }

    // we have at least one edge to flip in this patch
    extern __shared__ uint16_t shrd_mem[];
    LocalVertexT*              s_ev = reinterpret_cast<LocalVertexT*>(shrd_mem);
    uint16_t*                  s_fe = shrd_mem;
    uint16_t* s_ef = &shrd_mem[3 * num_faces + (3 * num_faces) % 2];

    // TODO fix bank conflicts
    for (uint16_t i = threadIdx.x; i < 2 * num_edges; i += blockThreads) {
        s_ef[i] = INVALID16;
    }

    load_patch_FE<blockThreads>(patch_info,
                                reinterpret_cast<LocalEdgeT*>(s_fe));

    // need a sync here so s_fe and s_ef are initialized before accessing them
    __syncthreads();

    // Transpose FE into EF so we obtain the two incident triangles to the
    // flipped edges. We use the version that is optimized for manifolds
    e_f_manifold<blockThreads>(num_edges, num_faces, s_fe, s_ef);

    local_id = threadIdx.x;
    e        = 0;
    while (local_id < num_owned_edges) {
        if (do_flip[e]) {

            const uint16_t f0 = s_ef[2 * local_id];
            const uint16_t f1 = s_ef[2 * local_id + 1];

            const uint16_t f0_e0 = s_fe[3 * f0];
            const uint16_t f0_e1 = s_fe[3 * f0 + 1];
            const uint16_t f0_e2 = s_fe[3 * f0 + 2];

            const uint16_t l0 = ((f0_e0 >> 1) == local_id) ?
                                    0 :
                                    (((f0_e1 >> 1) == local_id) ? 1 : 2);

            const uint16_t f1_e0 = s_fe[3 * f1];
            const uint16_t f1_e1 = s_fe[3 * f1 + 1];
            const uint16_t f1_e2 = s_fe[3 * f1 + 2];

            const uint16_t l1 = ((f1_e0 >> 1) == local_id) ?
                                    0 :
                                    (((f1_e1 >> 1) == local_id) ? 1 : 2);

            const uint16_t shift = (local_id / 3) * 3;

            //s_fe[shift + ((l0 +1)%3)] = 
        }
        local_id += blockThreads;
        e++;
    }
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