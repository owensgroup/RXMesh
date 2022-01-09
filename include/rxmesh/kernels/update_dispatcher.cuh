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
    bool do_flip[EDGE_FLIP_PER_THREAD];

    for (uint32_t i = 0; i < EDGE_FLIP_PER_THREAD; ++i) {
        do_flip[i] = false;
    }

    assert(EDGE_FLIP_PER_THREAD * blockThreads >= patch_info.num_owned_edges);

    // compute the predicate for each edge assigned to this thread and cache
    // the results in do_flip.
    bool     any_flip = false;
    uint16_t local_id = threadIdx.x;
    uint32_t e        = 0;
    while (local_id < patch_info.num_owned_edges) {
        do_flip[e] = flip({patch_info.patch_id, local_id});
        any_flip   = any_flip || do_flip[e];

        local_id += blockThreads;
        e++;
    }

    if (__syncthreads_or(any_flip) == 0) {
        return;
    }

    // we have at least on edge to flip in this patch
    extern __shared__ uint16_t shrd_mem[];
    LocalVertexT*              s_ev = reinterpret_cast<LocalVertexT*>(shrd_mem);
    LocalEdgeT*                s_fe = reinterpret_cast<LocalEdgeT*>(shrd_mem);
    uint16_t*                  s_ef = reinterpret_cast<uint16_t*>(
        &shrd_mem[3 * patch_info.num_faces + (3 * patch_info.num_faces) % 2]);

    // first load and update
    load_patch_FE<blockThreads>(patch_info, s_fe);

    // 1. transpose FE into EF so we obtain the two incident triangles to the
    // flipped edges. We can optimize this for manifold mesh
    for (uint16_t e = threadIdx.x; e < 3 * patch_info.num_faces;
         e += blockThreads) {
        uint16_t edge    = s_fe[e].id >> 1;
        uint16_t face_id = e / 3;

        auto ret = atomicCAS(s_ef + 2 * edge, INVALID16, face_id);
        if (ret != INVALID16) {
            ret = atomicCAS(s_ef + 2 * edge + 1, INVALID16, face_id);
            assert(ret == INVALID16);
        }
    }

    local_id = threadIdx.x;
    e        = 0;
    while (local_id < patch_info.num_owned_edges) {
        if (do_flip[e]) {
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