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

    // load the patch
    extern __shared__ uint16_t shrd_mem[];
    LocalVertexT*              s_ev = reinterpret_cast<LocalVertexT*>(shrd_mem);
    uint16_t*                  s_fe = shrd_mem;
    uint16_t* s_ef = &shrd_mem[3 * num_faces + (3 * num_faces) % 2];

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