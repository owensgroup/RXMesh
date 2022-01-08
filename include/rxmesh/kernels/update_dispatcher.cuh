#pragma

#include "rxmesh/context.h"
#include "rxmesh/handle.h"
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