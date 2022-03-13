#pragma once

#include "rxmesh/kernels/is_deleted.cuh"
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
    const uint16_t num_owned_edges = patch_info.num_owned_edges;


}
}  // namespace detail
}  // namespace rxmesh