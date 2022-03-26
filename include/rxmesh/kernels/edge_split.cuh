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
}

}  // namespace detail
}  // namespace rxmesh