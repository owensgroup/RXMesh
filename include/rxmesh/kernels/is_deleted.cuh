#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace rxmesh {
namespace detail {

/**
 * @brief Check if an input mesh element is deleted by checking on an input
 * bitmask
 * @param local_id input mesh element
 * @param bitmask input bitmask where each mesh element is represented by a
 * single bit. The total size of this bitmask is number of mesh element divided
 * by 64 (rounded up)
 * @return true if the mesh element bit is set to zero
 */
__device__ __host__ __forceinline__ bool is_deleted(const uint16_t& local_id,
                                                    const uint32_t* bitmask)
{

    const uint16_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    const uint32_t     mask = bitmask[idx];
    constexpr uint32_t one  = 1;
    return !(mask & (one << bit));
}

}  // namespace detail
}  // namespace rxmesh