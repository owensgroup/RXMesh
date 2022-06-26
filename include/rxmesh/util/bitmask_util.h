#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/macros.h"

namespace rxmesh {
namespace detail {

__host__ __device__ __inline__ bool is_set_bit(const uint16_t  local_id,
                                               const uint32_t* bitmask)
{
    const uint16_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    const uint32_t     mask = bitmask[idx];
    constexpr uint32_t one  = 1;
    return mask & (one << bit);
}

__host__ __device__ __inline__ void bitmask_set_bit(const uint16_t local_id,
                                                    uint32_t*      bitmask)
{
    const uint16_t     idx = local_id / 32;
    const uint32_t     bit = local_id % 32;
    constexpr uint32_t one = 1;
    bitmask[idx] |= (one << bit);
}

__host__ __device__ __inline__ void bitmask_clear_bit(const uint16_t local_id,
                                                      uint32_t*      bitmask)
{
    const uint16_t     idx = local_id / 32;
    const uint32_t     bit = local_id % 32;
    constexpr uint32_t one = 1;
    bitmask[idx] &= ~(one << bit);
}

__host__ __device__ __inline__ void bitmask_flip_bit(const uint16_t local_id,
                                                     uint32_t*      bitmask)
{
    const uint16_t     idx = local_id / 32;
    const uint32_t     bit = local_id % 32;
    constexpr uint32_t one = 1;
    bitmask[idx] ^= (one << bit);
}

__host__ __device__ __inline__ uint32_t mask_num_bytes(const uint32_t size)
{
    return DIVIDE_UP(size, 32) * sizeof(uint32_t);
}

/**
 * @brief Check if an input mesh element is deleted by checking on an input
 * bitmask
 * @param local_id input mesh element
 * @param active_bitmask input bitmask where each mesh element is represented by
 * a single bit. The total size of this bitmask is number of mesh element
 * divided by 64 (rounded up)
 * @return true if the mesh element bit is set to zero
 */
__device__ __host__ __forceinline__ bool is_deleted(
    const uint16_t  local_id,
    const uint32_t* active_bitmask)
{
    return !is_set_bit(local_id, active_bitmask);
}


__device__ __host__ __forceinline__ bool is_owned(const uint16_t  local_id,
                                                  const uint32_t* owned_bitmask)
{
    return is_set_bit(local_id, owned_bitmask);
}


}  // namespace detail
}  // namespace rxmesh