#include <cuda_runtime.h>
#include <stdint.h>

namespace rxmesh {
namespace detail {

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

    const uint16_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    const uint32_t     mask = active_bitmask[idx];
    constexpr uint32_t one  = 1;
    return !(mask & (one << bit));
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

}  // namespace detail
}  // namespace rxmesh