#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/macros.h"

namespace rxmesh {
namespace detail {

constexpr __host__ __device__ __inline__ bool is_set_bit(
    const uint32_t  local_id,
    const uint32_t* bitmask)
{
    const uint32_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    const uint32_t     mask = bitmask[idx];
    constexpr uint32_t one  = 1;
    return mask & (one << bit);
}

__host__ __inline__ uint16_t count_set_bits(const uint16_t  size,
                                            const uint32_t* bitmask)
{
    // size here is the number of item represented by this bitmask. So, if
    // the bitmask is a buffer of a single 32-bits, it could represent up to 32
    // items and thus 'size' could be up to 32
    uint16_t sum = 0;
    for (uint16_t i = 0; i < size; ++i) {
        sum += is_set_bit(i, bitmask);
    }
    return sum;
}


__host__ __inline__ uint16_t count_zero_bits(const uint16_t  size,
                                             const uint32_t* bitmask)
{
    // size here is the number of item represented by this bitmask. So, if
    // the bitmask is a buffer of a single 32-bits, it could represent up to 32
    // items and thus 'size' could be up to 32
    uint16_t sum = 0;
    for (uint16_t i = 0; i < size; ++i) {
        sum += !is_set_bit(i, bitmask);
    }
    return sum;
}


__host__ __device__ __inline__ void bitmask_set_bit(const uint16_t local_id,
                                                    uint32_t*      bitmask,
                                                    bool is_atomic = false)
{
    const uint16_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    constexpr uint32_t one  = 1;
    const uint32_t     mask = one << bit;

#ifdef __CUDA_ARCH__
    if (is_atomic) {
        ::atomicOr(bitmask + idx, mask);
        return;
    }
#endif

    bitmask[idx] |= mask;
}

__host__ __device__ __inline__ bool bitmask_try_set_bit(const uint16_t local_id,
                                                        uint32_t*      bitmask)
{
    const uint16_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    constexpr uint32_t one  = 1;
    const uint32_t     mask = one << bit;
    uint32_t           old;

#ifdef __CUDA_ARCH__
    old = ::atomicOr(bitmask + idx, mask);
#else
    old = bitmask[idx];
    bitmask[idx] |= mask;
#endif

    return !(old & mask);
}

__host__ __device__ __inline__ void bitmask_clear_bit(const uint16_t local_id,
                                                      uint32_t*      bitmask,
                                                      bool is_atomic = false)
{
    const uint16_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    constexpr uint32_t one  = 1;
    const uint32_t     mask = ~(one << bit);

#ifdef __CUDA_ARCH__
    if (is_atomic) {
        ::atomicAnd(bitmask + idx, mask);
        return;
    }
#endif
    bitmask[idx] &= mask;
}

__host__ __device__ __inline__ void bitmask_flip_bit(const uint16_t local_id,
                                                     uint32_t*      bitmask,
                                                     bool is_atomic = false)
{
    const uint16_t     idx  = local_id / 32;
    const uint32_t     bit  = local_id % 32;
    constexpr uint32_t one  = 1;
    const uint32_t     mask = one << bit;

#ifdef __CUDA_ARCH__
    if (is_atomic) {
        ::atomicXor(bitmask + idx, mask);
        return;
    }
#endif
    bitmask[idx] ^= mask;
}

constexpr __host__ __device__ __inline__ uint32_t mask_num_bytes(
    const uint32_t size)
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
constexpr __device__ __host__ __forceinline__ bool is_deleted(
    const uint16_t  local_id,
    const uint32_t* active_bitmask)
{
    return !is_set_bit(local_id, active_bitmask);
}


constexpr __device__ __host__ __forceinline__ bool is_owned(
    const uint16_t  local_id,
    const uint32_t* owned_bitmask)
{
    return is_set_bit(local_id, owned_bitmask);
}

template <int NumBits, typename T>
constexpr __device__ __host__ __forceinline__ T extract_low_bits(const T input)
{
    static_assert(std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>);

    return input & ((1 << NumBits) - 1);
}

template <int NumBits, typename T>
constexpr __device__ __host__ __forceinline__ T extract_high_bits(const T input)
{
    static_assert(std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>);

    return input >> (sizeof(T) * 8 - NumBits);
}

}  // namespace detail
}  // namespace rxmesh