#pragma once

#include <cuco/pair.cuh>
#include <cuco/priority_queue.cuh>


/**
 * @brief Return unique index of the local mesh element composed by the
 * patch id and the local index
 *
 * @param local_id the local within-patch mesh element id
 * @param patch_id the patch owning the mesh element
 * @return
 */
constexpr __device__ __host__ __forceinline__ uint32_t
unique_id32(const uint16_t local_id, const uint16_t patch_id)
{
    uint32_t ret = patch_id;
    ret          = (ret << 16);
    ret |= local_id;
    return ret;
}


/**
 * @brief unpack a 32 uint to its high and low 16 bits.
 * This is used to convert the unique id to its local id (16
 * low bit) and patch id (high 16 bit)
 * @param uid unique id
 * @return a std::pair storing the patch id and local id
 */
constexpr __device__ __host__ __forceinline__ std::pair<uint16_t, uint16_t>
                                              unpack32(uint32_t uid)
{
    uint16_t local_id = uid & ((1 << 16) - 1);
    uint16_t patch_id = uid >> 16;
    return std::make_pair(patch_id, local_id);
}


/**
 * @brief less than operator for std::pair
 * @tparam T
 */
template <typename T>
struct pair_less
{
    __host__ __device__ __forceinline__ bool operator()(const T& a,
                                                        const T& b) const
    {
        return a.first < b.first;
    }
};


// Priority queue setup. Use 'pair_less' to prioritize smaller values.
using PriorityPairT   = cuco::pair<float, uint32_t>;
using PriorityCompare = pair_less<PriorityPairT>;
using PriorityQueueT  = cuco::priority_queue<PriorityPairT, PriorityCompare>;
using PQViewT         = PriorityQueueT::device_mutable_view;
