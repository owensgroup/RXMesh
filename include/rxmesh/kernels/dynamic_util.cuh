#pragma once
#include "rxmesh/kernels/is_deleted.cuh"

namespace rxmesh {
namespace detail {
/**
 * @brief utility function that can be used when threads in a block loop over a
 * range to processes a set of items/tasks without warp divergence i.e., if the
 * range is not	multiple of 32, it will be rounded up to next multiple of 32.
 */
template <typename indexT,
          uint32_t blockThreads,
          bool     roundToMultipleOf32,
          typename functionT>
__device__ __inline__ void block_loop(indexT loop_len, functionT func)
{
    if constexpr (roundToMultipleOf32) {
        loop_len = round_to_next_multiple_32(loop_len);
    }

    for (indexT local_id = threadIdx.x; local_id < loop_len;
         local_id += blockThreads) {

        func(local_id);
    }
}


/**
 * @brief delete mesh element based on a predicate. Deletion here means marking
 * the mesh element as deleted in its bitmask. This function first check if the
 * mesh element is already deleted by checking on the input bitmask
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void update_bitmask(uint16_t   num_owned,
                                          uint32_t*  bitmask,
                                          predicateT predicate)
{
    // load over all owned elements -- one thread per element
    // we need to make sure that the whole warp go into the loop
    block_loop<uint16_t, blockThreads, true>(
        num_owned, [&](const uint16_t local_id) {
            // check if mesh element with local_id should be deleted and make
            // sure we only check on num_owned
            bool to_delete = false;
            if (local_id < num_owned) {
                bool already_deleted = is_deleted(local_id, bitmask);
                if (!already_deleted) {
                    if (predicate(local_id)) {
                        to_delete = true;
                    };
                }
            }

            // update the bitmask. This function should be called by the whole
            // warp
            warp_update_mask(to_delete, local_id, bitmask);
        });
}

}  // namespace detail
}  // namespace rxmesh