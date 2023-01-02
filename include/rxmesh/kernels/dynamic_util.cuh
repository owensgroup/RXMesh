#pragma once

#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/util.h"

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


__device__ __forceinline__ void warp_update_mask(const bool thread_predicate,
                                                 const uint16_t thread_id,
                                                 uint32_t*      bitmask)
{
    // if thread's thread_predicate in the N-th lane is true, then warp_maks
    // N-th bit will be 1
    __syncwarp();
    uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, thread_predicate);

    // let the thread in first lane writes the new bit mask
    uint32_t lane_id = threadIdx.x % 32;  // 32 = warp size
    if (lane_id == 0 && warp_mask != 0) {
        // here we first need to bitwise invert the warp_mask and then AND
        // it with bitmask
        // Example for 5 threads/faces where the first face/edge/vertex
        // has been deleted before and the second face/edge/vertex is deleted
        // now
        // Bitmask stored in bitmask is: 0 1 1 1 0
        // Bitmask in warp_mask:         0 1 0 0 1
        // The result we want to store:  0 0 1 1 0
        uint32_t mask_id  = thread_id / 32;
        uint32_t old_mask = bitmask[mask_id];
        bitmask[mask_id]  = ((~warp_mask) & old_mask);
    }
}

/**
 * @brief delete mesh element based on a predicate. Deletion here means marking
 * the mesh element as deleted in its bitmask. This function first check if the
 * mesh element is already deleted by checking on the input bitmask
 */
template <uint32_t blockThreads, typename predicateT>
__device__ __inline__ void update_bitmask(uint16_t   range,
                                          uint32_t*  bitmask,
                                          predicateT predicate,
                                          bool check_if_already_deleted = true)
{
    // load over all owned elements -- one thread per element
    // we need to make sure that the whole warp go into the loop
    block_loop<uint16_t, blockThreads, true>(
        range, [&](const uint16_t local_id) {
            // check if mesh element with local_id should be deleted and make
            // sure we only check on range
            bool to_delete = false;
            if (local_id < range) {
                bool already_deleted = false;
                if (check_if_already_deleted) {
                    already_deleted = is_deleted(local_id, bitmask);
                }
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