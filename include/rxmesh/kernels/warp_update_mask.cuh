#pragma once
#include <cuda_runtime.h>
#include <stdint.h>


namespace rxmesh {
namespace detail {

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
        // it with the bitmask stored in global memory
        // Example for 4 threads/faces where the first face/edge/vertex
        // has been deleted before and the second face/edge/vertex is deleted
        // now
        // Bitmask stored in global memory is: 0 1 1 1
        // Bitmask in warp_mask:               0 1 0 0
        // The result we want to store:        0 0 1 1
        uint32_t mask_id  = thread_id / 32;
        uint32_t old_mask = bitmask[mask_id];
        bitmask[mask_id]  = ((~warp_mask) & old_mask);
    }
}
}  // namespace detail
}  // namespace rxmesh