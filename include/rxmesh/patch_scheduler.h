#pragma once

// for debugging, this macro let the scheduler only generate one valid patch
// (corresponding to the blockIdx.x)
// #define PROCESS_SINGLE_PATCH

// inspired/taken from
// https://github.com/GPUPeople/Ouroboros/blob/9153c55abffb3bceb5aea4028dfcc00439b046d5/include/device/queues/Queue.h

#include <cuda_runtime.h>
#include <stdint.h>

namespace rxmesh {

struct PatchScheduler
{
    __device__ __host__ PatchScheduler()
        : count(nullptr), capacity(0), list(nullptr){};
    __device__ __host__ PatchScheduler(const PatchScheduler& other) = default;
    __device__ __host__ PatchScheduler(PatchScheduler&&)            = default;
    __device__ __host__ PatchScheduler& operator=(const PatchScheduler&) =
        default;
    __device__ __host__ PatchScheduler& operator=(PatchScheduler&&) = default;
    __device__                          __host__ ~PatchScheduler()  = default;

    /**
     * @brief add/push new patch of the list
     */
    __device__ bool push(const uint32_t pid);

    /**
     * @brief get a patch from the list. If the list is empty, return INVALID32
     */
    __device__ uint32_t pop();

    /**
     * @brief fill the list by sequential numbers
     */
    __host__ void refill(const uint32_t size);

    /**
     * @brief initialize all the memories
     */
    __host__ void init(uint32_t cap);

    __host__ void print_list() const;

    /**
     * @brief free all the memories
     */
    __host__ void free();

    /**
     * @brief return the size of the queue (that is not the capacity)
     */
    __host__ __device__ int size(cudaStream_t stream = NULL) const;

    /**
     * @brief check if the list empty. On the host, the check need to move data
     * from device to host
     * @return
     */
    __host__ __device__ bool is_empty(cudaStream_t stream = NULL);

    int*      count;
    int*      front;
    int*      back;
    uint32_t  capacity;
    uint32_t* list;
};

}  // namespace rxmesh
