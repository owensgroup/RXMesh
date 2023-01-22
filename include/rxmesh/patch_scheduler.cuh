#pragma once
// inpsired/taken from
// https://github.com/GPUPeople/Ouroboros/blob/9153c55abffb3bceb5aea4028dfcc00439b046d5/include/device/queues/Queue.h

#include "rxmesh/util/util.h"

namespace rxmesh {
struct PatchScheduler
{
    __device__ __host__ PatchScheduler()
        : list(nullptr), count(nullptr), capacity(0){};
    __device__ __host__ PatchScheduler(const PatchScheduler& other) = default;
    __device__ __host__ PatchScheduler(PatchScheduler&&)            = default;
    __device__ __host__ PatchScheduler& operator=(const PatchScheduler&) =
        default;
    __device__ __host__ PatchScheduler& operator=(PatchScheduler&&) = default;
    __device__                          __host__ ~PatchScheduler()  = default;


    /**
     * @brief add/push new patch of the list
     */
    __device__ __inline__ bool push(const uint32_t pid)
    {
#ifdef __CUDA_ARCH__
        if (::atomicAdd(count, 1) < static_cast<int>(capacity)) {
            int pos = ::atomicAdd(back, 1) % capacity;

            // the loop because another thread/block may have just decremented
            // the count but has not yet finish reading from the list
            while (::atomicCAS(list + pos, INVALID32, pid) != INVALID32) {
                // TODO do we really need to sleep if it is only one thread
                // in the block doing the job??
                //__nanosleep(10);
            }
            return true;
        } else {
            // for our configuration, this should not happen since a block only
            // pop a patch and, if not able to process it due to dependency
            // conflict, the block push the same patch again. So at all times
            // count is less than capacity if capacity is total number of
            // patches
            return false;
        }


#endif
    }

    /**
     * @brief get a patch from the list. If the list is empty, return INVALID32
     */
    __device__ __inline__ uint32_t pop()
    {
#ifdef __CUDA_ARCH__
        int readable = ::atomicSub(count, 1);

        uint32_t pid = INVALID32;

        if (readable <= 0) {
            ::atomicAdd(count, 1);
        } else {

            int pos = ::atomicAdd(front, 1) % capacity;

            // the loop because another thread/block may have just incremented
            // the count but has not yet wrote to the list
            while (pid == INVALID32) {
                pid = atomicExch(list + pos, INVALID32);
                // TODO do we really need to sleep if it is only one thread
                // in the block doing the job??
                //__nanosleep(10);
            }
        }
        return pid;

#endif
    }

    /**
     * @brief fill the list by sequential numbers
     */
    __host__ void refill()
    {
        std::vector<uint32_t> h_list(capacity);
        fill_with_sequential_numbers(h_list.data(), capacity);
        CUDA_ERROR(cudaMemcpy(list,
                              h_list.data(),
                              h_list.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_ERROR(
            cudaMemcpy(count, &capacity, sizeof(int), cudaMemcpyHostToDevice));

        CUDA_ERROR(
            cudaMemcpy(back, &capacity, sizeof(int), cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMemset(front, 0, sizeof(int)));
    }

    /**
     * @brief initialize all the memories
     */
    __host__ void init(uint32_t num_patches)
    {
        capacity = num_patches;
        CUDA_ERROR(cudaMalloc((void**)&count, sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&front, sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&back, sizeof(int)));
        CUDA_ERROR(cudaMalloc((void**)&list, sizeof(uint32_t) * capacity));
    }

    /**
     * @brief free all the memories
     */
    __host__ void free()
    {
        GPU_FREE(count);
        GPU_FREE(front);
        GPU_FREE(back);
        GPU_FREE(list);
    }

    /**
     * @brief check if the list empty. On the host, the check need to move data
     * from device to host
     * @return
     */
    __host__ __device__ __inline__ bool is_empty(cudaStream_t stream = NULL)
    {
#ifdef __CUDA_ARCH__
        return count[0] == 0;
#else
        int h_count = 0;
        CUDA_ERROR(cudaMemcpyAsync(
            &h_count, count, sizeof(int), cudaMemcpyDeviceToHost, stream));
        return h_count == 0;
#endif
    }

    int*      count;
    int*      front;
    int*      back;
    uint32_t  capacity;
    uint32_t* list;
};

}  // namespace rxmesh