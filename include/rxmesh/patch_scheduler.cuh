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
        // int id = ::atomicAdd(count, 1);
        //__threadfence();
        // assert(id < capacity);
        // list[id] = pid;

        int fill = ::atomicAdd(count, 1);
        if (fill < static_cast<int>(capacity)) {
            int pos = ::atomicAdd(back, 1);
            while (::atomicCAS(list + pos, INVALID32, pid) != INVALID32) {
                // TODO do we really need to sleep if it is only one thread
                // in the block doing the job??
                //__nanosleep(10);
            }
            return true;
        } else {
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
        // const int id = ::atomicAdd(count, -1);
        //__threadfence();
        // if (id >= 0) {
        //    return list[id];
        //} else {
        //    return INVALID32;
        //}

        int readable = ::atomicSub(count, 1);
        if (readable <= 0) {
            ::atomicAdd(count, 1);
            return INVALID32;
        }

        uint32_t pid = INVALID32;

        int pos = ::atomicAdd(front, 1);

        while (pid == INVALID32) {
            pid = atomicExch(list + pos, INVALID32);
            // TODO do we really need to sleep if it is only one thread
            // in the block doing the job??
            //__nanosleep(10);
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
            cudaMemcpy(front, &capacity, sizeof(int), cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMemset(back, 0, sizeof(int)));
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

    int*      count;
    int*      front;
    int*      back;
    uint32_t  capacity;
    uint32_t* list;
};

}  // namespace rxmesh